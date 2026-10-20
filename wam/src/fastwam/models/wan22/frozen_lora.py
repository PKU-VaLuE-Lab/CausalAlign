from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class FrozenLoRALinear(nn.Module):
    """A trainable base Linear plus a fixed, full-precision LoRA residual."""

    def __init__(
        self,
        base: nn.Linear,
        lora_a: torch.Tensor,
        lora_b: torch.Tensor,
        scaling: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = int(lora_a.shape[0])
        self.scaling = float(scaling)
        self.register_buffer("lora_A", lora_a.detach().to(device=base.weight.device, dtype=torch.float32))
        self.register_buffer("lora_B", lora_b.detach().to(device=base.weight.device, dtype=torch.float32))

    def _apply(self, fn, recurse=True):
        # DeepSpeed casts the model to bf16; retain the original FP32 residual.
        lora_a, lora_b = self.lora_A, self.lora_B
        super()._apply(fn, recurse=recurse)
        self.lora_A = lora_a.to(device=self.base.weight.device, dtype=torch.float32)
        self.lora_B = lora_b.to(device=self.base.weight.device, dtype=torch.float32)
        return self

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        base = self.base(value)
        with torch.autocast(device_type=value.device.type, enabled=False):
            delta = F.linear(F.linear(value.float(), self.lora_A), self.lora_B)
        return base + delta.to(dtype=base.dtype) * self.scaling


def _resolve_checkpoint(path: str | Path) -> tuple[Path, Path]:
    source = Path(path).expanduser().resolve()
    if source.is_dir():
        return source / "model.safetensors", source / "checkpoint.json"
    return source, source.with_name("checkpoint.json")


def apply_frozen_lora(video_expert: nn.Module, checkpoint: str | Path, *, resize_action: bool = False) -> dict:
    weights_path, metadata_path = _resolve_checkpoint(checkpoint)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    adapter_state = load_file(str(weights_path), device="cpu")

    rank = int(metadata["lora"]["rank"])
    alpha = float(metadata["lora"]["alpha"])
    scaling = alpha / rank
    modules = dict(video_expert.named_modules())
    applied = []
    skipped_zero = []

    for a_key in sorted(key for key in adapter_state if key.endswith(".lora_A")):
        stem = a_key[: -len(".lora_A")]
        b_key = stem + ".lora_B"
        lora_a = adapter_state[a_key]
        lora_b = adapter_state[b_key]
        if int(torch.count_nonzero(lora_b)) == 0:
            skipped_zero.append(stem)
            continue

        base = modules[stem]
        module_scaling = scaling
        if resize_action:
            # FastWAM derives the action Q/V by interpolating input width and
            # applying sqrt(d_video / d_action). Interpolation is linear, so
            # B @ resize(A) preserves the same adapted initialization in FP32.
            source_width = lora_a.shape[1]
            lora_a = F.interpolate(
                lora_a.float().unsqueeze(0), size=base.in_features,
                mode="linear", align_corners=True,
            ).squeeze(0)
            module_scaling *= (source_width / base.in_features) ** 0.5
        if not isinstance(base, nn.Linear):
            raise TypeError(f"Expected nn.Linear at {stem}, got {type(base).__name__}")
        if tuple(lora_a.shape) != (rank, base.in_features):
            raise ValueError(
                f"LoRA A shape mismatch at {stem}: {tuple(lora_a.shape)} vs "
                f"{(rank, base.in_features)}"
            )
        if tuple(lora_b.shape) != (base.out_features, rank):
            raise ValueError(
                f"LoRA B shape mismatch at {stem}: {tuple(lora_b.shape)} vs "
                f"{(base.out_features, rank)}"
            )

        parent_name, child_name = stem.rsplit(".", 1)
        parent = modules[parent_name]
        setattr(parent, child_name, FrozenLoRALinear(base, lora_a, lora_b, module_scaling))
        applied.append(stem)

    digest = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    expected_digest = metadata.get("weights_sha256")
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            f"Adapter checksum mismatch: expected {expected_digest}, got {digest}"
        )
    return {
        "checkpoint": str(weights_path),
        "sha256": digest,
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "resize_action": resize_action,
        "applied_modules": applied,
        "skipped_zero_modules": skipped_zero,
    }
