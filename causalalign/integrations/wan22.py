from __future__ import annotations

from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from causalalign.models.projector import temporal_transition


class _Captured(Exception):
    pass


@contextmanager
def capture_layers(transformer: nn.Module, layers: Iterable[int], detach: bool):
    requested = tuple(sorted(set(int(x) for x in layers)))
    blocks = getattr(transformer, "blocks", None)
    if blocks is None or not requested or requested[-1] >= len(blocks):
        raise ValueError(f"invalid Wan layer set {requested}")
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def hook(layer: int):
        def inner(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            captured[layer] = value.detach().clone() if detach else value
            if layer == requested[-1]:
                raise _Captured
        return inner

    for layer in requested:
        handles.append(blocks[layer].register_forward_hook(hook(layer)))
    try:
        try:
            yield captured
        except _Captured:
            pass
    finally:
        for handle in handles:
            handle.remove()


def reshape_hidden(hidden: torch.Tensor, latent: torch.Tensor, patch_size: Iterable[int]) -> torch.Tensor:
    patch = tuple(int(x) for x in patch_size)
    if len(patch) == 2:
        patch = (1, *patch)
    pt, ph, pw = patch
    frames, height, width = latent.shape[2] // pt, latent.shape[3] // ph, latent.shape[4] // pw
    if hidden.shape[1] != frames * height * width:
        raise ValueError(f"hidden token count {hidden.shape[1]} != {frames}x{height}x{width}")
    return hidden.reshape(hidden.shape[0], frames, height * width, hidden.shape[-1])


def flow_noise(clean: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    while sigma.ndim < clean.ndim:
        sigma = sigma.unsqueeze(-1)
    return (1 - sigma.to(clean)) * clean + sigma.to(clean) * noise


class Wan22TransitionBackend(nn.Module):
    """Differentiable Wan2.2 hidden-state backend used by paper Stage 3."""

    def __init__(self, model_path: str, expert: str = "transformer_2", dtype: str = "bf16", device: str = "cuda"):
        super().__init__()
        try:
            from diffusers import UniPCMultistepScheduler, WanTransformer3DModel
        except ImportError as exc:
            raise RuntimeError("install the optional Wan dependencies: pip install -e '.[wan]'") from exc
        torch_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
        local = Path(model_path).expanduser().is_dir()
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_path,
            subfolder=expert,
            torch_dtype=torch_dtype,
            local_files_only=local,
            low_cpu_mem_usage=True,
        ).to(device)
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_path, subfolder="scheduler", local_files_only=local
        )
        self.expert_name = expert
        self.device_name = device
        self.transformer.requires_grad_(False).eval()

    def enable_gradient_checkpointing(self) -> None:
        method = getattr(self.transformer, "enable_gradient_checkpointing", None)
        if method is not None:
            method()

    def scheduler_point(self, progress: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Map paper denoising progress to the native Wan timestep and sigma.

        Progress indexes the descending 1000-step scheduler: smaller indices
        are earlier, higher-noise states. It is not a native timestep value.
        """
        self.scheduler.set_timesteps(1000, device=device)
        index = int(progress)
        if index != progress or not 0 <= index < len(self.scheduler.timesteps):
            raise ValueError(f"denoising progress must be a scheduler index in [0, {len(self.scheduler.timesteps)}), got {progress}")
        if self.scheduler.timesteps[0] <= self.scheduler.timesteps[-1]:
            raise RuntimeError("Wan scheduler timesteps must descend in denoising order")
        timestep = self.scheduler.timesteps[index].to(device)
        if hasattr(self.scheduler, "sigmas") and len(self.scheduler.sigmas) > index:
            sigma = self.scheduler.sigmas[index].to(device=device, dtype=torch.float32)
        else:
            sigma = timestep.float() / float(self.scheduler.config.num_train_timesteps)
        return timestep, sigma.clamp(0, 1)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        layers: list[int],
        progress: list[int],
        detach: bool = False,
        select_pairs: bool = True,
    ) -> dict[str, torch.Tensor]:
        clean = batch["clean_latent"]
        noise = batch.get("noise")
        if noise is None or not detach:
            noise = torch.randn_like(clean)
        prompt = batch["prompt_embedding"]
        condition = batch.get("condition_latent")
        image_embedding = batch.get("image_embedding")
        result: dict[str, torch.Tensor] = {}
        for step in progress:
            timestep, sigma = self.scheduler_point(step, clean.device)
            noisy = flow_noise(clean, noise, sigma)
            hidden_input = torch.cat((noisy, condition), 1) if condition is not None else noisy
            kwargs: dict[str, Any] = {
                "hidden_states": hidden_input.to(dtype=self.transformer.dtype),
                "timestep": timestep.expand(clean.shape[0]),
                "encoder_hidden_states": prompt.to(dtype=self.transformer.dtype),
                "return_dict": True,
            }
            if image_embedding is not None:
                kwargs["encoder_hidden_states_image"] = image_embedding.to(dtype=self.transformer.dtype)
            cache = self.transformer.cache_context("cond") if hasattr(self.transformer, "cache_context") else nullcontext()
            with cache, capture_layers(self.transformer, layers, detach=detach) as captured:
                self.transformer(**kwargs)
            for layer in layers:
                hidden = reshape_hidden(captured[layer], clean, self.transformer.config.patch_size)
                transition = temporal_transition(hidden)
                if select_pairs:
                    pair = batch["pair_index"].long().reshape(-1)
                    transition = transition[torch.arange(transition.shape[0], device=transition.device), pair]
                result[f"l{layer}_s{step}"] = transition
        return result

