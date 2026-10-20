from __future__ import annotations

from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from causalalign.data import tensor_collate
from causalalign.spec import PAPER_SPEC


def setup(seed: int) -> tuple[torch.device, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return device, rank, world


def loader(dataset: Any, batch_size: int, world: int, rank: int, shuffle: bool = True) -> DataLoader:
    sampler = DistributedSampler(dataset, world, rank, shuffle=shuffle) if world > 1 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=torch.cuda.is_available(),
        collate_fn=tensor_collate,
        drop_last=shuffle,
    )


def endless(data: DataLoader):
    epoch = 0
    while True:
        if isinstance(data.sampler, DistributedSampler):
            data.sampler.set_epoch(epoch)
        yield from data
        epoch += 1


def to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


def autocast(device: torch.device, amp: str):
    if device.type != "cuda" or amp == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    stage: int,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    config: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    return {
        "format": "causalalign-paper-v1",
        "paper_spec_sha256": PAPER_SPEC.fingerprint(),
        "stage": stage,
        "step": step,
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    }


def adamw(parameters, *, lr: float, weight_decay: float, fused: bool = True) -> torch.optim.AdamW:
    """Construct AdamW with the CUDA fused kernel when the runtime supports it."""
    kwargs: dict[str, Any] = {"lr": lr, "weight_decay": weight_decay}
    if fused and torch.cuda.is_available():
        kwargs["fused"] = True
    try:
        return torch.optim.AdamW(parameters, **kwargs)
    except (TypeError, RuntimeError):
        kwargs.pop("fused", None)
        return torch.optim.AdamW(parameters, **kwargs)


def cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    minimum_ratio: float = 0.0,
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
        raise ValueError("require 0 <= warmup_steps < total_steps")

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return max(1.0 / max(1, warmup_steps), float(step + 1) / max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def two_phase_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    phase1_steps: int,
    total_steps: int,
    phase1_warmup_steps: int,
    phase2_warmup_steps: int,
    phase2_peak_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Two independent warmup/cosine phases used by the recovered Stage-1 recipe."""
    if not 0 < phase1_steps < total_steps:
        raise ValueError("phase1_steps must lie inside the training run")
    if not 0 < phase2_peak_ratio <= 1:
        raise ValueError("phase2_peak_ratio must be in (0, 1]")

    def phase_curve(position: int, length: int, warmup: int, peak: float) -> float:
        if position < warmup:
            return peak * max(1.0 / max(1, warmup), float(position + 1) / max(1, warmup))
        progress = min(1.0, (position - warmup) / max(1, length - warmup - 1))
        return peak * 0.5 * (1.0 + math.cos(math.pi * progress))

    def multiplier(step: int) -> float:
        if step < phase1_steps:
            return phase_curve(step, phase1_steps, phase1_warmup_steps, 1.0)
        return phase_curve(step - phase1_steps, total_steps - phase1_steps, phase2_warmup_steps, phase2_peak_ratio)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def log_metrics(step: int, metrics: dict[str, torch.Tensor], rank: int) -> None:
    if rank != 0:
        return
    value = {"step": step, **{key: float(item.detach().float().mean().cpu()) for key, item in metrics.items()}}
    print(json.dumps(value, sort_keys=True), flush=True)


def distributed_mean(value: torch.Tensor) -> torch.Tensor:
    value = value.detach().clone()
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
        value /= torch.distributed.get_world_size()
    return value


def validate_targets(layers: list[int], progress: list[int], weights: dict[int, float]) -> None:
    if tuple(layers) != PAPER_SPEC.target_layers:
        raise ValueError(f"paper target layers must be {PAPER_SPEC.target_layers}, got {layers}")
    if tuple(progress) != PAPER_SPEC.target_progress:
        raise ValueError(f"paper target progress must be {PAPER_SPEC.target_progress}, got {progress}")
    if set(weights) != set(progress) or abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("progress_weights must cover selected progress points and sum to one")

