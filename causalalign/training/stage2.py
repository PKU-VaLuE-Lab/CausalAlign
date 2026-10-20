from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from causalalign.config import load_config
from causalalign.data import RecordDataset
from causalalign.losses import stage2_objective
from causalalign.models.projector import ProjectorBank
from causalalign.spec import PAPER_SPEC
from causalalign.training.common import (
    adamw,
    atomic_checkpoint,
    autocast,
    checkpoint_payload,
    cosine_warmup_scheduler,
    distributed_mean,
    endless,
    loader,
    log_metrics,
    setup,
    to_device,
    validate_targets,
)


def _targets(config: dict) -> tuple[list[int], list[int], dict[int, float]]:
    target = config["targets"]
    layers = [int(x) for x in target["layers"]]
    progress = [int(x) for x in target["progress"]]
    weights = {int(key): float(value) for key, value in target["progress_weights"].items()}
    validate_targets(layers, progress, weights)
    return layers, progress, weights


@torch.no_grad()
def validate(model, data, device, amp, layers, progress, weights, max_batches=32):
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    for batch in data:
        batch = to_device(batch, device)
        with autocast(device, amp):
            _, metrics = stage2_objective(model, batch["transitions"], batch["teacher_token"], layers, progress, weights)
        total += metrics["loss"]
        count += 1
        if count >= max_batches:
            break
    model.train()
    return distributed_mean(total / max(count, 1))


def train(config: dict) -> Path:
    device, rank, world = setup(int(config["seed"]))
    layers, progress, weights = _targets(config)
    teacher = torch.load(config["teacher_checkpoint"], map_location="cpu", weights_only=False)
    if teacher.get("stage") != 1 or teacher.get("paper_spec_sha256") != PAPER_SPEC.fingerprint():
        raise ValueError("teacher checkpoint is not a compatible paper Stage-1 checkpoint")
    train_data = RecordDataset(config["data"]["train_index"])
    val_data = RecordDataset(config["data"]["val_index"])
    opt_cfg, model_cfg = config["optimization"], config["model"]
    train_iter = endless(loader(train_data, int(opt_cfg["batch_size"]), world, rank))
    val_loader = loader(val_data, int(opt_cfg["batch_size"]), world, rank, shuffle=False)
    base = ProjectorBank(layers, **model_cfg).to(device)
    model: torch.nn.Module = DDP(base, device_ids=[device.index]) if world > 1 else base
    optimizer = adamw(
        model.parameters(),
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        fused=bool(opt_cfg.get("fused_adamw", True)),
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_steps=int(opt_cfg["warmup_steps"]),
        total_steps=int(opt_cfg["max_updates"]),
        minimum_ratio=float(opt_cfg.get("minimum_lr_ratio", 0.0)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and opt_cfg["amp"] == "fp16")
    output = Path(config["output_dir"]).resolve()
    best = float("inf")
    accumulation = int(opt_cfg["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(opt_cfg["max_updates"]) + 1):
        combined: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            batch = to_device(next(train_iter), device)
            sync = nullcontext() if world == 1 or micro == accumulation - 1 else model.no_sync()
            with sync, autocast(device, str(opt_cfg["amp"])):
                loss, metrics = stage2_objective(model, batch["transitions"], batch["teacher_token"], layers, progress, weights)
                loss = loss / accumulation
            scaler.scale(loss).backward()
            for key, value in metrics.items():
                combined[key] = combined.get(key, torch.zeros_like(value)) + value / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(opt_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        combined["learning_rate"] = torch.tensor(scheduler.get_last_lr()[0], device=device)
        if step % int(opt_cfg["log_every"]) == 0:
            log_metrics(step, combined, rank)
        if step % int(opt_cfg["save_every"]) == 0 and rank == 0:
            atomic_checkpoint(output / f"step-{step:06d}.pt", checkpoint_payload(2, step, base, optimizer, config, scheduler))
        if step % int(opt_cfg["validate_every"]) == 0:
            val_loss = validate(model, val_loader, device, str(opt_cfg["amp"]), layers, progress, weights)
            log_metrics(step, {"val/loss": val_loss}, rank)
            if rank == 0 and float(val_loss) < best:
                best = float(val_loss)
                atomic_checkpoint(output / "best.pt", checkpoint_payload(2, step, base, optimizer, config, scheduler))
    if rank == 0:
        atomic_checkpoint(output / "last.pt", checkpoint_payload(2, int(opt_cfg["max_updates"]), base, optimizer, config, scheduler))
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

