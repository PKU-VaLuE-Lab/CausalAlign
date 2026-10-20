from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from causalalign.config import load_config
from causalalign.data import RecordDataset
from causalalign.losses import stage1_objective
from causalalign.models.teacher import StructuredDeltaTokenModel
from causalalign.spec import PAPER_SPEC
from causalalign.training.common import (
    adamw,
    atomic_checkpoint,
    autocast,
    checkpoint_payload,
    distributed_mean,
    endless,
    loader,
    log_metrics,
    setup,
    to_device,
    two_phase_cosine_scheduler,
)


@torch.no_grad()
def validate(model: torch.nn.Module, data, device, amp: str, weights: dict[str, float], max_batches: int = 32) -> dict[str, torch.Tensor]:
    model.eval()
    totals: dict[str, torch.Tensor] = {}
    count = 0
    for batch in data:
        batch = to_device(batch, device)
        with autocast(device, amp):
            _, metrics = stage1_objective(model, batch, weights)
        for key, value in metrics.items():
            totals[key] = totals.get(key, torch.zeros_like(value)) + value
        count += 1
        if count >= max_batches:
            break
    model.train()
    return {key: distributed_mean(value / max(count, 1)) for key, value in totals.items()}


def train(config: dict) -> Path:
    device, rank, world = setup(int(config["seed"]))
    opt_cfg, model_cfg, loss_cfg = config["optimization"], config["model"], config["loss"]
    if int(opt_cfg["max_updates"]) != PAPER_SPEC.stage1_updates:
        raise ValueError(f"paper Stage 1 requires {PAPER_SPEC.stage1_updates} updates")
    if float(loss_cfg["lambda_temporal"]) != PAPER_SPEC.lambda_temporal or float(loss_cfg["lambda_pose"]) != PAPER_SPEC.lambda_teacher_pose:
        raise ValueError("Stage-1 loss weights differ from the paper")
    warm = RecordDataset(config["data"]["train_index"], max_scenes=PAPER_SPEC.stage1_warmup_scenes)
    full = RecordDataset(config["data"]["train_index"], max_scenes=PAPER_SPEC.stage1_full_scenes)
    available_scenes = {str(row["scene_id"]) for row in RecordDataset(config["data"]["train_index"]).rows}
    if len(available_scenes) < PAPER_SPEC.stage1_full_scenes:
        raise ValueError(
            f"paper Stage 1 requires {PAPER_SPEC.stage1_full_scenes} scenes after warmup; "
            f"the index contains {len(available_scenes)}"
        )
    val = RecordDataset(config["data"]["val_index"])
    warm_iter = endless(loader(warm, int(opt_cfg["batch_size"]), world, rank))
    full_iter = endless(loader(full, int(opt_cfg["batch_size"]), world, rank))
    val_loader = loader(val, int(opt_cfg["batch_size"]), world, rank, shuffle=False)
    base = StructuredDeltaTokenModel(**model_cfg).to(device)
    model: torch.nn.Module = DDP(base, device_ids=[device.index], find_unused_parameters=False) if world > 1 else base
    phase1_lr = float(opt_cfg["phase1_learning_rate"])
    phase2_lr = float(opt_cfg["phase2_learning_rate"])
    optimizer = adamw(
        model.parameters(),
        lr=phase1_lr,
        weight_decay=float(opt_cfg["weight_decay"]),
        fused=bool(opt_cfg.get("fused_adamw", True)),
    )
    scheduler = two_phase_cosine_scheduler(
        optimizer,
        phase1_steps=PAPER_SPEC.stage1_warmup_updates,
        total_steps=int(opt_cfg["max_updates"]),
        phase1_warmup_steps=int(opt_cfg["phase1_warmup_steps"]),
        phase2_warmup_steps=int(opt_cfg["phase2_warmup_steps"]),
        phase2_peak_ratio=phase2_lr / phase1_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and opt_cfg["amp"] == "fp16")
    output = Path(config["output_dir"]).resolve()
    best = float("inf")
    accumulation = int(opt_cfg["gradient_accumulation"])
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(opt_cfg["max_updates"]) + 1):
        source = warm_iter if step <= PAPER_SPEC.stage1_warmup_updates else full_iter
        combined: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            batch = to_device(next(source), device)
            sync = torch.enable_grad() if world == 1 or micro == accumulation - 1 else model.no_sync()
            with sync, autocast(device, str(opt_cfg["amp"])):
                loss, metrics = stage1_objective(model, batch, loss_cfg)
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
            atomic_checkpoint(output / f"step-{step:06d}.pt", checkpoint_payload(1, step, base, optimizer, config, scheduler))
        if step % int(opt_cfg["validate_every"]) == 0:
            val_metrics = validate(model, val_loader, device, str(opt_cfg["amp"]), loss_cfg)
            log_metrics(step, {f"val/{key}": value for key, value in val_metrics.items()}, rank)
            if rank == 0 and float(val_metrics["loss"]) < best:
                best = float(val_metrics["loss"])
                atomic_checkpoint(output / "best.pt", checkpoint_payload(1, step, base, optimizer, config, scheduler))
    if rank == 0:
        atomic_checkpoint(output / "last.pt", checkpoint_payload(1, int(opt_cfg["max_updates"]), base, optimizer, config, scheduler))
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

