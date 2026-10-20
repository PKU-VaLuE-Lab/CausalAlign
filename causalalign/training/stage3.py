from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from causalalign.config import load_config
from causalalign.data import RecordDataset
from causalalign.integrations.wan22 import Wan22TransitionBackend
from causalalign.losses import stage3_objective
from causalalign.models.lora import freeze_except_lora, inject_lora, lora_state_dict
from causalalign.models.projector import ProjectorBank
from causalalign.models.teacher import StructuredDeltaTokenModel
from causalalign.spec import PAPER_SPEC
from causalalign.training.common import (
    adamw,
    atomic_checkpoint,
    autocast,
    cosine_warmup_scheduler,
    distributed_mean,
    endless,
    loader,
    log_metrics,
    setup,
    to_device,
    validate_targets,
)


def load_frozen_modules(config: dict, device: torch.device, layers: list[int]):
    teacher_ckpt = torch.load(config["teacher_checkpoint"], map_location="cpu", weights_only=False)
    projector_ckpt = torch.load(config["projector_checkpoint"], map_location="cpu", weights_only=False)
    if teacher_ckpt.get("stage") != 1 or projector_ckpt.get("stage") != 2:
        raise ValueError("Stage 3 needs compatible Stage-1 and Stage-2 checkpoints")
    if teacher_ckpt.get("paper_spec_sha256") != PAPER_SPEC.fingerprint() or projector_ckpt.get("paper_spec_sha256") != PAPER_SPEC.fingerprint():
        raise ValueError("checkpoint paper specification differs")
    teacher = StructuredDeltaTokenModel(**teacher_ckpt["config"]["model"]).to(device)
    teacher.load_state_dict(teacher_ckpt["model"], strict=True)
    projector_config = projector_ckpt["config"]["model"]
    projectors = ProjectorBank(layers, **projector_config).to(device)
    projectors.load_state_dict(projector_ckpt["model"], strict=True)
    teacher.requires_grad_(False).eval()
    projectors.requires_grad_(False).eval()
    return teacher.camera_decoder, projectors


def save_lora(path: Path, step: int, backend: Wan22TransitionBackend, optimizer, scheduler, config, injected):
    atomic_checkpoint(path, {
        "format": "causalalign-paper-lora-v1",
        "paper_spec_sha256": PAPER_SPEC.fingerprint(),
        "stage": 3,
        "step": step,
        "lora": lora_state_dict(backend.transformer),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "injected_modules": injected,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
    })


@torch.no_grad()
def validate(model, projectors, camera_decoder, data, device, amp, layers, progress, weights, lambda_delta, max_batches=8):
    model.eval()
    total = torch.zeros((), device=device)
    count = 0
    for batch in data:
        batch = to_device(batch, device)
        with autocast(device, amp):
            transitions = model(batch, layers, progress, False, True)
            _, metrics = stage3_objective(projectors, camera_decoder, transitions, batch["teacher_token"], batch["relative_pose"], batch["scene_scale"], layers, progress, weights, lambda_delta)
        total += metrics["loss"]
        count += 1
        if count >= max_batches:
            break
    model.train()
    return distributed_mean(total / max(count, 1))


def train(config: dict) -> Path:
    device, rank, world = setup(int(config["seed"]))
    target = config["targets"]
    layers = [int(x) for x in target["layers"]]
    progress = [int(x) for x in target["progress"]]
    weights = {int(key): float(value) for key, value in target["progress_weights"].items()}
    validate_targets(layers, progress, weights)
    lambda_delta = float(config["loss"]["lambda_delta"])
    if lambda_delta != PAPER_SPEC.lambda_delta:
        raise ValueError(f"paper lambda_delta is {PAPER_SPEC.lambda_delta}")
    experts = list(config["lora"]["experts"])
    if experts != ["transformer_2"]:
        raise ValueError("paper release trains the selected layers in Wan2.2 transformer_2")
    camera_decoder, projectors = load_frozen_modules(config, device, layers)
    backend = Wan22TransitionBackend(config["wan"]["model_path"], expert=experts[0], dtype=config["wan"]["dtype"], device=str(device))
    lora = config["lora"]
    injected = inject_lora(backend.transformer, layers, lora["modules"], int(lora["rank"]), float(lora["alpha"]), float(lora["dropout"]))
    freeze_except_lora(backend.transformer)
    if config["wan"].get("gradient_checkpointing", False):
        backend.enable_gradient_checkpointing()
    model: torch.nn.Module = DDP(backend, device_ids=[device.index], find_unused_parameters=False) if world > 1 else backend
    parameters = [value for value in model.parameters() if value.requires_grad]
    if not parameters:
        raise RuntimeError("LoRA injection produced no trainable parameters")
    opt_cfg = config["optimization"]
    optimizer = adamw(
        parameters,
        lr=float(opt_cfg["learning_rate"]),
        weight_decay=float(opt_cfg["weight_decay"]),
        fused=bool(opt_cfg.get("fused_adamw", True)),
    )
    scheduler = cosine_warmup_scheduler(
        optimizer,
        warmup_steps=int(opt_cfg["warmup_steps"]),
        total_steps=int(opt_cfg["max_updates"]),
        minimum_ratio=float(opt_cfg.get("minimum_lr_ratio", 0.1)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and opt_cfg["amp"] == "fp16")
    train_data = RecordDataset(config["data"]["train_index"])
    val_data = RecordDataset(config["data"]["val_index"])
    train_iter = endless(loader(train_data, int(opt_cfg["batch_size"]), world, rank))
    val_loader = loader(val_data, int(opt_cfg["batch_size"]), world, rank, shuffle=False)
    accumulation = int(opt_cfg["gradient_accumulation"])
    output = Path(config["output_dir"]).resolve()
    best = float("inf")
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, int(opt_cfg["max_updates"]) + 1):
        combined: dict[str, torch.Tensor] = {}
        for micro in range(accumulation):
            batch = to_device(next(train_iter), device)
            sync = nullcontext() if world == 1 or micro == accumulation - 1 else model.no_sync()
            with sync, autocast(device, str(opt_cfg["amp"])):
                transitions = model(batch, layers, progress, False, True)
                loss, metrics = stage3_objective(projectors, camera_decoder, transitions, batch["teacher_token"], batch["relative_pose"], batch["scene_scale"], layers, progress, weights, lambda_delta)
                loss = loss / accumulation
            scaler.scale(loss).backward()
            for key, value in metrics.items():
                combined[key] = combined.get(key, torch.zeros_like(value)) + value / accumulation
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, float(opt_cfg["grad_clip"]))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        combined["learning_rate"] = torch.tensor(scheduler.get_last_lr()[0], device=device)
        if step % int(opt_cfg["log_every"]) == 0:
            log_metrics(step, combined, rank)
        if step % int(opt_cfg["save_every"]) == 0 and rank == 0:
            save_lora(output / f"step-{step:06d}.pt", step, backend, optimizer, scheduler, config, injected)
        if step % int(opt_cfg["validate_every"]) == 0:
            val_loss = validate(model, projectors, camera_decoder, val_loader, device, str(opt_cfg["amp"]), layers, progress, weights, lambda_delta)
            log_metrics(step, {"val/loss": val_loss}, rank)
            if rank == 0 and float(val_loss) < best:
                best = float(val_loss)
                save_lora(output / "best.pt", step, backend, optimizer, scheduler, config, injected)
    if rank == 0:
        save_lora(output / "last.pt", int(opt_cfg["max_updates"]), backend, optimizer, scheduler, config, injected)
    return output / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()

