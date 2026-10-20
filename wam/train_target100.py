"""Target split fine-tuning: strict Human300 60k weights, fresh optimizer."""

import datetime, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from hydra.utils import instantiate
from omegaconf import OmegaConf

ROOT = Path(os.environ.get("WAM_RUN_ROOT", "runs/wam")).resolve()
from recipe_data import RecipeDataset
from training_utils import apply_train_mode

group = os.environ["TARGET_GROUP"]
assert group in ["atomic_seen", "composite_seen", "composite_unseen"]
OUT = ROOT / "runs" / group
DATA = ROOT / "groups" / group
OUT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["RANK"])
local = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 8
device = torch.device("cuda", local)
torch.cuda.set_device(device)
torch.manual_seed(42 + rank)
random.seed(42 + rank)
np.random.seed(42 + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cfg = OmegaConf.load(DATA / "config.yaml")
model = instantiate(cfg.model, device=str(device), model_dtype=torch.bfloat16)
payload = torch.load(
    cfg.initial_checkpoint, map_location="cpu", mmap=True, weights_only=False
)
assert payload["step"] == 60000
model.mot.load_state_dict(payload["mot"], strict=True)
model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
del payload
apply_train_mode(model)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
ddp = DistributedDataParallel(
    model,
    device_ids=[local],
    broadcast_buffers=False,
    find_unused_parameters=True,
    gradient_as_bucket_view=True,
    bucket_cap_mb=125,
)


def fp32_hook(state, bucket):
    b = bucket.buffer()
    x = b.float().div_(world)
    return (
        dist.all_reduce(x, async_op=True)
        .get_future()
        .then(lambda f: b.copy_(f.value()[0]))
    )


ddp.register_comm_hook(None, fp32_hook)
named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
masters = [torch.nn.Parameter(p.detach().float()) for n, p in named]
optimizer = torch.optim.AdamW(
    masters, lr=3e-5 / 3000, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True
)
step = 0
resume = os.environ.get("TARGET_RESUME_STATE")
if resume:
    state_dir = Path(resume)
    saved = torch.load(
        state_dir / "optimizer.pt", map_location="cpu", mmap=True, weights_only=False
    )
    model.load_checkpoint(
        str(state_dir.parent.parent / "weights" / f"{state_dir.name}.pt")
    )
    assert list(saved["masters"]) == [n for n, p in named]
    with torch.no_grad():
        for (name, p), m in zip(named, masters):
            m.copy_(saved["masters"][name])
    optimizer.load_state_dict(saved["optimizer"])
    step = saved["step"]
    del saved
initial_step = step


def lr_at(s):
    if s < 3000:
        return 3e-5 * (s + 1) / 3000
    return (
        3e-6
        + (3e-5 - 3e-6) * (1 + math.cos(math.pi * min(1.0, (s - 3000) / 57000))) / 2
    )


ranges = np.asarray(json.loads((DATA / "task_ranges.json").read_text()), dtype=np.int64)


class TaskSampler(Sampler):
    def __len__(self):
        return (60000 - initial_step) * 16

    def __iter__(self):
        for s in range(initial_step, 60000):
            gen = np.random.default_rng(42 + s * 104729)
            task = gen.integers(0, len(ranges), size=128)
            selected = ranges[task, 0] + (
                gen.random(128) * (ranges[task, 1] - ranges[task, 0])
            ).astype(np.int64)
            yield from selected[rank * 16 : (rank + 1) * 16].tolist()


dataset = RecipeDataset(DATA, training=True)
loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=TaskSampler(),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    generator=torch.Generator().manual_seed(42 + rank),
)
run = None
if rank == 0:
    import wandb

    resume_wandb = os.environ.get("TARGET_WANDB_ID")
    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project="fast-wam",
        name=str(cfg.wandb.name),
        group=str(cfg.wandb.group),
        dir=str(OUT),
        config=OmegaConf.to_container(cfg, resolve=True),
        **(dict(id=resume_wandb, resume="must") if resume_wandb else {}),
    )
    OmegaConf.save(cfg, OUT / "config.yaml")
    (OUT / "INIT.json").write_text(
        json.dumps(
            dict(
                init_checkpoint=str(cfg.initial_checkpoint),
                strict_load=True,
                optimizer_reset=not bool(resume),
                init_step=60000,
                target_step=step,
                world_size=world,
                global_batch=128,
                wandb=run.url,
                group=group,
            ),
            indent=2,
        )
    )
print("TARGET_TRAIN_READY", rank, group, step, flush=True)


def save(full):
    dist.barrier()
    tag = f"step_{step:06d}"
    weight = OUT / "checkpoints/weights" / f"{tag}.pt"
    state = OUT / "checkpoints/state" / tag
    if rank == 0:
        weight.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(weight) + ".tmp", step=step)
        Path(str(weight) + ".tmp").replace(weight)
        if full:
            state.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    masters={n: m.detach().cpu() for (n, p), m in zip(named, masters)},
                    optimizer=optimizer.state_dict(),
                    step=step,
                    source_checkpoint=str(cfg.initial_checkpoint),
                    group=group,
                ),
                state / "optimizer.pt.tmp",
            )
            (state / "optimizer.pt.tmp").replace(state / "optimizer.pt")
        (OUT / "LAST_CHECKPOINT.json").write_text(
            json.dumps(
                dict(step=step, weights=str(weight), state=str(state) if full else None)
            )
        )
    dist.barrier()


window = []
torch.cuda.reset_peak_memory_stats()
iterator = iter(loader)
while step < 60000:
    ddp.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    sample = next(iterator)
    loss, detail = ddp(sample)
    assert torch.isfinite(loss)
    loss.backward()
    for (name, p), m in zip(named, masters):
        m.grad = None if p.grad is None else p.grad.detach().float()
    norm = torch.nn.utils.clip_grad_norm_(masters, 1.0, foreach=True)
    assert torch.isfinite(norm)
    for g in optimizer.param_groups:
        g["lr"] = lr_at(step)
    optimizer.step()
    with torch.no_grad():
        for (name, p), m in zip(named, masters):
            p.copy_(m)
    step += 1
    torch.cuda.synchronize()
    window.append(
        [
            time.perf_counter() - begin,
            float(loss.detach()),
            detail["loss_action"],
            detail["loss_video"],
            float(norm),
            torch.cuda.max_memory_allocated() / 2**30,
        ]
    )
    stop = False
    if step % 10 == 0 or step <= initial_step + 3 or step == 60000:
        values = torch.tensor(window, device=device, dtype=torch.float64).mean(0)
        maximum = values.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(values)
        values /= world
        window = []
        if rank == 0:
            assert int(optimizer.state[masters[0]]["step"]) == step
            record = dict(
                group=group,
                step=step,
                max_steps=60000,
                source_step=60000,
                world_size=8,
                global_batch=128,
                seconds_per_step=float(maximum[0]),
                loss=float(values[1]),
                loss_action=float(values[2]),
                loss_video=float(values[3]),
                grad_norm=float(values[4]),
                peak_gib=float(maximum[5]),
                lr=lr_at(step),
                wandb=run.url,
                time=time.time(),
                resumed_from=initial_step,
                optimizer_preserved=bool(resume),
                host=os.uname().nodename,
            )
            (OUT / "TRAINING_STATUS.json").write_text(json.dumps(record, indent=2))
            print("TARGET_TRAIN", json.dumps(record), flush=True)
            run.log(
                {
                    "train/" + k: record[k]
                    for k in ["loss", "loss_action", "loss_video", "grad_norm", "lr"]
                },
                step=step,
            )
        flag = torch.tensor(
            [int((OUT / "SAVE_FULL_AND_EXIT.json").exists()) if rank == 0 else 0],
            device=device,
            dtype=torch.int,
        )
        dist.broadcast(flag, src=0)
        stop = bool(flag.item())
    if step % 5000 == 0 or step == 60000 or stop:
        save(full=step % 10000 == 0 or step == 60000 or stop)
    if stop or step == 60000:
        break
if rank == 0:
    run.finish()
    if step == 60000:
        (OUT / "TRAINING_DONE.json").write_text(
            json.dumps(dict(step=step, time=time.time()))
        )
dist.barrier()
dist.destroy_process_group()
