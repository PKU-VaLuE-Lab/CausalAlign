"""Human300 video-action training with native DDP."""

import datetime, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler
from hydra.utils import instantiate
from omegaconf import OmegaConf
from training_utils import apply_train_mode
from recipe_data import RecipeDataset

ROOT = Path(os.environ.get("WAM_RUN_ROOT", "runs/wam")).resolve()
OUT = ROOT / "runs/pretrain"
OUT.mkdir(parents=True, exist_ok=True)
rank = int(os.environ["RANK"])
local = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
device = torch.device("cuda", local)
torch.cuda.set_device(device)
torch.manual_seed(42 + rank)
random.seed(42 + rank)
np.random.seed(42 + rank)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
cfg = OmegaConf.load(ROOT / "config.yaml")
model = instantiate(cfg.model, device=str(device), model_dtype=torch.bfloat16)
apply_train_mode(model)
dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=15))
assert world == 16
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
    masters, lr=1e-4 / 3000, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01, fused=True
)
step = 0
epoch = 0
batch_in_epoch = 0
resume = os.environ.get("WAM_RESUME_STATE")
if resume:
    state_dir = Path(resume)
    p = torch.load(
        state_dir / "optimizer.pt", map_location="cpu", mmap=True, weights_only=False
    )
    model.load_checkpoint(
        str(state_dir.parent.parent / "weights" / f"{state_dir.name}.pt")
    )
    assert list(p["masters"]) == [n for n, _ in named]
    with torch.no_grad():
        for (name, _), master in zip(named, masters):
            master.copy_(p["masters"][name])
    optimizer.load_state_dict(p["optimizer"])
    step = p["trainer"]["global_step"]
    epoch = p["trainer"]["epoch"]
    batch_in_epoch = p["trainer"]["batch_in_epoch"]
    del p
initial_step = step


def lr_at(completed):
    if completed < 3000:
        return 1e-4 * (1 / 3000 + completed * (1 - 1 / 3000) / 3000)
    return 1e-4 * (
        0.01 + 0.99 * (1 + math.cos(math.pi * min(1.0, (completed - 3000) / 57000))) / 2
    )


class RankSampler(Sampler):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return (math.ceil(len(self.ds) / 256) - batch_in_epoch) * 16

    def __iter__(self):
        order = torch.randperm(
            len(self.ds), generator=torch.Generator().manual_seed(42 + epoch)
        )
        pad = (-len(order)) % 256
        if pad:
            order = torch.cat([order, order[:pad]])
        start_batch = batch_in_epoch
        for batch in range(start_batch, len(order) // 256):
            start = batch * 256 + rank * 16
            yield from order[start : start + 16].tolist()


dataset = RecipeDataset(ROOT, training=True)
loader = DataLoader(
    dataset,
    batch_size=16,
    sampler=RankSampler(dataset),
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
    generator=torch.Generator().manual_seed(42 + rank),
)
iterator = iter(loader)
if resume:
    rng = torch.load(
        Path(resume) / f"random_states_{rank}.pkl",
        map_location="cpu",
        weights_only=False,
    )
    random.setstate(rng["random_state"])
    np.random.set_state(rng["numpy_random_seed"])
    torch.set_rng_state(rng["torch_manual_seed"])
    torch.cuda.set_rng_state_all(rng["torch_cuda_manual_seed"])
    del rng
run = None
if rank == 0:
    import wandb

    run = wandb.init(
        entity=os.environ.get("WANDB_ENTITY"),
        project="fast-wam",
        name=str(cfg.wandb.name),
        group=str(cfg.wandb.group),
        dir=str(OUT),
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    OmegaConf.save(cfg, OUT / "config.yaml")
    (OUT / "INIT.json").write_text(
        json.dumps(
            dict(
                start_step=step,
                world_size=16,
                global_batch=256,
                model_paths=model.model_paths,
                resume=resume,
                wandb=run.url,
            ),
            indent=2,
        )
    )
print("WAM_TRAIN_READY", rank, step, flush=True)


def save(full=False):
    dist.barrier()
    tag = f"step_{step:06d}"
    weight = OUT / "checkpoints/weights" / f"{tag}.pt"
    state_dir = OUT / "checkpoints/state" / tag
    if rank == 0:
        weight.parent.mkdir(parents=True, exist_ok=True)
        model.save_checkpoint(str(weight) + ".tmp", step=step)
        Path(str(weight) + ".tmp").replace(weight)
    if full:
        state_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                random_state=random.getstate(),
                numpy_random_seed=np.random.get_state(),
                torch_manual_seed=torch.get_rng_state(),
                torch_cuda_manual_seed=torch.cuda.get_rng_state_all(),
            ),
            state_dir / f"random_states_{rank}.pkl",
        )
        if rank == 0:
            progress = dict(
                global_step=step, epoch=epoch, batch_in_epoch=batch_in_epoch
            )
            torch.save(
                dict(
                    masters={n: m.detach().cpu() for (n, p), m in zip(named, masters)},
                    optimizer=optimizer.state_dict(),
                    trainer=progress,
                ),
                state_dir / "optimizer.pt.tmp",
            )
            (state_dir / "optimizer.pt.tmp").replace(state_dir / "optimizer.pt")
            (state_dir / "trainer_state.json").write_text(json.dumps(progress))
    dist.barrier()
    if rank == 0:
        (OUT / "LAST_CHECKPOINT.json").write_text(
            json.dumps(
                dict(
                    step=step,
                    weights=str(weight),
                    state=str(state_dir) if full else None,
                )
            )
        )


window = []
torch.cuda.reset_peak_memory_stats()
while step < 60000:
    ddp.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    try:
        sample = next(iterator)
    except StopIteration:
        epoch += 1
        batch_in_epoch = 0
        iterator = iter(loader)
        sample = next(iterator)
    loss, detail = ddp(sample)
    assert torch.isfinite(loss)
    loss.backward()
    for (name, p), m in zip(named, masters):
        m.grad = None if p.grad is None else p.grad.detach().float()
    norm = torch.nn.utils.clip_grad_norm_(masters, 1.0, foreach=True)
    assert torch.isfinite(norm)
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
    optimizer.step()
    with torch.no_grad():
        for (name, p), m in zip(named, masters):
            p.copy_(m)
    step += 1
    batch_in_epoch += 1
    for group in optimizer.param_groups:
        group["lr"] = lr_at(step)
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
    check = step % 10 == 0 or step <= initial_step + 3 or step == 60000
    stop = False
    if check:
        values = torch.tensor(window, device=device, dtype=torch.float64).mean(0)
        maximum = values.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(values)
        values /= world
        window = []
        if rank == 0:
            assert int(optimizer.state[masters[0]]["step"]) == step
            record = dict(
                step=step,
                max_steps=60000,
                world_size=16,
                batch=16,
                global_batch=256,
                seconds_per_step=float(maximum[0]),
                loss=float(values[1]),
                loss_action=float(values[2]),
                loss_video=float(values[3]),
                grad_norm=float(values[4]),
                peak_gib=float(maximum[5]),
                lr=lr_at(step),
                wandb=run.url,
                time=time.time(),
                video_checkpointing=bool(
                    cfg.model.video_dit_config.use_gradient_checkpointing
                ),
            )
            (OUT / "TRAINING_STATUS.json").write_text(json.dumps(record, indent=2))
            print("WAM_TRAIN", json.dumps(record), flush=True)
            run.log(
                {
                    "train/" + k: record[k]
                    for k in ["loss", "loss_action", "loss_video", "grad_norm", "lr"]
                }
                | {
                    "performance/seconds_per_step": record["seconds_per_step"],
                    "performance/samples_per_second": 256 / record["seconds_per_step"],
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
    if step % 2000 == 0 or step == 60000 or stop:
        save(full=step % 10000 == 0 or step == 60000 or stop)
    if stop:
        if rank == 0:
            (OUT / "SAVED_FOR_RESTART.json").write_text(
                (OUT / "LAST_CHECKPOINT.json").read_text()
            )
        break
if rank == 0:
    run.finish()
dist.barrier()
dist.destroy_process_group()
