"""Write the Target100 evaluation plan and one case file per task and seed."""

import argparse
import hashlib
import json
from pathlib import Path
from common import ROOT, TRAIN, GROUPS, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks-json", type=Path)
    a = p.parse_args()
    if a.tasks_json:
        tasks = json.loads(a.tasks_json.read_text())
    else:
        from robocasa.utils.dataset_registry import (
            TARGET_TASKS,
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )

        registry = dict(ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS)
        tasks = [
            dict(group=g, task=t, horizon=registry[t]["horizon"])
            for g, items in TARGET_TASKS.items()
            for t in items
        ]
    assert len(tasks) == 50 and len({t["task"] for t in tasks}) == 50
    for g, n in zip(GROUPS, [18, 16, 16]):
        assert sum(t["group"] == g for t in tasks) == n
    checks = {}
    for g in GROUPS:
        path = TRAIN / "runs" / g / "checkpoints/weights/step_050000.pt"
        before = path.stat()
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(16 * 1024 * 1024), b""):
                h.update(block)
        assert path.stat().st_mtime_ns == before.st_mtime_ns
        checks[g] = dict(
            path=str(path),
            bytes=before.st_size,
            mtime_ns=before.st_mtime_ns,
            sha256=h.hexdigest(),
        )
    norms = {
        g: hashlib.sha256(
            (TRAIN / "groups" / g / "data/normalization.json").read_bytes()
        ).hexdigest()
        for g in GROUPS
    }
    assert len(set(norms.values())) == 1
    plan = dict(
        tasks=tasks,
        seeds=list(range(10000, 10050)),
        checkpoints=checks,
        normalization_sha256=norms,
        checkpoint_step=50000,
        predict_actions=32,
        execute_actions=8,
        denoising_steps=10,
        control_hz=20,
        render="CPU llvmpipe",
    )
    if (ROOT / "PLAN.json").exists():
        assert json.loads((ROOT / "PLAN.json").read_text()) == plan
    for name in [
        "logs",
        "control",
        "results",
        "errors",
        "paired_cases",
        "cache",
        "cases",
    ]:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    write(ROOT / "PLAN.json", plan)
    for index, t in enumerate(tasks):
        for seed in plan["seeds"]:
            write(
                ROOT / "cases" / f"{index:02d}_{seed}.json",
                dict(**t, task_index=index, seed=seed),
            )


if __name__ == "__main__":
    main()
