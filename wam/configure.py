"""Configure training around prepared Human300 and Target100 data."""

import argparse
import copy
import json
from pathlib import Path
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--stage", choices=["human300", "target100"], required=True)
    p.add_argument("--initial-checkpoint", type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    model = OmegaConf.to_container(
        OmegaConf.load(Path(__file__).parent / "configs/model.yaml"), resolve=True
    )
    groups = (
        ["pretrain"]
        if a.stage == "human300"
        else ["atomic_seen", "composite_seen", "composite_unseen"]
    )
    if a.stage == "target100" and not a.initial_checkpoint:
        p.error("--initial-checkpoint is required")
    if a.initial_checkpoint and not a.initial_checkpoint.is_file():
        raise FileNotFoundError(a.initial_checkpoint)
    reports = []
    for group in groups:
        data = root if a.stage == "human300" else root / "groups" / group
        index = json.loads((data / "data/index.json").read_text())
        normalization = json.loads((data / "data/normalization.json").read_text())
        expected = (
            300 if a.stage == "human300" else (18 if group == "atomic_seen" else 16)
        )
        demos_per_task = 100 if a.stage == "human300" else 500
        assert (
            index["tasks"] == expected
            and len(index["episodes"]) == expected * demos_per_task
        )
        if a.stage == "target100":
            assert (data / "task_ranges.json").is_file()
        for key, width in [("state", 19), ("action", 15)]:
            assert (
                len(normalization[key]["min"])
                == len(normalization[key]["max"])
                == width
            )
        cfg = dict(
            model=copy.deepcopy(model),
            wandb=dict(name=group, group="causalalign-robocasa"),
        )
        if a.stage == "target100":
            cfg["model"].update(
                skip_dit_load_from_pretrain=True, action_dit_pretrained_path=None
            )
            cfg["initial_checkpoint"] = str(a.initial_checkpoint.resolve())
        OmegaConf.save(OmegaConf.create(cfg), data / "config.yaml")
        (root / "runs" / group).mkdir(parents=True, exist_ok=True)
        reports.append(
            dict(group=group, tasks=expected, demonstrations=expected * demos_per_task)
        )
    if a.stage == "target100":
        norms = [
            (root / "groups" / g / "data/normalization.json").read_bytes()
            for g in groups
        ]
        assert len(set(norms)) == 1, (
            "All splits must use the same Human300 normalization"
        )
    (root / "PREPARED.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
