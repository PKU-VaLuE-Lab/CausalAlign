#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from causalalign.data import RecordDataset
from causalalign.models.teacher import StructuredDeltaTokenModel
from causalalign.spec import PAPER_SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze Stage-1 delta-token targets for Stages 2/3")
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--input-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != 1 or checkpoint.get("paper_spec_sha256") != PAPER_SPEC.fingerprint():
        raise ValueError("incompatible Stage-1 checkpoint")
    model_config = checkpoint["config"]["model"]
    model = StructuredDeltaTokenModel(**model_config).to(args.device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = RecordDataset(args.input_index)
    output = Path(args.output_root).resolve()
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    temporary = output / "index.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as index:
        with torch.inference_mode():
            for number in range(len(dataset)):
                item = dataset[number]
                token = model.encode(item["state_t"].to(args.device)[None], item["state_t1"].to(args.device)[None])[0].cpu()
                value = {
                    "teacher_token": token,
                    "relative_pose": item["relative_pose"],
                    "scene_scale": item["scene_scale"],
                    "source_frame": item.get("source_frame", torch.tensor(-1)),
                    "target_frame": item.get("target_frame", torch.tensor(-1)),
                    "pair_index": item.get("pair_index", torch.tensor(-1)),
                }
                path = records / f"sample-{number:09d}.pt"
                torch.save(value, path)
                source_row = dataset.rows[number]
                index.write(json.dumps({
                    "scene_id": source_row["scene_id"],
                    "file": f"records/{path.name}",
                    "pair_index": int(source_row.get("pair_index", source_row.get("index", -1))),
                }) + "\n")
        index.flush()
        os.fsync(index.fileno())
    os.replace(temporary, output / "index.jsonl")


if __name__ == "__main__":
    main()

