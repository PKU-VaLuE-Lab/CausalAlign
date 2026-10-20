#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import importlib
import json
import os
from pathlib import Path

import torch

from causalalign.data.records import read_jsonl


def factory(name: str):
    module, function = name.split(":", 1)
    return getattr(importlib.import_module(module), function)


def target_path(row: dict) -> Path:
    value = Path(str(row["file"]))
    return value if value.is_absolute() else Path(row["_index_root"]) / value


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache only Wan inputs; hidden states remain live in Stage 3")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--teacher-token-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wan-model", required=True)
    parser.add_argument("--preprocessor", default="causalalign.integrations.wan_preprocess:create_preprocessor")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--max-frames", type=int, default=81)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    targets: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.teacher_token_index):
        targets[str(row["scene_id"])].append(row)
    preprocessor = factory(args.preprocessor)(args)
    output = Path(args.output_root).resolve()
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    temporary = output / "index.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as index:
        for number, scene in enumerate(read_jsonl(args.scene_manifest)):
            scene_id = str(scene["scene_id"])
            if scene_id not in targets:
                continue
            frames = scene["frames"][: int(args.max_frames)]
            if (len(frames) - 1) % 4:
                frames = frames[: len(frames) - ((len(frames) - 1) % 4)]
            encoded = preprocessor.encode([str(item["image"]) for item in frames], str(scene.get("caption", "")))
            path = records / f"scene-{number:06d}.pt"
            torch.save(encoded, path)
            latent_pairs = int(encoded["clean_latent"].shape[1]) - 1
            for target in targets[scene_id]:
                pair_index = int(target.get("pair_index", target.get("index", -1)))
                if 0 <= pair_index < latent_pairs:
                    index.write(json.dumps({
                        "scene_id": scene_id,
                        "file": f"records/{path.name}",
                        "target_file": str(target_path(target).resolve()),
                        "pair_index": pair_index,
                    }, ensure_ascii=False) + "\n")
        index.flush()
        os.fsync(index.fileno())
    os.replace(temporary, output / "index.jsonl")


if __name__ == "__main__":
    main()

