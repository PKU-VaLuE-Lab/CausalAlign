#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from causalalign.data.records import read_jsonl
from causalalign.integrations.wan22 import Wan22TransitionBackend
from causalalign.spec import PAPER_SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen Wan transitions for paper Stage 2")
    parser.add_argument("--wan-input-index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wan-model", required=True)
    parser.add_argument("--expert", default="transformer_2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    args = parser.parse_args()
    rows = read_jsonl(args.wan_input_index)
    backend = Wan22TransitionBackend(args.wan_model, expert=args.expert, device=args.device).eval()
    output = Path(args.output_root).resolve()
    records = output / "records"
    records.mkdir(parents=True, exist_ok=True)
    scene_files: dict[str, Path] = {}
    temporary = output / "index.jsonl.tmp"
    with temporary.open("w", encoding="utf-8", newline="\n") as index:
        for row in rows:
            scene_id = str(row["scene_id"])
            if scene_id not in scene_files:
                source = Path(str(row["file"]))
                if not source.is_absolute():
                    source = Path(row["_index_root"]) / source
                value = torch.load(source, map_location=args.device, weights_only=True)
                batch = {key: tensor[None].to(args.device) for key, tensor in value.items()}
                seed = int(hashlib.sha256(f"{args.seed}:{scene_id}".encode()).hexdigest()[:15], 16)
                generator = torch.Generator(device=args.device).manual_seed(seed)
                batch["noise"] = torch.randn(batch["clean_latent"].shape, device=args.device, dtype=batch["clean_latent"].dtype, generator=generator)
                with torch.inference_mode():
                    transitions = backend(batch, list(PAPER_SPEC.target_layers), list(PAPER_SPEC.target_progress), detach=True, select_pairs=False)
                path = records / f"scene-{len(scene_files):06d}.pt"
                torch.save({"transitions": {key: item[0].cpu() for key, item in transitions.items()}}, path)
                scene_files[scene_id] = path
            index.write(json.dumps({
                "scene_id": scene_id,
                "file": str(scene_files[scene_id].resolve()),
                "target_file": str(Path(row["target_file"]).resolve()),
                "pair_index": int(row["pair_index"]),
            }, ensure_ascii=False) + "\n")
        index.flush()
        os.fsync(index.fileno())
    os.replace(temporary, output / "index.jsonl")


if __name__ == "__main__":
    main()

