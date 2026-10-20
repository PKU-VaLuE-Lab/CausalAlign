#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from causalalign.data.records import read_jsonl
from causalalign.geometry import relative_pose


def load_factory(name: str):
    module, function = name.split(":", 1)
    return getattr(importlib.import_module(module), function)


def rgb_features(paths: list[str]) -> torch.Tensor:
    values = []
    for path in paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((56, 32), Image.Resampling.BICUBIC)
            values.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0)
    return torch.stack(values)


def geometry_features(depth: torch.Tensor) -> torch.Tensor:
    depth = F.interpolate(depth[:, None].float(), size=(16, 28), mode="bilinear", align_corners=False)[:, 0].clamp_min(1e-5)
    log_depth = depth.log()
    dx = F.pad(log_depth[:, :, 2:] - log_depth[:, :, :-2], (1, 1, 0, 0)) * 0.5
    dy = F.pad(log_depth[:, 2:] - log_depth[:, :-2], (0, 0, 1, 1)) * 0.5
    normals = F.normalize(torch.stack((-dx, -dy, torch.ones_like(dx)), 1), dim=1)
    return torch.cat((log_depth[:, None], normals), 1).flatten(1)


def scene_scale(poses: torch.Tensor) -> torch.Tensor:
    centers = poses[:, :3, 3]
    baseline = torch.linalg.vector_norm(centers[1:] - centers[:-1], dim=-1)
    positive = baseline[baseline > 1e-8]
    return positive.median() if positive.numel() else torch.tensor(1.0)


def atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frozen LingBot features for paper Stage 1")
    parser.add_argument("--scene-manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--extractor", default="causalalign.integrations.lingbot_map:create_extractor")
    parser.add_argument("--lingbot-repo", required=True)
    parser.add_argument("--lingbot-checkpoint", required=True)
    parser.add_argument("--camera-num-iterations", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extractor = load_factory(args.extractor)(args)
    output = Path(args.output_root).resolve()
    index_path = output / "index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = index_path.with_suffix(".jsonl.tmp")
    with temporary_index.open("w", encoding="utf-8", newline="\n") as index:
        for scene_number, scene in enumerate(read_jsonl(args.scene_manifest)):
            frames = scene["frames"]
            paths = [str(item["image"]) for item in frames]
            extracted = extractor.encode(paths)
            state = extracted["state"].float()
            if state.shape != (len(frames), 4096):
                raise ValueError(f"LingBot state must be [F,4096], got {tuple(state.shape)}")
            geometry = geometry_features(extracted["depth"])
            visual = rgb_features(paths).flatten(1)
            poses = torch.tensor([item["c2w_opencv"] for item in frames], dtype=torch.float32)
            scale = scene_scale(poses)
            stride = int(args.frame_stride)
            sources = list(range(0, len(frames) - 2 * stride, stride))
            if not sources:
                continue
            first = torch.tensor(sources, dtype=torch.long)
            second = first + stride
            third = second + stride
            windows = len(sources)
            record = {
                "state_t": state.index_select(0, first),
                "state_t1": state.index_select(0, second),
                "state_t2": state.index_select(0, third),
                "geometry_delta": geometry.index_select(0, second) - geometry.index_select(0, first),
                "visual_delta": visual.index_select(0, second) - visual.index_select(0, first),
                "relative_pose": torch.stack([relative_pose(poses[i], poses[i + stride]) for i in sources]),
                "scene_scale": scale.expand(windows).clone(),
                "source_frame": first,
                "target_frame": second,
                "pair_index": torch.arange(windows, dtype=torch.long),
            }
            name = f"scene-{scene_number:06d}.pt"
            atomic_torch_save(output / "records" / name, record)
            for sample in range(windows):
                index.write(json.dumps({"scene_id": scene["scene_id"], "file": f"records/{name}", "index": sample, "pair_index": sample}, ensure_ascii=False) + "\n")
        index.flush()
        os.fsync(index.fileno())
    os.replace(temporary_index, index_path)


if __name__ == "__main__":
    main()

