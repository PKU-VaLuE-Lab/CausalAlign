from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
import yaml


CV_CONVERSION = np.diag([1.0, -1.0, -1.0, 1.0])


def stable_score(name: str, seed: int) -> int:
    return int(hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(), 16)


def _resolve_image(scene: Path, name: str) -> Path:
    candidate = Path(name)
    candidates = [candidate if candidate.is_absolute() else scene / candidate]
    for extension in (".png", ".jpg", ".jpeg", ".JPG"):
        candidates.append(scene / f"{candidate.as_posix()}{extension}")
        candidates.append(scene / "images" / f"{candidate.stem}{extension}")
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(f"cannot resolve frame {name!r} below {scene}")


def scan_transforms(root: Path, dataset: str, min_frames: int, default_caption: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in sorted(root.rglob("transforms.json")):
        value = json.loads(source.read_text(encoding="utf-8"))
        frames = value.get("frames", [])
        if len(frames) < min_frames:
            continue
        scene_id = f"{dataset}/{source.parent.relative_to(root).as_posix()}"
        frame_rows = []
        for index, frame in enumerate(frames):
            image = _resolve_image(source.parent, str(frame["file_path"]))
            with Image.open(image) as decoded:
                width, height = decoded.size
            fx = frame.get("fl_x", value.get("fl_x"))
            fy = frame.get("fl_y", value.get("fl_y", fx))
            if fx is None:
                angle = float(frame.get("camera_angle_x", value["camera_angle_x"]))
                fx = 0.5 * width / math.tan(0.5 * angle)
                fy = fx if fy is None else fy
            cx = float(frame.get("cx", value.get("cx", width / 2)))
            cy = float(frame.get("cy", value.get("cy", height / 2)))
            pose_gl = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if pose_gl.shape != (4, 4):
                raise ValueError(f"invalid pose in {source}, frame {index}")
            pose_cv = pose_gl @ CV_CONVERSION
            frame_rows.append({
                "frame_id": index,
                "image": str(image),
                "K": [[float(fx), 0.0, cx], [0.0, float(fy), cy], [0.0, 0.0, 1.0]],
                "c2w_opencv": pose_cv.tolist(),
            })
        caption = str(value.get("caption") or default_caption).strip()
        rows.append({"scene_id": scene_id, "dataset": dataset, "caption": caption, "frames": frame_rows})
    return rows


def scan_realestate10k(root: Path, dataset: str, min_frames: int, default_caption: str) -> list[dict[str, Any]]:
    """Read standard RE10K camera txt files and decoded timestamp images.

    Each pose line is interpreted using its final 12 values as a 3x4 w2c
    matrix; the first four values after timestamp are normalized fx/fy/cx/cy.
    """
    rows: list[dict[str, Any]] = []
    image_index: dict[tuple[str, str], Path] = {}
    for image in root.rglob("*"):
        if image.is_file() and image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            key = (image.parent.name, image.stem)
            if key in image_index and image_index[key] != image:
                raise ValueError(f"duplicate RE10K frame key {key}: {image_index[key]} and {image}")
            image_index[key] = image.resolve()
    for source in sorted(root.rglob("*.txt")):
        lines = []
        for raw in source.read_text(encoding="utf-8").splitlines():
            fields = raw.split()
            # Official files may start with a video URL; camera rows start
            # with an integer timestamp and contain intrinsics plus a 3x4 pose.
            if len(fields) < 17:
                continue
            try:
                int(fields[0])
                [float(value) for value in fields[1:]]
            except ValueError:
                continue
            lines.append(fields)
        if len(lines) < min_frames:
            continue
        frames = []
        for index, fields in enumerate(lines):
            timestamp = fields[0]
            image = image_index.get((source.stem, timestamp))
            if image is None:
                continue
            with Image.open(image) as decoded:
                width, height = decoded.size
            fx, fy, cx, cy = map(float, fields[1:5])
            if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 2.0:
                fx, fy, cx, cy = fx * width, fy * height, cx * width, cy * height
            w2c = np.eye(4)
            w2c[:3] = np.asarray(fields[-12:], dtype=np.float64).reshape(3, 4)
            c2w = np.linalg.inv(w2c)
            frames.append({
                "frame_id": index,
                "timestamp": timestamp,
                "image": str(image),
                "K": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                "c2w_opencv": c2w.tolist(),
            })
        if len(frames) >= min_frames:
            rows.append({"scene_id": f"{dataset}/{source.stem}", "dataset": dataset, "caption": default_caption, "frames": frames})
    return rows


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(config_path: str | Path) -> dict[str, int]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed, minimum = int(config["seed"]), int(config["min_frames"])
    default_caption = str(config["default_caption"]).strip()
    if not default_caption:
        raise ValueError("default_caption must be non-empty")
    all_scenes: list[dict[str, Any]] = []
    for name, item in config["datasets"].items():
        root = Path(item["root"]).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"dataset root does not exist: {root}")
        if item["format"] == "transforms":
            all_scenes += scan_transforms(root, name, minimum, default_caption)
        elif item["format"] == "realestate10k":
            all_scenes += scan_realestate10k(root, name, minimum, default_caption)
        else:
            raise ValueError(f"unknown dataset format: {item['format']}")
    # Split each source independently, as stated by the paper.
    train, evaluation = [], []
    fraction = float(config["train_fraction"])
    for dataset in config["datasets"]:
        scenes = sorted((x for x in all_scenes if x["dataset"] == dataset), key=lambda x: stable_score(x["scene_id"], seed))
        cut = int(len(scenes) * fraction)
        train += scenes[:cut]
        evaluation += scenes[cut:]
    train = sorted(train, key=lambda x: stable_score(x["scene_id"], seed))
    output = Path(config["output_root"]).resolve()
    atomic_jsonl(output / "train_scenes.jsonl", train)
    atomic_jsonl(output / "eval_scenes.jsonl", evaluation)
    for stage, stage_fraction in config["stage_fractions"].items():
        count = int(len(train) * float(stage_fraction))
        atomic_jsonl(output / f"{stage}_scenes.jsonl", train[:count])
    summary = {"all": len(all_scenes), "train": len(train), "eval": len(evaluation)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary

