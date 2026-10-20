from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from causalalign.data.manifest import scan_realestate10k


def test_realestate10k_header_and_camera_rows(tmp_path: Path):
    scene = "scene-a"
    image_dir = tmp_path / "frames" / scene
    image_dir.mkdir(parents=True)
    pose = " ".join(str(value) for value in (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0))
    rows = ["https://www.youtube.com/watch?v=example"]
    for index in range(9):
        timestamp = str(1000 + index)
        Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(image_dir / f"{timestamp}.jpg")
        rows.append(f"{timestamp} 0.8 0.8 0.5 0.5 {pose}")
    (tmp_path / f"{scene}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    scenes = scan_realestate10k(tmp_path, "realestate10k", 9, "fixed caption")
    assert len(scenes) == 1
    assert len(scenes[0]["frames"]) == 9
    assert scenes[0]["caption"] == "fixed caption"
    assert abs(scenes[0]["frames"][0]["K"][0][0] - 9.6) < 1e-6
