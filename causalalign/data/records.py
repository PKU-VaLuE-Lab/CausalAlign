from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row is not an object: {path}:{number}")
            value["_index_root"] = str(path.parent)
            rows.append(value)
    if not rows:
        raise ValueError(f"empty index: {path}")
    return rows


def _select(value: Any, index: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, Mapping):
        return {key: _select(item, index) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


class RecordDataset(Dataset[dict[str, Any]]):
    """Index of torch records; an optional index selects a sample in a scene shard."""

    def __init__(self, index: str | Path, max_scenes: int | None = None):
        rows = read_jsonl(index)
        if max_scenes is not None:
            selected: set[str] = set()
            kept: list[dict[str, Any]] = []
            for row in rows:
                scene = str(row["scene_id"])
                if scene not in selected and len(selected) >= max_scenes:
                    continue
                selected.add(scene)
                kept.append(row)
            rows = kept
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        path = Path(str(row["file"]))
        if not path.is_absolute():
            path = Path(row["_index_root"]) / path
        record = torch.load(path, map_location="cpu", weights_only=True)
        if "index" in row:
            record = _select(record, int(row["index"]))
        if not isinstance(record, dict):
            raise ValueError(f"record must be a mapping: {path}")
        if "target_file" in row:
            target_path = Path(str(row["target_file"]))
            if not target_path.is_absolute():
                target_path = Path(row["_index_root"]) / target_path
            target = torch.load(target_path, map_location="cpu", weights_only=True)
            if not isinstance(target, dict):
                raise ValueError(f"target record must be a mapping: {target_path}")
            record = {**record, **target}
        if "pair_index" in row:
            pair_index = int(row["pair_index"])
            if "transitions" in record:
                record["transitions"] = {
                    key: value[pair_index] for key, value in record["transitions"].items()
                }
            record["pair_index"] = torch.tensor(pair_index, dtype=torch.long)
        return {**record, "scene_id": str(row["scene_id"])}


def tensor_collate(batch: Sequence[Any]) -> Any:
    first = batch[0]
    if isinstance(first, torch.Tensor):
        return torch.stack(list(batch))
    if isinstance(first, Mapping):
        return {key: tensor_collate([item[key] for item in batch]) for key in first if key != "scene_id"}
    if isinstance(first, (float, int)):
        return torch.tensor(batch)
    return list(batch)

