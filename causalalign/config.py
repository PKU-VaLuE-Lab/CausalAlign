from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    value = deepcopy(value)
    value["_config_path"] = str(path)
    return value


def require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing required configuration key: {key}")
    return mapping[key]

