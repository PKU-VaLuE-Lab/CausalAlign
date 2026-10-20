"""Evaluation paths shared by the policy server and simulator."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("WAM_EVAL_ROOT", "runs/wam_eval")).resolve()
TRAIN = Path(os.environ.get("WAM_RUN_ROOT", "runs/wam")).resolve()
SIM = Path(os.environ.get("ROBOCASA_ENV", sys.prefix)).resolve()
GROUPS = ["atomic_seen", "composite_seen", "composite_unseen"]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)
