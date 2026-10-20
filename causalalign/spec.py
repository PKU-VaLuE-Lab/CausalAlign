"""Immutable method contract shared by data preparation and all three stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class PaperSpec:
    token_dim: int = 128
    ego_dim: int = 64
    geometry_dim: int = 40
    visual_dim: int = 24
    stage1_fraction: float = 0.30
    stage2_fraction: float = 0.10
    stage3_fraction: float = 1.00
    train_fraction: float = 0.90
    stage1_updates: int = 40_000
    stage1_warmup_updates: int = 10_000
    stage1_warmup_scenes: int = 4_000
    stage1_full_scenes: int = 8_312
    lambda_temporal: float = 0.1
    lambda_teacher_pose: float = 0.2
    lambda_delta: float = 0.2
    target_layers: tuple[int, ...] = (10, 12, 14, 20)
    primary_layer: int = 14
    target_progress: tuple[int, ...] = (398, 799)

    def validate(self) -> None:
        if self.ego_dim + self.geometry_dim + self.visual_dim != self.token_dim:
            raise ValueError("the structured token slices do not sum to token_dim")
        if self.primary_layer not in self.target_layers:
            raise ValueError("primary layer must be a target layer")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


PAPER_SPEC = PaperSpec()
PAPER_SPEC.validate()

