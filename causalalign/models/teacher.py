from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from causalalign.geometry import so3_exp
from causalalign.spec import PAPER_SPEC


@dataclass(frozen=True)
class TokenLayout:
    ego: slice = slice(0, 64)
    geometry: slice = slice(64, 104)
    visual: slice = slice(104, 128)

    def split(self, value: torch.Tensor) -> dict[str, torch.Tensor]:
        if value.shape[-1] != PAPER_SPEC.token_dim:
            raise ValueError(f"expected 128-D token, got {tuple(value.shape)}")
        return {"ego": value[..., self.ego], "geometry": value[..., self.geometry], "visual": value[..., self.visual]}


TOKEN_LAYOUT = TokenLayout()


class ResidualMLP(nn.Module):
    def __init__(self, width: int, depth: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, width))
            for _ in range(depth)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = value + block(value)
        return value


class DeltaEncoder(nn.Module):
    def __init__(self, state_dim: int, width: int, depth: int):
        super().__init__()
        self.input = nn.Linear(3 * state_dim, width)
        self.body = ResidualMLP(width, depth)
        self.norm = nn.LayerNorm(width)
        self.ego = nn.Linear(width, 64)
        self.geometry = nn.Linear(width, 40)
        self.visual = nn.Linear(width, 24)

    def forward(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> torch.Tensor:
        if state_t.shape != state_t1.shape:
            raise ValueError("paired teacher states must have identical shapes")
        value = torch.cat((state_t, state_t1, state_t1 - state_t), -1)
        value = self.norm(self.body(self.input(value)))
        return torch.cat((self.ego(value), self.geometry(value), self.visual(value)), -1)


class TransitionDecoder(nn.Module):
    """One transition decoder; geometry/RGB readouts are components of L_tr."""

    def __init__(self, state_dim: int, width: int, depth: int, geometry_dim: int, visual_dim: int):
        super().__init__()
        self.input = nn.Linear(state_dim + 128, width)
        self.body = ResidualMLP(width, depth)
        self.state = nn.Linear(width, state_dim)
        # Hard information routing gives the declared token regions meaning.
        self.geometry = nn.Sequential(nn.LayerNorm(40), nn.Linear(40, width), nn.SiLU(), nn.Linear(width, geometry_dim))
        self.visual = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, width), nn.SiLU(), nn.Linear(width, visual_dim))

    def forward(self, state_t: torch.Tensor, token: torch.Tensor) -> dict[str, torch.Tensor]:
        parts = TOKEN_LAYOUT.split(token)
        hidden = self.body(self.input(torch.cat((state_t, token), -1)))
        return {
            "state_t1": state_t + self.state(hidden),
            "geometry_delta": self.geometry(parts["geometry"]),
            "visual_delta": self.visual(parts["visual"]),
        }


class CameraDecoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(64), nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.rotation = nn.Linear(width, 3)
        self.translation = nn.Linear(width, 3)

    def forward(self, token: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(TOKEN_LAYOUT.split(token)["ego"])
        so3 = self.rotation(hidden).float()
        translation = self.translation(hidden).float()
        return {"so3": so3, "rotation": so3_exp(so3), "translation": translation}


class StructuredDeltaTokenModel(nn.Module):
    """Stage-1 encoder, transition decoder, temporal predictor and camera decoder."""

    def __init__(self, state_dim: int = 4096, width: int = 1024, depth: int = 4, geometry_dim: int = 1792, visual_dim: int = 5376):
        super().__init__()
        self.config = dict(state_dim=state_dim, width=width, depth=depth, geometry_dim=geometry_dim, visual_dim=visual_dim)
        self.encoder = DeltaEncoder(state_dim, width, depth)
        self.transition_decoder = TransitionDecoder(state_dim, width, depth, geometry_dim, visual_dim)
        self.temporal_predictor = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, width), nn.SiLU(), nn.Linear(width, 128))
        self.camera_decoder = CameraDecoder(width)

    def encode(self, state_t: torch.Tensor, state_t1: torch.Tensor) -> torch.Tensor:
        return self.encoder(state_t, state_t1)

    def forward(
        self,
        state_t: torch.Tensor,
        state_t1: torch.Tensor,
        state_t2: torch.Tensor | None = None,
    ) -> dict[str, object]:
        token = self.encode(state_t, state_t1)
        output = {
            "token": token,
            "transition": self.transition_decoder(state_t, token),
            "next_token": self.temporal_predictor(token),
            "camera": self.camera_decoder(token),
        }
        if state_t2 is not None:
            output["target_next_token"] = self.encode(state_t1, state_t2)
        return output

