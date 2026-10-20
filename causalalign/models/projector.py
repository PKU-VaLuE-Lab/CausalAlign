from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch
from torch import nn


def temporal_transition(hidden: torch.Tensor) -> torch.Tensor:
    """Fixed Gamma: spatial mean then [h_t, h_t+1, h_t+1-h_t].

    Accepts [B,F,S,C] or already pooled [B,F,C] and returns [B,F-1,3C].
    """
    if hidden.ndim == 4:
        hidden = hidden.mean(dim=2)
    if hidden.ndim != 3 or hidden.shape[1] < 2:
        raise ValueError(f"hidden state must be [B,F,(S),C] with F>=2, got {tuple(hidden.shape)}")
    first, second = hidden[:, :-1], hidden[:, 1:]
    return torch.cat((first, second, second - first), -1)


class DeltaProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 1024, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers += [nn.Linear(hidden_dim, 128)]
        self.net = nn.Sequential(*layers)

    def forward(self, transition: torch.Tensor) -> torch.Tensor:
        return self.net(transition)


class ProjectorBank(nn.Module):
    """One Q_l per layer, shared by both selected denoising progress points."""

    def __init__(self, layers: Iterable[int], input_dim: int, hidden_dim: int = 1024, depth: int = 2):
        super().__init__()
        self.layers = tuple(int(layer) for layer in layers)
        self.projectors = nn.ModuleDict({str(layer): DeltaProjector(input_dim, hidden_dim, depth) for layer in self.layers})

    def forward(
        self,
        layer: int | Mapping[str, torch.Tensor],
        transition: torch.Tensor | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        # Batched dictionary mode keeps Stage-2 DDP to one forward per update.
        if isinstance(layer, Mapping):
            return {
                key: self.projectors[key.split("_")[0].removeprefix("l")](value)
                for key, value in layer.items()
            }
        key = str(int(layer))
        if key not in self.projectors:
            raise KeyError(f"no projector for layer {layer}")
        if transition is None:
            raise ValueError("a transition tensor is required")
        return self.projectors[key](transition)

