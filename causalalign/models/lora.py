from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = float(alpha) / rank
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        update = torch.nn.functional.linear(torch.nn.functional.linear(self.dropout(value), self.lora_A), self.lora_B)
        return self.base(value) + update * self.scale


def _resolve(parent: nn.Module, dotted: str) -> tuple[nn.Module, str]:
    parts = dotted.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    transformer: nn.Module,
    layers: Iterable[int],
    module_names: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> list[str]:
    """Inject adapters only in selected transformer blocks."""
    blocks = getattr(transformer, "blocks", None)
    if blocks is None:
        raise AttributeError("Wan transformer does not expose .blocks")
    replaced: list[str] = []
    for layer in layers:
        block = blocks[int(layer)]
        for relative in module_names:
            # Explicit paths can target both self- and cross-attention. Short
            # projection names remain supported for Wan variants that call
            # self-attention either ``attn1`` or ``self_attn``.
            if relative.startswith("self."):
                projection = relative.removeprefix("self.")
                candidates = (f"attn1.{projection}", f"self_attn.{projection}")
            elif relative.startswith("cross."):
                projection = relative.removeprefix("cross.")
                candidates = (f"attn2.{projection}", f"cross_attn.{projection}")
            elif relative.startswith(("attn1.", "attn2.", "self_attn.", "cross_attn.")):
                candidates = (relative,)
            else:
                candidates = (f"attn1.{relative}", f"self_attn.{relative}", relative)
            found = None
            for candidate in candidates:
                try:
                    parent, name = _resolve(block, candidate)
                    current = parent[int(name)] if name.isdigit() else getattr(parent, name)
                except (AttributeError, IndexError, KeyError):
                    continue
                if isinstance(current, nn.Linear):
                    found = parent, name, current, candidate
                    break
            if found is None:
                raise AttributeError(f"cannot find linear {relative!r} in Wan block {layer}")
            parent, name, current, candidate = found
            wrapped = LoRALinear(current, rank, alpha, dropout)
            if name.isdigit():
                parent[int(name)] = wrapped
            else:
                setattr(parent, name, wrapped)
            replaced.append(f"blocks.{layer}.{candidate}")
    return replaced


def freeze_except_lora(module: nn.Module) -> None:
    module.requires_grad_(False)
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.lora_A.requires_grad_(True)
            child.lora_B.requires_grad_(True)


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in module.state_dict().items() if name.endswith(("lora_A", "lora_B"))}


def load_lora_state_dict(module: nn.Module, state: dict[str, torch.Tensor]) -> None:
    expected = set(lora_state_dict(module))
    if set(state) != expected:
        raise ValueError(f"LoRA key mismatch: missing={sorted(expected-set(state))[:8]}, unexpected={sorted(set(state)-expected)[:8]}")
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name]))


def merge_lora_(module: nn.Module) -> None:
    """Merge every adapter into its frozen base Linear for inference."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            base = child.base
            with torch.no_grad():
                base.weight.add_((child.lora_B @ child.lora_A).to(base.weight) * child.scale)
            setattr(module, name, base)
        else:
            merge_lora_(child)

