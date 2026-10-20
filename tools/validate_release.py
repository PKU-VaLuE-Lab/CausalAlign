#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from causalalign.config import load_config
from causalalign.data import RecordDataset
from causalalign.spec import PAPER_SPEC
from causalalign.training.common import validate_targets


def require_values(mapping: dict, expected: dict, scope: str) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"{scope}.{key} must be {value!r}, got {mapping.get(key)!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-fast validation before expensive training")
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--stage3-config", required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--check-wan", action="store_true")
    args = parser.parse_args()
    configs = [load_config(args.stage1_config), load_config(args.stage2_config), load_config(args.stage3_config)]
    if int(configs[0]["optimization"]["max_updates"]) != PAPER_SPEC.stage1_updates:
        raise ValueError("Stage 1 is not configured for 40,000 updates")
    require_values(configs[0]["model"], {"state_dim": 4096, "width": 512, "depth": 2}, "stage1.model")
    require_values(configs[0]["optimization"], {
        "batch_size": 4,
        "gradient_accumulation": 1,
        "phase1_learning_rate": 0.0002,
        "phase2_learning_rate": 0.000005,
        "phase1_warmup_steps": 500,
        "phase2_warmup_steps": 500,
        "weight_decay": 0.0001,
    }, "stage1.optimization")
    require_values(configs[1]["model"], {"input_dim": 15360, "hidden_dim": 1024, "depth": 2}, "stage2.model")
    require_values(configs[1]["optimization"], {
        "max_updates": 50000,
        "warmup_steps": 1000,
        "batch_size": 4,
        "gradient_accumulation": 1,
        "learning_rate": 0.0002,
        "weight_decay": 0.01,
    }, "stage2.optimization")
    require_values(configs[2]["lora"], {
        "rank": 16,
        "alpha": 16,
        "dropout": 0.0,
        "modules": ["self.to_q", "self.to_k", "self.to_v", "self.to_out.0", "cross.to_q", "cross.to_k", "cross.to_v", "cross.to_out.0"],
        "experts": ["transformer_2"],
    }, "stage3.lora")
    require_values(configs[2]["optimization"], {
        "max_updates": 12,
        "warmup_steps": 2,
        "batch_size": 1,
        "gradient_accumulation": 9,
        "learning_rate": 0.000007,
        "weight_decay": 0.01,
    }, "stage3.optimization")
    for config in configs[1:]:
        target = config["targets"]
        validate_targets(
            [int(x) for x in target["layers"]],
            [int(x) for x in target["progress"]],
            {int(key): float(value) for key, value in target["progress_weights"].items()},
        )
    if args.check_data:
        for config in configs:
            for name in ("train_index", "val_index"):
                dataset = RecordDataset(config["data"][name])
                sample = dataset[0]
                if not all(torch.isfinite(value).all() for value in sample.values() if isinstance(value, torch.Tensor) and value.is_floating_point()):
                    raise ValueError(f"non-finite value in {config['data'][name]}")
    if args.check_wan:
        from diffusers import UniPCMultistepScheduler, WanTransformer3DModel

        model_path = configs[2]["wan"]["model_path"]
        local = Path(model_path).expanduser().is_dir()
        model_config = WanTransformer3DModel.load_config(
            model_path, subfolder="transformer_2", local_files_only=local
        )
        scheduler_config = UniPCMultistepScheduler.load_config(
            model_path, subfolder="scheduler", local_files_only=local
        )
        require_values(model_config, {
            "in_channels": 36,
            "num_layers": 40,
            "num_attention_heads": 40,
            "attention_head_dim": 128,
            "text_dim": 4096,
        }, "wan.transformer_2")
        require_values(scheduler_config, {
            "num_train_timesteps": 1000,
            "prediction_type": "flow_prediction",
            "use_flow_sigmas": True,
        }, "wan.scheduler")
    print("CausalAlign release contract: OK")


if __name__ == "__main__":
    main()

