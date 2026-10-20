#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from causalalign.models.lora import inject_lora, load_lora_state_dict, merge_lora_
from causalalign.spec import PAPER_SPEC


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a Stage-3 checkpoint into Wan transformer_2")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--wan-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from diffusers import WanTransformer3DModel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "causalalign-paper-lora-v1" or checkpoint.get("paper_spec_sha256") != PAPER_SPEC.fingerprint():
        raise ValueError("not a compatible CausalAlign Stage-3 checkpoint")
    transformer = WanTransformer3DModel.from_pretrained(
        args.wan_model,
        subfolder="transformer_2",
        torch_dtype=torch.bfloat16,
        local_files_only=Path(args.wan_model).is_dir(),
        low_cpu_mem_usage=True,
    ).to(args.device)
    config = checkpoint["config"]["lora"]
    inject_lora(transformer, PAPER_SPEC.target_layers, config["modules"], int(config["rank"]), float(config["alpha"]), float(config["dropout"]))
    load_lora_state_dict(transformer, checkpoint["lora"])
    merge_lora_(transformer)
    transformer.save_pretrained(Path(args.output) / "transformer_2", safe_serialization=True)


if __name__ == "__main__":
    main()
