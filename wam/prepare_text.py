"""Cache UMT5 embeddings of the raw task prompts used by the prepared data."""
import argparse
import json
from pathlib import Path

import torch
from fastwam.models.wan22.helpers.loader import _load_registered_model
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from recipe_data import text_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True, help="Wan2.2-TI2V-5B directory")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    root = a.root.resolve()
    indexes = [root / "data/index.json"] + sorted(root.glob("groups/*/data/index.json"))
    prompts = sorted({e["prompt"] for path in indexes if path.is_file()
                      for e in json.loads(path.read_text())["episodes"]})
    cache = root / "text_cache"
    cache.mkdir(parents=True, exist_ok=True)
    missing = [prompt for prompt in prompts if not text_path(cache, prompt).exists()]
    print("prompts", len(prompts), "missing", len(missing), flush=True)
    if not missing:
        return
    tokenizer = HuggingfaceTokenizer(str(a.model_dir / "google/umt5-xxl"), seq_len=512, clean="whitespace")
    encoder = _load_registered_model(str(a.model_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                                     "wan_video_text_encoder", torch_dtype=torch.bfloat16, device=a.device)
    encoder.eval().requires_grad_(False)
    with torch.inference_mode():
        for i in range(0, len(missing), 16):
            texts = missing[i:i + 16]
            ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
            context = encoder(ids.to(a.device), mask.to(a.device)).cpu()
            for prompt, c, m in zip(texts, context, mask):
                torch.save(c[:int(m.sum())].clone(), text_path(cache, prompt))


if __name__ == "__main__":
    main()
