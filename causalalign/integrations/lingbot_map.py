from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import torch


class LingBotMapExtractor:
    """Concrete adapter for the public ``lingbot_map.models.gct_stream`` API.

    It returns the final camera token concatenated with the mean final-layer
    patch token (4096-D), plus the model depth estimate.  Input preprocessing
    is delegated to LingBot-Map's own loader.
    """

    def __init__(self, args: Any):
        repo = Path(args.lingbot_repo).resolve()
        sys.path.insert(0, str(repo))
        from lingbot_map.models.gct_stream import GCTStream

        self.device = torch.device(args.device)
        self.model = GCTStream(
            img_size=518,
            patch_size=14,
            enable_camera=True,
            enable_depth=True,
            enable_point=False,
            enable_local_point=False,
            enable_track=False,
            enable_3d_rope=True,
            max_frame_num=1024,
            kv_cache_sliding_window=64,
            kv_cache_scale_frames=8,
            kv_cache_cross_frame_special=True,
            kv_cache_include_scale_frames=True,
            use_sdpa=True,
            camera_num_iterations=int(args.camera_num_iterations),
        )
        payload = torch.load(args.lingbot_checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("model", payload)
        normalized = {}
        for key, value in state.items():
            key = key.removeprefix("module.").removeprefix("model.")
            normalized[key] = value
        incompatible = self.model.load_state_dict(normalized, strict=False)
        critical = [x for x in incompatible.missing_keys if x.startswith(("aggregator.", "camera_head.", "depth_head."))]
        if critical:
            raise RuntimeError(f"LingBot checkpoint is missing critical weights: {critical[:8]}")
        self.model.to(self.device).eval()
        self.captured: list[torch.Tensor] = []
        self.handle = self.model.aggregator.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output) -> None:
        layers, patch_start = output[0], int(output[1])
        if patch_start != 6:
            raise RuntimeError(f"unexpected LingBot patch start: {patch_start}")
        self.captured.append(layers[-1].detach())

    @torch.inference_mode()
    def encode(self, paths: list[str]) -> dict[str, torch.Tensor]:
        from lingbot_map.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(paths, mode="crop", image_size=518, patch_size=14).to(self.device)
        self.captured.clear()
        self.model.clean_kv_cache()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            predictions = self.model.inference_streaming(
                images, num_scale_frames=8, keyframe_interval=1, output_device=torch.device("cpu")
            )
        self.model.clean_kv_cache()
        hidden = torch.cat(self.captured, dim=1)[0].float().cpu()
        camera = hidden[:, 0]
        patches = hidden[:, 6:]
        state = torch.cat((camera, patches.mean(1)), -1)
        depth = predictions["depth"]
        if depth.ndim == 5 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        elif depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise RuntimeError(f"unexpected LingBot depth shape: {tuple(depth.shape)}")
        return {"state": state.contiguous(), "depth": depth.float().cpu().contiguous()}


def create_extractor(args: Any) -> LingBotMapExtractor:
    return LingBotMapExtractor(args)

