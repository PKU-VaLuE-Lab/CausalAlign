from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
import torch


class WanI2VPreprocessor:
    """Build reusable Wan I2V latents/conditioning from a real clip.

    This adapter follows the public Diffusers component interfaces and stores
    inputs before the transformer. It never caches trainable hidden states.
    """

    def __init__(self, args: Any):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanTransformer3DModel
        from transformers import UMT5EncoderModel

        self.device = torch.device(args.device)
        local = Path(args.wan_model).expanduser().is_dir()
        vae = AutoencoderKLWan.from_pretrained(
            args.wan_model,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local,
            low_cpu_mem_usage=True,
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            args.wan_model,
            subfolder="text_encoder",
            torch_dtype=torch.bfloat16,
            local_files_only=local,
            low_cpu_mem_usage=True,
        )
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            args.wan_model,
            vae=vae,
            text_encoder=text_encoder,
            transformer=None,
            transformer_2=None,
            torch_dtype=torch.bfloat16,
            local_files_only=local,
        ).to(self.device)
        transformer_config = WanTransformer3DModel.load_config(
            args.wan_model, subfolder="transformer_2", local_files_only=local
        )
        self.transformer_in_channels = int(transformer_config["in_channels"])
        self.transformer_image_dim = transformer_config.get("image_dim")
        self.height, self.width = int(args.height), int(args.width)

    def _vae(self, video: torch.Tensor) -> torch.Tensor:
        encoded = self.pipe.vae.encode(video.to(self.pipe.vae.dtype))
        if hasattr(encoded, "latent_dist"):
            latent = encoded.latent_dist.mean
        elif hasattr(encoded, "latents"):
            latent = encoded.latents
        elif isinstance(encoded, (tuple, list)):
            latent = encoded[0]
        else:
            raise RuntimeError(f"unsupported Wan VAE encoder output: {type(encoded)!r}")
        mean = torch.tensor(self.pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        scale = 1 / torch.tensor(self.pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype).view(1, -1, 1, 1, 1)
        return (latent - mean) * scale

    @torch.inference_mode()
    def encode(self, paths: list[str], caption: str = "") -> dict[str, torch.Tensor]:
        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(ImageOps.exif_transpose(image).convert("RGB").copy())
        video = self.pipe.video_processor.preprocess_video(frames, height=self.height, width=self.width).to(self.device)
        clean = self._vae(video)
        prompt, _ = self.pipe.encode_prompt(prompt=caption, do_classifier_free_guidance=False, device=self.device)
        output = {"clean_latent": clean[0].cpu(), "prompt_embedding": prompt[0].cpu()}
        # I2V-A14B has 36 input channels: noisy latent plus 4-channel mask and
        # a 16-channel first-frame condition. T2V checkpoints skip this block.
        if self.transformer_in_channels > clean.shape[1]:
            conditioned_video = torch.zeros_like(video)
            conditioned_video[:, :, 0] = video[:, :, 0]
            conditioned = self._vae(conditioned_video)
            mask = torch.zeros((1, 4, clean.shape[2], clean.shape[3], clean.shape[4]), device=self.device, dtype=clean.dtype)
            mask[:, :, 0] = 1
            output["condition_latent"] = torch.cat((mask, conditioned), 1)[0].cpu()
            # Wan2.1 I2V uses CLIP image embeddings; Wan2.2-A14B normally
            # leaves image_dim unset and conditions through the latent only.
            if self.transformer_image_dim is not None:
                output["image_embedding"] = self.pipe.encode_image(frames[0], self.device)[0].cpu()
        return output


def create_preprocessor(args: Any) -> WanI2VPreprocessor:
    return WanI2VPreprocessor(args)

