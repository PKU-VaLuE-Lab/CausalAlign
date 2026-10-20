from __future__ import annotations

import torch
import torch.nn.functional as F


def skew(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError(f"expected [...,3], got {tuple(vector.shape)}")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(
        *vector.shape[:-1], 3, 3
    )


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues exponential map."""
    work = omega.float() if omega.dtype in (torch.float16, torch.bfloat16) else omega
    theta2 = work.square().sum(-1, keepdim=True)
    theta = theta2.clamp_min(1e-12).sqrt()
    small = theta2 < 1e-6
    a = torch.where(small, 1 - theta2 / 6 + theta2.square() / 120, torch.sin(theta) / theta)
    b = torch.where(
        small,
        0.5 - theta2 / 24 + theta2.square() / 720,
        (1 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    k = skew(work)
    eye = torch.eye(3, dtype=work.dtype, device=work.device).expand(*work.shape[:-1], 3, 3)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    if rotation.shape[-2:] != (3, 3):
        raise ValueError(f"expected [...,3,3], got {tuple(rotation.shape)}")
    r = rotation.float()
    cosine = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    theta = torch.acos(cosine)
    vee = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1)
    scale = theta / (2 * torch.sin(theta).clamp_min(1e-7))
    small = theta < 1e-4
    return torch.where(small[..., None], 0.5 * vee, scale[..., None] * vee)


def relative_pose(c2w_t: torch.Tensor, c2w_t1: torch.Tensor) -> torch.Tensor:
    """Paper convention: Delta T_t = inverse(T_t) @ T_(t+1)."""
    if c2w_t.shape[-2:] != (4, 4) or c2w_t1.shape != c2w_t.shape:
        raise ValueError("camera poses must have matching [...,4,4] shapes")
    return torch.linalg.inv(c2w_t) @ c2w_t1


def opengl_to_opencv_c2w(c2w: torch.Tensor) -> torch.Tensor:
    conversion = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], device=c2w.device, dtype=c2w.dtype))
    return c2w @ conversion


def geodesic_rotation_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    relative = target.detach().float().transpose(-1, -2) @ prediction.float()
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5).clamp(-1, 1)
    vee = torch.stack(
        (relative[..., 2, 1] - relative[..., 1, 2], relative[..., 0, 2] - relative[..., 2, 0], relative[..., 1, 0] - relative[..., 0, 1]),
        -1,
    )
    sine = 0.5 * torch.linalg.vector_norm(vee, dim=-1)
    return torch.atan2(sine, cosine)


def normalized_translation_loss(
    prediction: torch.Tensor, target: torch.Tensor, scene_scale: torch.Tensor
) -> torch.Tensor:
    scale = scene_scale.detach().float().clamp_min(1e-6)
    while scale.ndim < prediction.ndim:
        scale = scale.unsqueeze(-1)
    return F.smooth_l1_loss(prediction.float() / scale, target.detach().float() / scale, reduction="none").mean(-1)


def pose_loss(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    target_transform: torch.Tensor,
    scene_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_r = target_transform[..., :3, :3]
    target_t = target_transform[..., :3, 3]
    rotation_loss = geodesic_rotation_loss(rotation, target_r)
    translation_loss = normalized_translation_loss(translation, target_t, scene_scale)
    total = rotation_loss + translation_loss
    return total.mean(), {
        "pose": total.mean().detach(),
        "rotation_geodesic": rotation_loss.mean().detach(),
        "translation_smooth_l1": translation_loss.mean().detach(),
    }

