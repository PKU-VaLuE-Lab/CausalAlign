from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F

from causalalign.geometry import pose_loss


def stage1_objective(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch["state_t"], batch["state_t1"], batch["state_t2"])
    next_token = output["target_next_token"]
    transition = output["transition"]
    l_state = F.mse_loss(transition["state_t1"].float(), batch["state_t1"].float())
    l_geometry = F.mse_loss(transition["geometry_delta"].float(), batch["geometry_delta"].float())
    l_visual = F.mse_loss(transition["visual_delta"].float(), batch["visual_delta"].float())
    l_transition = (
        float(weights["transition_state"]) * l_state
        + float(weights["transition_geometry"]) * l_geometry
        + float(weights["transition_visual"]) * l_visual
    )
    l_temporal = F.mse_loss(output["next_token"].float(), next_token.float())
    camera = output["camera"]
    l_pose, pose_metrics = pose_loss(
        camera["rotation"], camera["translation"], batch["relative_pose"], batch["scene_scale"]
    )
    total = l_transition + float(weights["lambda_temporal"]) * l_temporal + float(weights["lambda_pose"]) * l_pose
    metrics = {
        "loss": total.detach(),
        "transition": l_transition.detach(),
        "transition_state": l_state.detach(),
        "transition_geometry": l_geometry.detach(),
        "transition_visual": l_visual.detach(),
        "temporal": l_temporal.detach(),
        **pose_metrics,
    }
    return total, metrics


def stage2_objective(
    projectors: torch.nn.Module,
    transitions: Mapping[str, torch.Tensor],
    teacher_token: torch.Tensor,
    layers: list[int],
    progress: list[int],
    progress_weights: Mapping[int, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    total = teacher_token.new_zeros((), dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    target = teacher_token.detach().float()
    predictions = projectors(transitions)
    for layer in layers:
        for step in progress:
            key = f"l{layer}_s{step}"
            prediction = predictions[key].float()
            loss = F.mse_loss(prediction, target)
            total = total + float(progress_weights[int(step)]) * loss
            metrics[f"map/{key}"] = loss.detach()
    metrics["loss"] = total.detach()
    return total, metrics


def stage3_objective(
    projectors: torch.nn.Module,
    camera_decoder: torch.nn.Module,
    transitions: Mapping[str, torch.Tensor],
    teacher_token: torch.Tensor,
    relative_pose: torch.Tensor,
    scene_scale: torch.Tensor,
    layers: list[int],
    progress: list[int],
    progress_weights: Mapping[int, float],
    lambda_delta: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    delta_total = teacher_token.new_zeros((), dtype=torch.float32)
    pose_total = teacher_token.new_zeros((), dtype=torch.float32)
    metrics: dict[str, torch.Tensor] = {}
    target = teacher_token.detach().float()
    for layer in layers:
        for step in progress:
            key = f"l{layer}_s{step}"
            token = projectors(layer, transitions[key])
            l_delta = F.mse_loss(token.float(), target)
            camera = camera_decoder(token)
            l_pose, pose_metrics = pose_loss(camera["rotation"], camera["translation"], relative_pose, scene_scale)
            weight = float(progress_weights[int(step)])
            delta_total = delta_total + weight * l_delta
            pose_total = pose_total + weight * l_pose
            metrics[f"delta/{key}"] = l_delta.detach()
            metrics[f"pose/{key}"] = l_pose.detach()
            metrics[f"rotation/{key}"] = pose_metrics["rotation_geodesic"]
    total = pose_total + float(lambda_delta) * delta_total
    metrics.update(loss=total.detach(), pose=pose_total.detach(), delta=delta_total.detach())
    return total, metrics

