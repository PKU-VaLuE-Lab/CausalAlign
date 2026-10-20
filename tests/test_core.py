from __future__ import annotations

import torch
from torch import nn
import pytest

from causalalign.geometry import opengl_to_opencv_c2w, relative_pose, so3_exp
from causalalign.integrations.wan22 import Wan22TransitionBackend
from causalalign.losses import stage1_objective, stage2_objective, stage3_objective
from causalalign.models.lora import LoRALinear, freeze_except_lora, inject_lora, merge_lora_
from causalalign.models.projector import ProjectorBank, temporal_transition
from causalalign.models.teacher import StructuredDeltaTokenModel, TOKEN_LAYOUT
from causalalign.spec import PAPER_SPEC
from causalalign.training.common import cosine_warmup_scheduler, two_phase_cosine_scheduler


def test_paper_contract():
    assert (PAPER_SPEC.ego_dim, PAPER_SPEC.geometry_dim, PAPER_SPEC.visual_dim) == (64, 40, 24)
    assert PAPER_SPEC.target_layers == (10, 12, 14, 20)
    assert PAPER_SPEC.target_progress == (398, 799)
    value = TOKEN_LAYOUT.split(torch.zeros(2, 128))
    assert {key: item.shape[-1] for key, item in value.items()} == {"ego": 64, "geometry": 40, "visual": 24}


def test_wan_progress_indexes_descending_scheduler_not_native_timestep():
    class FakeScheduler:
        def set_timesteps(self, count, device):
            assert count == 1000
            self.timesteps = torch.arange(1000, 0, -1, device=device)
            self.sigmas = self.timesteps.float() / 1000

    backend = Wan22TransitionBackend.__new__(Wan22TransitionBackend)
    nn.Module.__init__(backend)
    backend.scheduler = FakeScheduler()
    early_t, early_sigma = backend.scheduler_point(398, torch.device("cpu"))
    late_t, late_sigma = backend.scheduler_point(799, torch.device("cpu"))
    assert (early_t.item(), late_t.item()) == (602, 201)
    assert early_t > late_t and early_sigma > late_sigma
    assert torch.isclose(early_sigma, torch.tensor(0.602))
    with pytest.raises(ValueError, match="scheduler index"):
        backend.scheduler_point(1000, torch.device("cpu"))


def test_pose_conventions():
    first = torch.eye(4)
    second = torch.eye(4)
    second[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    assert torch.allclose(relative_pose(first, second), second)
    converted = opengl_to_opencv_c2w(first)
    assert torch.allclose(converted.diag(), torch.tensor([1.0, -1.0, -1.0, 1.0]))
    assert torch.allclose(so3_exp(torch.zeros(2, 3)), torch.eye(3).expand(2, 3, 3))


def test_stage1_forward_and_loss_backward():
    model = StructuredDeltaTokenModel(state_dim=16, width=32, depth=1, geometry_dim=8, visual_dim=6)
    batch = {
        "state_t": torch.randn(3, 16),
        "state_t1": torch.randn(3, 16),
        "state_t2": torch.randn(3, 16),
        "geometry_delta": torch.randn(3, 8),
        "visual_delta": torch.randn(3, 6),
        "relative_pose": torch.eye(4).expand(3, 4, 4).clone(),
        "scene_scale": torch.ones(3),
    }
    loss, metrics = stage1_objective(model, batch, {
        "transition_state": 1.0,
        "transition_geometry": 1.0,
        "transition_visual": 1.0,
        "lambda_temporal": 0.1,
        "lambda_pose": 0.2,
    })
    loss.backward()
    assert torch.isfinite(loss)
    assert "rotation_geodesic" in metrics
    assert model.encoder.ego.weight.grad is not None


def test_projector_and_stage2_contract():
    hidden = torch.randn(2, 3, 5, 12)
    transition = temporal_transition(hidden)
    assert transition.shape == (2, 2, 36)
    bank = ProjectorBank(PAPER_SPEC.target_layers, input_dim=36, hidden_dim=24, depth=1)
    transitions = {f"l{layer}_s{step}": transition[:, 0] for layer in PAPER_SPEC.target_layers for step in PAPER_SPEC.target_progress}
    loss, _ = stage2_objective(bank, transitions, torch.randn(2, 128), list(PAPER_SPEC.target_layers), list(PAPER_SPEC.target_progress), {398: 0.4, 799: 0.6})
    loss.backward()
    assert bank.projectors["14"].net[-1].weight.grad is not None


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.to_k = nn.Linear(8, 8)
        self.to_v = nn.Linear(8, 8)
        self.to_out = nn.Sequential(nn.Linear(8, 8))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attention()
        self.attn2 = Attention()


class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(21))


def test_only_selected_lora_is_trainable():
    model = FakeTransformer()
    replaced = inject_lora(model, [10, 12, 14, 20], ["to_q", "to_v"], 2, 2)
    freeze_except_lora(model)
    assert len(replaced) == 8
    assert isinstance(model.blocks[14].attn1.to_q, LoRALinear)
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable and all(name.endswith(("lora_A", "lora_B")) for name in trainable)
    assert not isinstance(model.blocks[13].attn1.to_q, LoRALinear)


def test_explicit_self_and_cross_attention_lora_paths():
    model = FakeTransformer()
    replaced = inject_lora(model, [14], ["self.to_q", "cross.to_out.0"], 2, 2)
    assert replaced == ["blocks.14.attn1.to_q", "blocks.14.attn2.to_out.0"]
    assert isinstance(model.blocks[14].attn1.to_q, LoRALinear)
    assert isinstance(model.blocks[14].attn2.to_out[0], LoRALinear)


def test_lora_merge_preserves_output():
    layer = nn.Sequential(LoRALinear(nn.Linear(8, 4), rank=2, alpha=2))
    nn.init.normal_(layer[0].lora_B)
    value = torch.randn(3, 8)
    expected = layer(value)
    merge_lora_(layer)
    assert isinstance(layer[0], nn.Linear)
    assert torch.allclose(layer(value), expected, atol=1e-6)


def test_stage3_gradient_crosses_frozen_projector_and_camera_decoder():
    layers, progress = [10, 12, 14, 20], [398, 799]
    bank = ProjectorBank(layers, input_dim=36, hidden_dim=24, depth=1).requires_grad_(False)
    teacher = StructuredDeltaTokenModel(state_dim=16, width=32, depth=1, geometry_dim=8, visual_dim=6)
    camera = teacher.camera_decoder.requires_grad_(False)
    transitions = {f"l{layer}_s{step}": torch.randn(2, 36, requires_grad=True) for layer in layers for step in progress}
    target_pose = torch.eye(4).expand(2, 4, 4).clone()
    loss, _ = stage3_objective(bank, camera, transitions, torch.randn(2, 128), target_pose, torch.ones(2), layers, progress, {398: 0.5, 799: 0.5}, 0.2)
    loss.backward()
    assert all(value.grad is not None for value in transitions.values())
    assert all(value.grad is None for value in bank.parameters())
    assert all(value.grad is None for value in camera.parameters())


def test_release_learning_rate_schedules_are_finite():
    parameter = nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    schedule = two_phase_cosine_scheduler(
        optimizer,
        phase1_steps=10,
        total_steps=40,
        phase1_warmup_steps=2,
        phase2_warmup_steps=2,
        phase2_peak_ratio=0.025,
    )
    values = []
    for _ in range(40):
        optimizer.step()
        schedule.step()
        values.append(schedule.get_last_lr()[0])
    assert all(0.0 <= value <= 2e-4 for value in values)
    optimizer = torch.optim.AdamW([parameter], lr=7e-6)
    schedule = cosine_warmup_scheduler(optimizer, warmup_steps=2, total_steps=12, minimum_ratio=0.1)
    for _ in range(12):
        optimizer.step()
        schedule.step()
        assert 0.0 < schedule.get_last_lr()[0] <= 7e-6

