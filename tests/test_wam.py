"""CPU checks for the WAM data, model, and simulator interfaces."""

import inspect
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
WAM = ROOT / "wam"


@pytest.fixture
def wam(monkeypatch):
    for dependency in (
        "av",
        "einops",
        "ftfy",
        "huggingface_hub",
        "modelscope",
        "omegaconf",
        "pandas",
        "pyarrow",
        "rich",
        "scipy",
        "transformers",
    ):
        pytest.importorskip(dependency)
    for path in [WAM / "evaluation", WAM / "src", WAM]:
        monkeypatch.syspath_prepend(str(path))


# WAM data and interfaces


def test_native_action_compact_roundtrip(wam):
    import recipe_data

    rng = np.random.default_rng(42)
    state = rng.normal(size=(8, 16)).astype(np.float32)
    state[:, 3:7] = [0, 0, 0, 1]
    state[:, 10:14] = [0, 0, 0, 1]
    action = rng.normal(size=(8, 12)).astype(np.float32)
    action[:, 8:11] *= 0.1
    compact_state, compact_action = recipe_data.compact_rows(state, action)
    assert compact_state.shape == (8, 19)
    assert compact_action.shape == (8, 15)
    np.testing.assert_allclose(recipe_data.compact_to_native(compact_action), action, rtol=1e-5, atol=2e-6)


def test_dataset_observation_and_action_start_match(wam, monkeypatch, tmp_path):
    import recipe_data
    import reference_video_io
    from PIL import Image

    data = tmp_path / "data"
    data.mkdir()
    (tmp_path / "text_cache").mkdir()
    ep = dict(root=str(tmp_path), array="task_000.npy", offset=0, episode_index=0, length=50, prompt="move",
              chunks_size=1000, video_template="{video_key}/{episode_index}.mp4")
    (data / "index.json").write_text(json.dumps(dict(episodes=[ep])))
    stats = {name: dict(min=[-1] * width, max=[1] * width) for name, width in [("state", 19), ("action", 15)]}
    (data / "normalization.json").write_text(json.dumps(stats))
    values = np.zeros((50, 34), np.float32)
    values[:, 19] = np.arange(50) / 100
    np.save(data / "task_000.npy", values)
    monkeypatch.setattr(recipe_data, "text_context", lambda path: torch.zeros(512, 4096))
    requested = []

    def decode(path, indices, height, width):
        requested.append(indices)
        return [Image.new("RGB", (width, height), (i, i, i)) for i in indices]

    monkeypatch.setattr(reference_video_io, "decode_video_frames", decode)
    sample = recipe_data.RecipeDataset(tmp_path, training=False)[12]
    assert all(indices == list(range(12, 45, 4)) for indices in requested)
    assert sample["video"].shape == (3, 9, 384, 320)
    torch.testing.assert_close(sample["video"][:, 0], torch.full((3, 384, 320), 12 / 127.5 - 1))
    assert sample["action"][0, 0].item() == pytest.approx(0.12)
    assert sample["action_dim_mask"].sum() == 15 and sample["proprio"].shape == (1, 80)
    assert sample["action_is_pad"].sum() == 0 and sample["image_is_pad"].sum() == 0


def test_observation_layout_and_action_decoding(wam):
    import adapter
    import inference_adapter as ia
    import recipe_data

    rng = np.random.default_rng(0)
    obs = {"video." + name: np.full((256, 256, 3), value, np.uint8)
           for name, value in zip(adapter.CAMERAS, [10, 200, 90])}
    state = rng.normal(size=16).astype(np.float32)
    state[3:7] = [0, 0, 0, 1]
    state[10:14] = [0, 0, np.sin(0.1), np.cos(0.1)]
    start = 0
    for key, n in ia.STATE_FIELDS:
        obs["state." + key] = state[start:start + n]
        start += n
    np.testing.assert_array_equal(adapter.pack_state(obs), state)
    stats = {"state": dict(min=[-5.0] * 19, max=[4.0] * 19), "action": dict(min=[-2.0] * 15, max=[2.0] * 15)}
    image, unified = ia.observation_inputs(obs, stats)
    # Left agent view on top, wrist camera bottom-left, right agent view bottom-right.
    assert image.shape == (3, 384, 320)
    pixel = lambda y, x: round(float((image[0, y, x] + 1) * 127.5))
    assert (pixel(100, 100), pixel(300, 50), pixel(300, 250)) == (10, 90, 200)
    assert unified.shape == (1, 80) and np.count_nonzero(unified[0, recipe_data.STATE_INDEX]) == 19
    assert not unified[0, np.setdiff1d(np.arange(80), recipe_data.STATE_INDEX)].any()

    native = np.zeros((4, 12), np.float32)
    native[:, :4] = rng.uniform(-0.5, 0.5, (4, 4))
    native[:, 4] = [1, -1, 1, -1]
    native[:, 5:8] = rng.uniform(-0.5, 0.5, (4, 3))
    native[:, 8:11] = rng.uniform(-0.2, 0.2, (4, 3))
    native[:, 11] = [-1, -1, 1, 1]
    _, compact = recipe_data.compact_rows(np.tile(state, (4, 1)), native)
    model_output = np.zeros((4, 80), np.float32)
    model_output[:, recipe_data.ACTION_INDEX] = recipe_data.normalize(compact, stats["action"])
    decoded = ia.decode_action(model_output, stats)
    packed = np.concatenate([decoded["action." + key] for key, n in ia.ACTION_FIELDS], -1)
    np.testing.assert_allclose(packed, native, atol=1e-5)
    for i in range(4):
        np.testing.assert_array_equal(adapter.unpack_action(packed[i])["action.gripper_close"], native[i, 11:])


def test_prepare_data_converts_lerobot_episodes(wam, tmp_path):
    import pandas as pd
    import prepare_data

    ds = tmp_path / "dataset"
    (ds / "meta").mkdir(parents=True)
    info = dict(fps=20, chunks_size=1000, data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                features={"observation.images.robot0_eye_in_hand": {"dtype": "video"}})
    (ds / "meta/info.json").write_text(json.dumps(info))
    lengths = [5, 7, 6]
    (ds / "meta/episodes.jsonl").write_text("\n".join(json.dumps(dict(episode_index=i, length=n, tasks=[f"task {i}"])) for i, n in enumerate(lengths)))
    rng = np.random.default_rng(1)
    for i, n in enumerate(lengths):
        state = rng.normal(size=(n, 16)).astype(np.float32)
        state[:, 3:7] = [0, 0, 0, 1]
        state[:, 10:14] = [0, 0, 0, 1]
        action = rng.normal(size=(n, 12)).astype(np.float32) * 0.1
        path = ds / info["data_path"].format(episode_chunk=0, episode_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"observation.state": list(state), "action": list(action)}).to_parquet(path)
        video = ds / info["video_path"].format(episode_chunk=0, episode_index=i, video_key="observation.images.robot0_eye_in_hand")
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(b"")
    ids = prepare_data.select_episodes(ds, 2)
    expected = [0, 1, 2]
    random.Random(0).shuffle(expected)
    assert ids == expected[:2]
    out = tmp_path / "out"
    out.mkdir()
    result = prepare_data.convert_task((3, str(ds), ids, out))
    rows = np.load(out / "task_003.npy")
    assert rows.shape == (sum(lengths[i] for i in ids), 34)
    assert [e["offset"] for e in result["index"]] == [0, lengths[ids[0]]]
    assert result["index"][0]["prompt"] == f"task {ids[0]}"


def test_model_config_matches_constructor(wam, monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    from fastwam.runtime import create_fastwam

    monkeypatch.setenv("WAM_ACTION_BACKBONE", "/assets/backbone.pt")
    monkeypatch.setenv("WAM_VIDEO_LORA", "/assets/video_lora")
    cfg = OmegaConf.to_container(OmegaConf.load(WAM / "configs/model.yaml"), resolve=True)
    params = set(inspect.signature(create_fastwam).parameters)
    assert set(cfg) - {"_target_"} <= params
    assert cfg["tokenizer_model_id"] == cfg["model_id"] == "Wan-AI/Wan2.2-TI2V-5B"
    assert cfg["video_dit_config"]["video_attention_mask_mode"] == "first_frame_causal"
    assert cfg["proprio_dim"] == cfg["video_dit_config"]["action_dim"] == cfg["action_dit_config"]["action_dim"] == 80
    assert cfg["video_dit_lora_checkpoint"] == "/assets/video_lora"

    import configure

    for group, tasks in [("atomic_seen", 18), ("composite_seen", 16), ("composite_unseen", 16)]:
        data = tmp_path / "groups" / group / "data"
        data.mkdir(parents=True)
        (data / "index.json").write_text(json.dumps(dict(tasks=tasks, episodes=[{}] * (tasks * 500))))
        stats = {name: dict(min=[0] * width, max=[1] * width) for name, width in [("state", 19), ("action", 15)]}
        (data / "normalization.json").write_text(json.dumps(stats))
        (data.parent / "task_ranges.json").write_text("[]")
    checkpoint = tmp_path / "step_060000.pt"
    checkpoint.write_bytes(b"")
    monkeypatch.setattr(sys, "argv", ["configure.py", "--root", str(tmp_path), "--stage", "target100", "--initial-checkpoint", str(checkpoint)])
    configure.main()
    written = OmegaConf.load(tmp_path / "groups/atomic_seen/config.yaml")
    assert written.initial_checkpoint == str(checkpoint.resolve())
    assert written.model.skip_dit_load_from_pretrain and written.model.video_dit_lora_checkpoint == "/assets/video_lora"


# Real FastWAM modules at reduced width


class TinyVAE:
    """Deterministic stand-in for the Wan VAE: 16x spatial and 4x temporal reduction to 4 channels."""

    temporal_downsample_factor = 4

    def to(self, *args, **kwargs):
        return self

    def encode(self, videos, device=None, **kwargs):
        video = torch.stack(list(videos)) if isinstance(videos, list) else videos
        pooled = F.avg_pool3d(video.float(), (1, 16, 16))[:, :, ::4]
        return torch.cat([pooled, pooled.mean(1, keepdim=True)], 1)


def tiny_fastwam():
    from fastwam.models.wan22.action_dit import ActionDiT
    from fastwam.models.wan22.fastwam import FastWAM
    from fastwam.models.wan22.mot import MoT
    from fastwam.models.wan22.wan_video_dit import WanVideoDiT

    torch.manual_seed(0)
    video = WanVideoDiT(hidden_dim=64, in_dim=4, ffn_dim=128, out_dim=4, text_dim=32, freq_dim=32, eps=1e-6,
                        patch_size=(1, 2, 2), num_heads=2, attn_head_dim=32, num_layers=2, has_image_input=False,
                        seperated_timestep=True, fuse_vae_embedding_in_latents=True, action_dim=80,
                        action_group_causal_mask_mode="group_diagonal", video_attention_mask_mode="first_frame_causal")
    action = ActionDiT(hidden_dim=32, action_dim=80, ffn_dim=64, text_dim=32, freq_dim=32, eps=1e-6,
                       num_heads=2, attn_head_dim=32, num_layers=2)
    mot = MoT(mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False)
    return FastWAM(video_expert=video, action_expert=action, mot=mot, vae=TinyVAE(), text_dim=32, proprio_dim=80)


def test_action_attention_reaches_only_first_frame(wam):
    model = tiny_fastwam()
    mask = model._build_mot_attention_mask(video_seq_len=12, action_seq_len=8, video_tokens_per_frame=4,
                                           device=torch.device("cpu"))
    reach = mask.clone()
    for _ in range(8):
        reach = reach | ((reach.float() @ reach.float()) > 0)
    assert reach[12:, :4].all()
    assert not reach[12:, 4:12].any()
    assert not reach[:4, 4:].any()


def test_infer_action_chunk_and_inactive_dimensions(wam):
    import recipe_data

    torch.set_num_threads(4)
    model = tiny_fastwam().eval()
    image = torch.rand(3, 64, 64) * 2 - 1
    kwargs = dict(prompt=None, input_image=image, proprio=torch.rand(1, 80), context=torch.randn(8, 32),
                  context_mask=torch.ones(8, dtype=torch.bool), action_horizon=32, num_inference_steps=3,
                  active_action_indices=recipe_data.ACTION_INDEX.tolist())
    first = model.infer_action(seed=7, **kwargs)["action"]
    again = model.infer_action(seed=7, **kwargs)["action"]
    other = model.infer_action(seed=8, **kwargs)["action"]
    assert first.shape == (32, 80) and torch.isfinite(first).all()
    torch.testing.assert_close(first, again, rtol=0, atol=0)
    assert not torch.equal(first, other)
    inactive = np.setdiff1d(np.arange(80), recipe_data.ACTION_INDEX)
    assert first[:, inactive].abs().max() < 1e-6


def test_action_loss_ignores_future_video_frames(wam):
    model = tiny_fastwam()
    rng = torch.Generator().manual_seed(3)
    video = torch.rand(2, 3, 9, 64, 64, generator=rng) * 2 - 1
    sample = dict(video=video, action=torch.randn(2, 32, 80, generator=rng), proprio=torch.randn(2, 1, 80, generator=rng),
                  context=torch.randn(2, 8, 32, generator=rng), context_mask=torch.ones(2, 8, dtype=torch.bool),
                  action_dim_mask=torch.ones(2, 80, dtype=torch.bool), action_is_pad=torch.zeros(2, 32, dtype=torch.bool),
                  image_is_pad=torch.zeros(2, 9, dtype=torch.bool))
    changed = dict(sample, video=torch.cat([video[:, :, :1], -video[:, :, 1:]], 2))
    torch.manual_seed(11)
    _, base = model.training_loss(sample)
    torch.manual_seed(11)
    _, future = model.training_loss(changed)
    assert base["loss_action"] == future["loss_action"]
    assert base["loss_video"] != future["loss_video"]


