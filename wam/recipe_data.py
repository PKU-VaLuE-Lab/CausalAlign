"""RoboCasa v2 data and compact action/state conversion."""

import functools
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance
from scipy.spatial.transform import Rotation

ACTION_INDEX = np.r_[0:10, 68:73]
STATE_INDEX = np.r_[0:10, 68:77]
CAMERAS = ["robot0_agentview_left", "robot0_eye_in_hand", "robot0_agentview_right"]


def rot6d(matrix):
    return np.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1).astype(
        np.float32
    )


def compact_rows(state, action):
    # Native v2 order is defined by each repo's meta/modality.json, NOT gym's
    # concatenated action order. In v2 the base commands precede the EEF.
    state = np.asarray(state, np.float32)
    action = np.asarray(action, np.float32)
    s = np.concatenate(
        [
            state[:, 7:10],
            rot6d(Rotation.from_quat(state[:, 10:14]).as_matrix()),
            np.clip(20 * (state[:, 14:15] - state[:, 15:16]) - 1, -1, 1),
            state[:, :3],
            rot6d(Rotation.from_quat(state[:, 3:7]).as_matrix()),
        ],
        axis=-1,
    )
    a = np.concatenate(
        [
            action[:, 5:8],
            rot6d(Rotation.from_rotvec(action[:, 8:11]).as_matrix()),
            -action[:, 11:12],
            action[:, :5],
        ],
        axis=-1,
    )
    return s.astype(np.float32), a.astype(np.float32)


def compact_to_native(action):
    a = np.asarray(action, np.float64)
    b1 = a[..., 3:6] / np.maximum(
        np.linalg.norm(a[..., 3:6], axis=-1, keepdims=True), 1e-8
    )
    b2 = a[..., 6:9] - (b1 * a[..., 6:9]).sum(-1, keepdims=True) * b1
    b2 /= np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    matrix = np.stack([b1, b2, np.cross(b1, b2)], axis=-1)
    rotation = (
        Rotation.from_matrix(matrix.reshape(-1, 3, 3))
        .as_rotvec()
        .reshape(a.shape[:-1] + (3,))
    )
    return np.concatenate(
        [a[..., 10:15], a[..., :3], rotation, -a[..., 9:10]], -1
    ).astype(np.float32)


def normalize(values, stats):
    low, high = (
        np.asarray(stats["min"], np.float32),
        np.asarray(stats["max"], np.float32),
    )
    return np.clip((values - low) / np.maximum(high - low, 1e-6) * 2 - 1, -1, 1)


def denormalize(values, stats):
    low, high = (
        np.asarray(stats["min"], np.float32),
        np.asarray(stats["max"], np.float32),
    )
    return (values + 1) * 0.5 * (high - low) + low


def text_path(root, prompt):
    return Path(root) / (hashlib.sha256(prompt.encode()).hexdigest() + ".pt")


@functools.lru_cache(maxsize=64)
def read_array(path):
    return np.load(path, mmap_mode="r")


@functools.lru_cache(maxsize=128)
def text_context(path):
    real = torch.load(path, map_location="cpu", weights_only=True)
    context = torch.zeros(512, 4096, dtype=real.dtype)
    context[: len(real)] = real
    return context


class RecipeDataset(torch.utils.data.Dataset):
    def __init__(self, root, training=True):
        self.root = Path(root)
        self.training = training
        self.meta = json.loads((self.root / "data/index.json").read_text())
        self.stats = json.loads((self.root / "data/normalization.json").read_text())
        self.episodes = self.meta["episodes"]
        self.ends = np.cumsum([max(0, e["length"] - 1) for e in self.episodes])

    def __len__(self):
        return int(self.ends[-1])

    def __getitem__(self, index):
        # Import pinned official decoder directly; no re-encoding or video copy.
        from reference_video_io import decode_video_frames

        epid = int(np.searchsorted(self.ends, index, side="right"))
        ep = self.episodes[epid]
        start = int(index - (self.ends[epid - 1] if epid else 0))
        valid = min(32, ep["length"] - start - 1)
        data = read_array(str(self.root / "data" / ep["array"]))
        row = ep["offset"] + start
        s = data[row : row + 1, :19].copy()
        a = data[row : row + valid, 19:].copy()
        a = np.concatenate([a, np.repeat(a[-1:], 32 - valid, axis=0)])
        act = np.zeros((32, 80), np.float32)
        state = np.zeros((1, 80), np.float32)
        act[:, ACTION_INDEX] = normalize(a, self.stats["action"])
        state[:, STATE_INDEX] = normalize(s, self.stats["state"])
        offsets = [i for i in range(0, 33, 4) if start + i < ep["length"]]
        frames = []
        for cam, (height, width) in zip(CAMERAS, [(256, 320), (128, 160), (128, 160)]):
            path = str(
                Path(ep["root"])
                / ep["video_template"].format(
                    episode_chunk=ep["episode_index"] // ep["chunks_size"],
                    episode_index=ep["episode_index"],
                    video_key="observation.images." + cam,
                )
            )
            frames.append(
                decode_video_frames(path, [start + i for i in offsets], height, width)
            )
        videos = []
        factors = (
            [1 + random.uniform(-0.2, 0.2) for _ in range(3)]
            if self.training
            else [1] * 3
        )
        for t in range(len(offsets)):
            im = Image.new("RGB", (320, 384))
            for images, xy in zip(frames, [(0, 0), (0, 256), (160, 256)]):
                im.paste(images[t], xy)
            if self.training:
                for enhancer, factor in zip(
                    [
                        ImageEnhance.Brightness,
                        ImageEnhance.Contrast,
                        ImageEnhance.Color,
                    ],
                    factors,
                ):
                    im = enhancer(im).enhance(factor)
            videos.append(np.asarray(im))
        videos += [videos[-1]] * (9 - len(videos))
        video = (
            torch.from_numpy(np.stack(videos))
            .permute(3, 0, 1, 2)
            .float()
            .div_(127.5)
            .sub_(1)
        )
        dim_mask = torch.zeros(80, dtype=torch.bool)
        dim_mask[ACTION_INDEX] = True
        return dict(
            video=video,
            action=torch.from_numpy(act),
            proprio=torch.from_numpy(state),
            prompt=ep["prompt"],
            context=text_context(
                str(text_path(self.root / "text_cache", ep["prompt"]))
            ),
            context_mask=torch.ones(512, dtype=torch.bool),
            action_dim_mask=dim_mask,
            action_is_pad=torch.arange(32) >= valid,
            image_is_pad=torch.arange(9) >= len(offsets),
        )
