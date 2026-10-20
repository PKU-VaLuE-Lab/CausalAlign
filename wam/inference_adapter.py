"""RoboCasa observation and action conversion for policy inference."""

import numpy as np
import torch
from PIL import Image
from recipe_data import (
    ACTION_INDEX,
    STATE_INDEX,
    CAMERAS,
    compact_rows,
    compact_to_native,
    normalize,
    denormalize,
)

STATE_FIELDS = [
    ("base_position", 3),
    ("base_rotation", 4),
    ("end_effector_position_relative", 3),
    ("end_effector_rotation_relative", 4),
    ("gripper_qpos", 2),
]
ACTION_FIELDS = [
    ("base_motion", 4),
    ("control_mode", 1),
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
]


def observation_inputs(obs, stats):
    native = np.concatenate(
        [np.asarray(obs["state." + key]).reshape(n) for key, n in STATE_FIELDS]
    )[None]
    state, _ = compact_rows(native, np.zeros((1, 12), np.float32))
    unified = np.zeros((1, 80), np.float32)
    unified[:, STATE_INDEX] = normalize(state, stats["state"])
    canvas = Image.new("RGB", (320, 384))
    for cam, size, xy in zip(
        CAMERAS, [(320, 256), (160, 128), (160, 128)], [(0, 0), (0, 256), (160, 256)]
    ):
        canvas.paste(
            Image.fromarray(np.asarray(obs["video." + cam], np.uint8)).resize(
                size, Image.LANCZOS
            ),
            xy,
        )
    image = torch.from_numpy(np.array(canvas)).permute(2, 0, 1).float() / 127.5 - 1
    return image, torch.from_numpy(unified)


def decode_action(unified, stats):
    compact = denormalize(np.asarray(unified)[..., ACTION_INDEX], stats["action"])
    native = compact_to_native(compact)
    native[..., 5:11] = np.clip(native[..., 5:11], -1, 1)
    native[..., 4] = np.where(native[..., 4] >= 0.5, 1.0, -1.0)
    native[..., 11] = np.where(native[..., 11] >= 0.5, 1.0, -1.0)
    result = {}
    start = 0
    for key, n in ACTION_FIELDS:
        result["action." + key] = native[..., start : start + n]
        start += n
    return result
