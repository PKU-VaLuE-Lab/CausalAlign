"""RoboCasa PandaOmron 12D action / 16D state convention from modality.json."""

import numpy as np

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
CAMERAS = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def pack_state(obs):
    return np.concatenate(
        [np.asarray(obs["state." + name]).reshape(n) for name, n in STATE_FIELDS]
    ).astype(np.float32)


def unpack_action(action):
    action = np.asarray(action, dtype=np.float32).reshape(12)
    result, start = {}, 0
    for name, n in ACTION_FIELDS:
        result["action." + name] = action[start : start + n].copy()
        start += n
    return result


def pack_cameras(obs):
    # Official gym wrapper already flips rendered images to the data convention.
    return [obs["video." + name] for name in CAMERAS]
