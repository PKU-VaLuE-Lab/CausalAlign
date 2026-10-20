"""Run one target-split episode against a policy server, rendering with CPU llvmpipe."""
import os
import sys

if os.environ.get("RC_RENDER_BACKEND") != "llvmpipe":
    from cpu_environment import enable_cpu

    os.execve(sys.executable, [sys.executable] + sys.argv, enable_cpu(os.environ))

import argparse
import base64
import ctypes
import faulthandler
import hashlib
import json
import random
import time
from pathlib import Path
import urllib.request
import gymnasium as gym
import numpy as np
import mujoco
import robocasa  # registers the robocasa/<task> gym environments
from OpenGL import GL
from adapter import pack_state, pack_cameras, unpack_action
from common import ROOT, write

faulthandler.enable()


def rpc(port, route, data):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/{route}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--worker", required=True)
    a = p.parse_args()
    c = json.loads(a.case.read_text())
    task, seed, group = c["task"], c["seed"], c["group"]
    out = ROOT / "results" / task / f"{seed}.json"
    assert not out.exists()
    loaded = rpc(a.port, "load", dict(step=50000, group=group))
    assert (
        loaded["step"] == 50000 and loaded["group"] == group and loaded["strict_load"]
    )
    # Isolate the optional process-global asset cache; no physics/render settings change.
    assert mujoco.__version__ == "3.3.1"
    lib = ctypes.CDLL(str(Path(mujoco.__file__).parent / "libmujoco.so.3.3.1"))
    lib.mj_globalCache.argtypes = []
    lib.mj_globalCache.restype = ctypes.c_void_p
    lib.mj_setCacheSize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mj_setCacheSize.restype = None
    lib.mj_setCacheSize(lib.mj_globalCache(), 0)
    random.seed(seed)
    np.random.seed(seed)
    began = time.monotonic()
    env = gym.make(
        f"robocasa/{task}",
        split="target",
        seed=seed,
        camera_widths=256,
        camera_heights=256,
    )
    try:
        obs, _ = env.reset(seed=seed)
        raw = env.unwrapped.env
        assert int(raw.control_freq) == 20
        raw.sim._render_context_offscreen.gl_ctx.make_current()
        while GL.glGetError() != GL.GL_NO_ERROR:
            pass
        vendor = GL.glGetString(GL.GL_VENDOR).decode()
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower(), (vendor, renderer)
        state = np.asarray(raw.sim.get_state().flatten(), dtype=np.float64)
        initial_hash = hashlib.sha256(state.tobytes()).hexdigest()
        image_hashes = [
            hashlib.sha256(x.tobytes()).hexdigest() for x in pack_cameras(obs)
        ]
        instruction = obs["annotation.human.task_description"]
        case = ROOT / "paired_cases" / task / f"{seed}.npz"
        case.parent.mkdir(parents=True, exist_ok=True)
        hashes = case.with_suffix(".json")
        signature = dict(
            state=initial_hash, images=image_hashes, instruction=instruction
        )
        if not case.exists():
            meta = json.dumps(
                raw.get_ep_meta(),
                default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
            )
            with case.with_suffix(".tmp").open("wb") as f:
                np.savez_compressed(
                    f, xml=raw.sim.model.get_xml(), meta=meta, state=state
                )
            case.with_suffix(".tmp").replace(case)
            write(hashes, signature)
        else:
            with np.load(case, allow_pickle=False) as saved:
                saved_state = saved["state"].copy()
                saved_xml = str(saved["xml"])
                saved_meta = json.loads(str(saved["meta"]))
            expected = json.loads(hashes.read_text()) if hashes.exists() else None
            if signature != expected:
                raw.set_ep_meta(saved_meta)
                raw.reset()
                raw.reset_from_xml_string(saved_xml)
                raw.sim.set_state_from_flattened(saved_state)
                raw.sim.forward()
                raw.update_state()
                obs = env.unwrapped.get_observation(
                    raw._get_observations(force_update=True)
                )
                raw.sim._render_context_offscreen.gl_ctx.make_current()
                state = np.asarray(raw.sim.get_state().flatten(), dtype=np.float64)
                initial_hash = hashlib.sha256(state.tobytes()).hexdigest()
                image_hashes = [
                    hashlib.sha256(x.tobytes()).hexdigest() for x in pack_cameras(obs)
                ]
                instruction = obs["annotation.human.task_description"]
                signature = dict(
                    state=initial_hash, images=image_hashes, instruction=instruction
                )
            assert np.array_equal(state, saved_state)
            if expected is None:
                write(hashes, signature)
            else:
                assert signature == expected, (
                    f"Initial state/image mismatch: {task}/{seed}"
                )
        actions = []
        success = False
        inference = 0.0
        executed = 0
        path_length = 0.0
        previous = pack_state(obs)[:3].copy()
        replay_checked = False
        for t in range(c["horizon"]):
            if not actions:
                start = time.monotonic()
                query = dict(
                    group=group,
                    cameras=[
                        base64.b64encode(x.tobytes()).decode()
                        for x in pack_cameras(obs)
                    ],
                    state=pack_state(obs).tolist(),
                    instruction=instruction,
                    noise_seed=seed * 100000 + c["task_index"] * 1000 + t // 8,
                )
                reply = rpc(a.port, "predict", query)
                assert (
                    reply["group"] == group
                    and reply["checkpoint"] == loaded["checkpoint"]
                )
                # One duplicated request per model checks independence from intervening clients.
                if c.get("check_request_replay") and not replay_checked:
                    repeated = rpc(a.port, "predict", query)
                    assert np.array_equal(
                        np.asarray(reply["action"]), np.asarray(repeated["action"])
                    )
                    replay_checked = True
                    write(
                        ROOT / "control" / f"replay_check.{a.worker}.json",
                        dict(exact=True, task=task, seed=seed, time=time.time()),
                    )
                actions = reply["action"][:8]
                inference += time.monotonic() - start
            obs, reward, terminated, truncated, info = env.step(
                unpack_action(actions.pop(0))
            )
            current = pack_state(obs)[:3]
            path_length += float(np.linalg.norm(current - previous))
            previous = current.copy()
            executed = t + 1
            if executed % 100 == 0:
                write(
                    ROOT / "control" / f"progress.{a.worker}.json",
                    dict(
                        **c,
                        steps=executed,
                        seconds=time.monotonic() - began,
                        time=time.time(),
                        worker=a.worker,
                    ),
                )
            success = bool(info["success"])
            if success:
                break
            if terminated or truncated:
                raise RuntimeError(
                    f"{task} terminated at {executed} before horizon {c['horizon']}"
                )
        row = dict(
            **c,
            checkpoint_step=50000,
            checkpoint=loaded["checkpoint"],
            split="target",
            success=success,
            steps=executed,
            seconds=time.monotonic() - began,
            inference_seconds=inference,
            base_path_length=path_length,
            initial_state_sha256=initial_hash,
            initial_image_sha256=image_hashes,
            instruction=instruction,
            render_backend="llvmpipe",
            renderer=renderer,
            worker=a.worker,
            time=time.time(),
        )
        write(out, row)
        print("EPISODE", json.dumps(row), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
