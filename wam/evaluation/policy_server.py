import base64
import fcntl
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from common import ROOT, TRAIN, GROUPS, write
from recipe_data import ACTION_INDEX, text_context, text_path
from inference_adapter import (
    observation_inputs,
    decode_action,
    STATE_FIELDS,
    ACTION_FIELDS,
)

worker = sys.argv[1]
port = int(sys.argv[2])
gpu = int(sys.argv[3])
torch.set_num_threads(2)
device = torch.device("cuda", 0)
torch.cuda.set_device(device)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
plan = json.loads((ROOT / "PLAN.json").read_text())
cfg = OmegaConf.load(TRAIN / "groups/atomic_seen/config.yaml")
cfg.model.load_text_encoder = True
cfg.model.skip_dit_load_from_pretrain = True
with open(f"/tmp/wam-model-load-{gpu % 4}.lock", "a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cpu")
    model.to(device).eval()
    model.device = device
stats = json.loads((TRAIN / "groups/atomic_seen/data/normalization.json").read_text())
contexts = {}
loaded_group = None
checkpoint = None


@torch.inference_mode()
def load_group(group):
    global loaded_group, checkpoint
    assert group in GROUPS
    if loaded_group != group:
        p = Path(plan["checkpoints"][group]["path"])
        stat = p.stat()
        assert (
            stat.st_size == plan["checkpoints"][group]["bytes"]
            and stat.st_mtime_ns == plan["checkpoints"][group]["mtime_ns"]
        )
        payload = torch.load(p, map_location="cpu", mmap=True, weights_only=False)
        assert payload["step"] == 50000
        model.mot.load_state_dict(payload["mot"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        loaded_group = group
        checkpoint = str(p)
        write(
            ROOT / "control" / f"loaded.{worker}.json",
            dict(
                group=group,
                checkpoint=checkpoint,
                step=50000,
                strict_load=True,
                time=time.time(),
            ),
        )
        print("GROUP_LOADED", group, checkpoint, flush=True)
    return dict(step=50000, group=loaded_group, checkpoint=checkpoint, strict_load=True)


def build_observation(cameras, state):
    obs = {
        "video." + k: np.frombuffer(base64.b64decode(v), np.uint8).reshape(256, 256, 3)
        for k, v in zip(
            ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"],
            cameras,
        )
    }
    start = 0
    for k, n in STATE_FIELDS:
        obs["state." + k] = np.asarray(state[start : start + n])
        start += n
    return observation_inputs(obs, stats)


@torch.inference_mode()
def predict(data):
    assert data["group"] == loaded_group
    image, state = build_observation(data["cameras"], data["state"])
    prompt = data["instruction"]
    if prompt not in contexts:
        p = text_path(TRAIN / "text_cache", prompt)
        contexts[prompt] = (
            (text_context(str(p)), torch.ones(512, dtype=torch.bool))
            if p.exists()
            else model.encode_prompt(prompt)
        )
    result = model.infer_action(
        prompt=None,
        context=contexts[prompt][0],
        context_mask=contexts[prompt][1],
        input_image=image,
        proprio=state,
        action_horizon=32,
        num_inference_steps=10,
        active_action_indices=ACTION_INDEX.tolist(),
        seed=int(data["noise_seed"]),
        rand_device="cpu",
        tiled=False,
    )
    decoded = decode_action(result["action"].float().cpu().numpy(), stats)
    native = np.concatenate([decoded["action." + k] for k, n in ACTION_FIELDS], axis=-1)
    assert native.shape == (32, 12) and np.isfinite(native).all()
    return dict(
        action=native.tolist(),
        checkpoint_step=50000,
        checkpoint=checkpoint,
        group=loaded_group,
    )


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ready":true}')

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/load":
                assert data["step"] == 50000
                result = load_group(data["group"])
            else:
                result = predict(data)
            body = json.dumps(result).encode()
            self.send_response(200)
        except Exception:
            body = json.dumps(dict(error=traceback.format_exc())).encode()
            print(body.decode(), flush=True)
            self.send_response(500)
        self.end_headers()
        self.wfile.write(body)


print("POLICY_READY", worker, port, flush=True)
HTTPServer(("127.0.0.1", port), Handler).serve_forever()
