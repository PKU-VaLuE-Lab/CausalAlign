# Fast-WAM on RoboCasa365

The video expert is Wan2.2-TI2V-5B with the CausalAlign LoRA applied as a frozen
adapter. The action expert is derived from the native Wan2.2 DiT with Fast-WAM's
parameter interpolation. The policy receives the current images of three cameras
and the current proprioceptive state, predicts 32 actions with 10 denoising steps
and executes the first 8 before replanning at 20 Hz.

This directory is an extension experiment built on top of a trained CausalAlign
video model; it is not part of the three-stage CausalAlign training pipeline. It
contains the current-frame WAM training and RoboCasa365 evaluation path only. No
architecture or objective ablation code is included.

## Interfaces

- The simulator uses 12D actions and 16D states (`evaluation/adapter.py`).
- `recipe_data.compact_rows` converts them to a 15D action (end-effector position,
  rotation as 6D, gripper, 5D base command) and a 19D state (end-effector position,
  6D rotation, gripper, base position, 6D base rotation). These are written into
  80D model vectors at `ACTION_INDEX` and `STATE_INDEX`; the action loss and the
  sampler ignore the other dimensions.
- Min-max normalization is computed on Human300 with the 6D rotation entries fixed
  to [-1, 1]. Target100 reuses the Human300 statistics.
- Images are combined into one 320×384 frame: left agent view on top (320×256),
  wrist camera bottom-left and right agent view bottom-right (160×128 each).

## Environment

Install the WAM dependencies from the repository root:

```sh
python -m pip install -e '.[wam]'
```

```sh
export PYTHONPATH="$PWD/wam:$PWD/wam/src:$PWD/wam/evaluation${PYTHONPATH:+:$PYTHONPATH}"
export WAM_RUN_ROOT=/path/to/wam_run             # prepared data, configs, checkpoints
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/models # contains Wan-AI/Wan2.2-TI2V-5B
export DIFFSYNTH_SKIP_DOWNLOAD=true
export WAM_VIDEO_LORA=/path/to/video_lora        # directory with model.safetensors and checkpoint.json
export WAM_ACTION_BACKBONE=/path/to/action_backbone.pt
```

`WAM_VIDEO_LORA` must target the Wan2.2-TI2V-5B topology. The directory contains
`model.safetensors` with module-qualified `lora_A`/`lora_B` tensors and a
`checkpoint.json` with `lora.rank`, `lora.alpha`, and an optional
`weights_sha256`. The default A14B checkpoint produced by the core release is not
interchangeable with this 5B adapter. Model weights are external assets and are
not committed to the repository.

Create the action backbone once:

```sh
python wam/scripts/preprocess_action_dit_backbone.py --model-config wam/configs/model.yaml \
  --output "$WAM_ACTION_BACKBONE" --dtype bfloat16
```

## Data

```sh
python wam/prepare_data.py --root "$WAM_RUN_ROOT" --stage human300 --datasets human300.json
python wam/prepare_data.py --root "$WAM_RUN_ROOT" --stage target100 --datasets target100.json
python wam/prepare_text.py --root "$WAM_RUN_ROOT" --model-dir "$DIFFSYNTH_MODEL_BASE_PATH/Wan-AI/Wan2.2-TI2V-5B"
```

`human300.json` lists the roots of the 300 RoboCasa365 human pretraining datasets
(LeRobot v2). `target100.json` lists `{"task", "group", "root"}` for the 50 target
tasks, with groups `atomic_seen` (18 tasks), `composite_seen` (16) and
`composite_unseen` (16). From each dataset, 100 (Human300) or 500 (Target100)
episodes are selected by shuffling the episode indices with `random.Random(0)`,
the rule used by RoboCasa's `get_subset_demos_filter_key`. Videos are read in
place. Prompt embeddings (UMT5, 512 tokens) are cached under `text_cache/`.

## Training

```sh
python wam/configure.py --root "$WAM_RUN_ROOT" --stage human300
# 16 ranks in total (for example two nodes of 8 GPUs) with torchrun:
torchrun --nnodes=2 --nproc_per_node=8 <rendezvous options> wam/train_human300.py

python wam/configure.py --root "$WAM_RUN_ROOT" --stage target100 \
  --initial-checkpoint "$WAM_RUN_ROOT/runs/pretrain/checkpoints/weights/step_060000.pt"
TARGET_GROUP=atomic_seen torchrun --standalone --nproc_per_node=8 wam/train_target100.py
```

| | Human300 pretraining | Target100 fine-tuning |
| --- | --- | --- |
| Data | 300 tasks × 100 episodes | 500 episodes per task, one model per group |
| Batch | 16 ranks × 16 = 256 | 8 ranks × 16 = 128 |
| Updates | 60k | 60k (evaluation uses 50k) |
| Learning rate | 1e-4, 3k warmup, cosine to 1e-6 | 3e-5, 3k warmup, cosine to 3e-6 |
| Sampling | all windows, shuffled per epoch | task-uniform, then window-uniform |

Both stages use AdamW (betas 0.9/0.95, weight decay 0.01), gradient clipping at
1.0, BF16 forward/backward with FP32 master weights and gradient reduction, and
TF32. Target100 loads the Human300 weights strictly and starts a new optimizer.
Weights are saved every 2k (Human300) or 5k (Target100) updates. Set
`WAM_RESUME_STATE` or `TARGET_RESUME_STATE` to a `checkpoints/state/step_*` directory
to resume. Runs log to W&B; set `WANDB_MODE=disabled` to turn this off.

## Evaluation

The simulator runs in a separate environment with RoboCasa365, robosuite and
MuJoCo 3.3.1 (`evaluation/requirements.txt`), and renders with Mesa llvmpipe on CPU.

```sh
export ROBOCASA_ENV=/path/to/simulator_env   # Python prefix of the simulator environment
export WAM_EVAL_ROOT=/path/to/eval_output
python wam/evaluation/prepare.py
CUDA_VISIBLE_DEVICES=0 python wam/evaluation/policy_server.py worker0 8700 0
# In a second shell, with the same variables, in the simulator environment:
for case in "$WAM_EVAL_ROOT"/cases/*.json; do
  "$ROBOCASA_ENV/bin/python" wam/evaluation/episode.py --case "$case" --port 8700 --worker worker0
done
```

`prepare.py` writes one case for each of the 50 target tasks and seeds 10000–10049,
and records the 50k checkpoint of each group. Each episode runs on the target split
with the task's original horizon and success check. The action-noise seed of a
replanning step is `seed·100000 + task_index·1000 + step//8`, so the task order
matters; pass `--tasks-json` to `prepare.py` to fix it. The first run of a case
stores the initial scene and image hashes in `paired_cases/`; reruns must reproduce
them. Each episode writes `results/<task>/<seed>.json` with a `success` field; the
success rate of a group is the fraction of successful episodes over its tasks.
