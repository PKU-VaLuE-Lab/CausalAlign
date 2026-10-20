<h1 align="center">CausalAlign: Causal Alignment of Video Generation Models for Consistent Ego-Motion</h1>

<p align="center">Kai Zhu<sup>1,*</sup>, Yintianrun Zhang<sup>1,*</sup>, Shuwei Jiang<sup>1</sup>, Ruolin Zuo<sup>1</sup>, Yixin Chen<sup>2</sup>, Duomin Wang<sup>3</sup>, Miao Liu<sup>4</sup>, Yin Li<sup>5</sup>, Daquan Zhou<sup>1</sup>, Yiwu Zhong<sup>1,†</sup></p>

<p align="center"><sup>1</sup> Peking University · <sup>2</sup> Beijing Institute for General Artificial Intelligence · <sup>3</sup> NVIDIA · <sup>4</sup> Tsinghua University · <sup>5</sup> University of Wisconsin–Madison<br><small><sup>*</sup> Equal contribution · <sup>†</sup> Corresponding author</small></p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.02133"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B" alt="arXiv paper"></a>
  <a href="https://pku-value-lab.github.io/CausalAlign-homepage/"><img src="https://img.shields.io/badge/Project-Page-1F6FEB" alt="Project page"></a>
  <a href="https://huggingface.co/papers/2603.02133"><img src="https://img.shields.io/badge/Hugging%20Face-Daily%20Paper-FFD21E" alt="Hugging Face Daily Paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="Apache-2.0 license"></a>
</p>

<p align="center"><strong>Official training implementation for learning and causally aligning structured ego-motion states in Wan2.2.</strong></p>

## Demo

[![Watch the CausalAlign demo](demo/demo-poster.webp)](https://pku-value-lab.github.io/CausalAlign-homepage/#demo)

## Overview

CausalAlign uses a three-stage pipeline:

1. **Structured delta-token learning.** A frozen LingBot-Map teacher supplies consecutive 4096D scene states. An encoder, transition decoder, temporal predictor, and camera decoder learn a 128D delta-token split into 64 camera, 40 geometry, and 24 visual dimensions.
2. **VGM state alignment.** Layer-specific projectors map temporal transitions from frozen Wan2.2 hidden states into the structured delta-token space.
3. **Causal LoRA alignment.** The teacher and projectors remain frozen while LoRA adapters update the causally selected layers and denoising timesteps of Wan2.2.

This repository includes deterministic DL3DV/RealEstate10K manifests, LingBot feature extraction, Wan input and hidden-state caching, all three training stages, LoRA export, fail-fast validation, and tests.

Detailed Chinese documentation is available in [README_CN.md](README_CN.md).

## Method contract

| Component | Configuration |
|---|---|
| Backbone | Wan2.2-I2V-A14B Diffusers, low-noise transformer_2 |
| Datasets | DL3DV + RealEstate10K; independent 90/10 scene splits |
| Stage subsets | 30% / 10% / 100% of the training split |
| Delta-token | 128D = 64 camera + 40 geometry + 24 visual |
| Stage 1 | 40,000 updates; 4,000-scene warmup followed by 8,312 scenes |
| Stage 1 objective | L_transition + 0.1 L_temporal + 0.2 L_pose |
| Stage 2 targets | layers 10, 12, 14, 20 at denoising-progress indices 398 and 799 |
| Stage 3 LoRA | rank 16, alpha 16, self/cross-attention Q/K/V/O |
| Stage 3 objective | L_pose + 0.2 L_delta |
| Stage 3 optimizer | AdamW, LR 7e-6, global batch 72, 12 updates |

The paper's denoising progress is the index of the descending 1,000-step Wan scheduler. Progress 398 is earlier and higher-noise than progress 799; the native timestep and sigma are read from those scheduler entries.

## Installation

Python 3.10 or 3.11, PyTorch 2.4+, and CUDA 12.x are recommended.

~~~bash
git clone https://github.com/j-zhu23/CausalAlign.git
cd CausalAlign
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[wan,test]'
~~~

Install the additional world-action-model dependencies only when running the WAM
extension:

~~~bash
python -m pip install -e '.[wam,test]'
~~~

## Required assets

- [Wan-AI/Wan2.2-I2V-A14B-Diffusers](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)
- LingBot-Map source and a checkpoint with camera and depth heads enabled
- DL3DV RGB frames, camera intrinsics, and camera-to-world poses
- RealEstate10K camera files and decoded timestamp frames

Model and dataset files are not committed to this repository.

## Data preparation

Set the machine-local roots in configs/data.yaml and build deterministic manifests:

~~~bash
python tools/prepare_data.py --config configs/data.yaml
~~~

Extract LingBot teacher states and dense targets:

~~~bash
python tools/extract_teacher_features.py \
  --scene-manifest /data/causalalign/manifests/stage1_scenes.jsonl \
  --output-root /data/causalalign/teacher_features/stage1 \
  --lingbot-repo /models/lingbot-map \
  --lingbot-checkpoint /models/lingbot-map/checkpoint.pt \
  --frame-stride 4 --device cuda
~~~

Run the extraction for Stage 2, Stage 3, and held-out manifests. After Stage 1 has converged, cache frozen teacher targets:

~~~bash
python tools/cache_teacher_tokens.py \
  --teacher-checkpoint outputs/stage1/best.pt \
  --input-index /data/causalalign/teacher_features/stage2/index.jsonl \
  --output-root /data/causalalign/teacher_tokens/stage2 \
  --device cuda
~~~

Cache Wan inputs and detached Stage 2 transitions:

~~~bash
python tools/cache_wan_inputs.py \
  --scene-manifest /data/causalalign/manifests/stage2_scenes.jsonl \
  --teacher-token-index /data/causalalign/teacher_tokens/stage2/index.jsonl \
  --output-root /data/causalalign/wan_inputs/stage2 \
  --wan-model /models/Wan-AI/Wan2.2-I2V-A14B-Diffusers --device cuda

python tools/cache_wan_transitions.py \
  --wan-input-index /data/causalalign/wan_inputs/stage2/index.jsonl \
  --output-root /data/causalalign/wan_transitions/stage2 \
  --wan-model /models/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --expert transformer_2 --device cuda
~~~

Stage 3 caches VAE, prompt, and I2V-conditioning inputs but executes Wan live so gradients reach the LoRA adapters.

## Training

Set cache, checkpoint, and model paths in the stage configuration files. The optimization recipe is fixed by the release configs.

~~~bash
torchrun --standalone --nproc_per_node=8 \
  -m causalalign.training.stage1 --config configs/stage1.yaml

torchrun --standalone --nproc_per_node=8 \
  -m causalalign.training.stage2 --config configs/stage2.yaml

torchrun --standalone --nproc_per_node=8 \
  -m causalalign.training.stage3 --config configs/stage3.yaml
~~~

## Validation

Run the release, data, and Wan architecture checks before allocating a full job:

~~~bash
python tools/validate_release.py \
  --stage1-config configs/stage1.yaml \
  --stage2-config configs/stage2.yaml \
  --stage3-config configs/stage3.yaml \
  --check-data --check-wan
pytest -q
~~~

The validator checks the fixed recipe, data finiteness, 40-layer A14B architecture, 36-channel I2V input, 5120D hidden representation, and flow scheduler.

## WAM extension experiment

[`wam/`](wam/) transfers a frozen CausalAlign Wan2.2-TI2V-5B video expert into a
Fast-WAM policy and provides RoboCasa365 data preparation, Human300 pretraining,
Target100 fine-tuning, and closed-loop evaluation. It is an extension experiment,
not a fourth CausalAlign training stage, and no ablation code is included.

The WAM path uses a current-frame observation made from three cameras plus the
current proprioceptive state. Its frozen video adapter must target the 5B model;
the A14B Stage 3 checkpoint from the default recipe is not architecture-compatible.
See the [WAM reproduction guide](wam/README.md) for the exact adapter contract,
data interfaces, training schedule, and evaluation commands.

## Export

~~~bash
python tools/export_lora.py \
  --checkpoint outputs/stage3/best.pt \
  --wan-model /models/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --output outputs/stage3/merged
~~~

Stage 1/2 checkpoints contain model, optimizer, scheduler, step, complete configuration, and method fingerprint. Stage 3 checkpoints contain LoRA weights plus optimizer/scheduler state and an exact injected-module manifest.

## Repository layout

~~~text
causalalign/              Core package
  data/                   Manifests and record datasets
  integrations/           LingBot-Map and Wan2.2 adapters
  models/                 Structured teacher, projectors, and LoRA
  training/               Three stage trainers
configs/                  Fixed release recipes
tools/                    Preparation, caching, validation, and export
tests/                    Unit and contract tests
wam/                      Fast-WAM transfer and RoboCasa365 evaluation
third_party/              Notices and licenses for WAM-derived components
demo/                     Project demo assets
~~~

## Citation

~~~bibtex
@article{zhu2026causalalign,
  title   = {CausalAlign: Causal Alignment of Video Generation Models for Consistent Ego-Motion},
  author  = {Zhu, Kai and Zhang, Yintianrun and Jiang, Shuwei and Zuo, Ruolin and Chen, Yixin and Wang, Duomin and Liu, Miao and Li, Yin and Zhou, Daquan and Zhong, Yiwu},
  journal = {arXiv preprint arXiv:2603.02133},
  year    = {2026}
}
~~~

## License

This project is released under the [Apache License 2.0](LICENSE). WAM-derived
components retain their upstream terms; see [third_party/NOTICES.md](third_party/NOTICES.md).
