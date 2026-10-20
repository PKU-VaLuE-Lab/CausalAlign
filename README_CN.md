# CausalAlign：三阶段训练代码

本目录提供 CausalAlign 的完整、固定且可复现的训练实现：从 DL3DV/RealEstate10K 原始数据建立 scene split，提取 LingBot-Map teacher state，训练 128D structured delta-token，训练 Wan hidden-state projector，最后在 Wan2.2-I2V-A14B 的第二个 Transformer 上训练 selected-layer LoRA。

代码使用论文方法约束以及历史成功实验中恢复出的训练设置。所有网络规模、优化器、学习率、batch、训练长度、LoRA 范围和 timestep 权重均已固定在 YAML 中；checkpoint 会保存完整配置和方法指纹。

## 1. 固定训练协议

| 项目 | Release recipe |
|---|---|
| 数据 | DL3DV + RealEstate10K，分别按 scene 固定切分 90% train / 10% held-out |
| Stage-1/2/3 数据量 | train split 的固定 30% / 10% / 100%，子集确定且嵌套 |
| delta token | 128D：ego 64D / geometry 40D / visual 24D |
| Stage 1 网络 | 4096D LingBot transition state；512-wide、2-block residual MLP encoder/decoder |
| Stage 1 训练 | 40,000 updates；前 10,000 使用 4,000 scenes，后 30,000 使用 8,312 scenes |
| Stage 1 优化 | AdamW；per-GPU batch 4；两段 cosine，peak LR `2e-4` / `5e-6`，各 warmup 500 |
| Stage 1 loss | `L_T = L_tr + 0.1 L_temp + 0.2 L_pose^T` |
| Stage 2 网络 | 每个目标层独立的 2-layer、1024-wide `Q_l`；两个 timestep 共享同一个 `Q_l` |
| Stage 2 训练 | 50,000 updates；AdamW `2e-4`；warmup 1,000 + cosine；per-GPU batch 4 |
| Stage 2 loss | `sum_{l,s} omega_s ||Q_l(Gamma(h_t,h_t+1))-delta_T||²` |
| Stage 3 位置 | Wan2.2 第二个 Transformer 的 layers `{10,12,14,20}` |
| Stage 3 去噪进度 | `{398,799}` 是降序 1,000 步 scheduler 的索引，运行时由对应位置取得原生 timestep 与 sigma；398 比 799 更早、噪声更高 |
| Stage 3 LoRA | rank/alpha `16/16`；self/cross-attention 的 Q/K/V/O；dropout 0 |
| Stage 3 优化 | 12 updates；AdamW `7e-6`；warmup 2 + cosine；8 GPU、microbatch 1、accumulation 9，global batch 72 |
| Stage 3 loss | `L_pose + 0.2 L_delta` |
| timestep 权重 | `omega_398=0.5`，`omega_799=0.5` |

Stage 2 冻结 Wan 和 Stage 1。Stage 3 冻结 Wan base、LingBot、Stage 1 和 Stage 2，只更新选定层中的 LoRA。推理只需 Wan base 与合并后的 LoRA，不依赖 LingBot、projector 或训练 cache。

## 2. 数据字段与坐标约定

训练使用以下信息：

- RGB 视频：LingBot-Map 输入、Wan VAE 输入、I2V 第一帧条件，以及 visual transition target。
- camera-to-world pose：相对位姿监督，定义为 `inverse(T_t) @ T_(t+1)`。
- 相机内参：保存进 manifest，并用于几何处理和审计。
- LingBot-Map final camera token 与 final patch-token mean：拼接为 4096D teacher state。
- LingBot depth：生成 16×28 log-depth 与 normal transition target。
- RGB：生成 32×56 visual transition target。
- caption：优先读取场景 caption；无 caption 时固定使用 `A realistic video of the scene with natural camera motion.`。

DL3DV 的 OpenGL c2w 右乘 `diag(1,-1,-1,1)` 转为 OpenCV camera axes。Wan temporal VAE 将 RGB 时间轴降采样，因此 teacher pair 使用固定 `feature_stride=4`，并通过 `pair_index` 与 Wan transition 对齐。

## 3. 环境和权重

推荐 Python 3.10/3.11、PyTorch 2.2+、CUDA 12.x：

```bash
cd causalalign_paper_release
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[wan,test]'
```

训练需要：

1. `Wan-AI/Wan2.2-I2V-A14B-Diffusers` 权重，包含 `transformer`、`transformer_2`、VAE、scheduler 和 text encoder。
2. LingBot-Map 源码与启用 camera/depth head 的 checkpoint。
3. DL3DV 原始 RGB、`transforms.json`、相机内外参。
4. RealEstate10K 官方 camera txt 与按 timestamp 解码的 RGB 帧。

## 4. 数据准备

在 `configs/data.yaml` 设置两个数据根目录和输出根目录，然后执行：

```bash
python tools/prepare_data.py --config configs/data.yaml
```

输出包括 `train_scenes.jsonl`、`eval_scenes.jsonl`、`stage1_scenes.jsonl`、`stage2_scenes.jsonl`、`stage3_scenes.jsonl`。排序使用 `seed + scene_id` 的 SHA-256，不依赖文件系统遍历顺序。

### 4.1 提取 LingBot teacher feature

```bash
python tools/extract_teacher_features.py \
  --scene-manifest /data/causalalign/manifests/stage1_scenes.jsonl \
  --output-root /data/causalalign/teacher_features/stage1 \
  --lingbot-repo /models/lingbot-map \
  --lingbot-checkpoint /models/lingbot-map/checkpoint.pt \
  --frame-stride 4 --device cuda

python tools/extract_teacher_features.py \
  --scene-manifest /data/causalalign/manifests/eval_scenes.jsonl \
  --output-root /data/causalalign/teacher_features/eval \
  --lingbot-repo /models/lingbot-map \
  --lingbot-checkpoint /models/lingbot-map/checkpoint.pt \
  --frame-stride 4 --device cuda
```

将生成的两个 `index.jsonl` 分别设为 Stage 1 的 train/val index。

### 4.2 生成 Stage 2/3 的 frozen teacher token

对 `stage2_scenes.jsonl`、`stage3_scenes.jsonl` 和 `eval_scenes.jsonl` 执行同样的 LingBot feature extraction，再运行：

```bash
python tools/cache_teacher_tokens.py \
  --teacher-checkpoint outputs/stage1/best.pt \
  --input-index /data/causalalign/teacher_features/stage2/index.jsonl \
  --output-root /data/causalalign/teacher_tokens/stage2 --device cuda
```

Stage 3 与 held-out 数据使用各自目录执行同一命令，teacher checkpoint 始终为 Stage 1 的 `best.pt`。

### 4.3 缓存 Wan 输入和 Stage 2 transition

```bash
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
```

Stage 2 的 Wan 已冻结，可以缓存 transition。Stage 3 只缓存 VAE/prompt/I2V condition，训练时实时执行 Wan，使梯度从 frozen projector 穿回 LoRA。

## 5. 三阶段训练

在三个 stage YAML 中设置 cache、checkpoint 和模型的机器路径。固定训练参数无需修改。

```bash
torchrun --standalone --nproc_per_node=8 -m causalalign.training.stage1 \
  --config configs/stage1.yaml

torchrun --standalone --nproc_per_node=8 -m causalalign.training.stage2 \
  --config configs/stage2.yaml

torchrun --standalone --nproc_per_node=8 -m causalalign.training.stage3 \
  --config configs/stage3.yaml
```

训练前执行完整静态和数据审计：

```bash
python tools/validate_release.py \
  --stage1-config configs/stage1.yaml \
  --stage2-config configs/stage2.yaml \
  --stage3-config configs/stage3.yaml \
  --check-data --check-wan
pytest -q
```

Stage 1/2 保存 model、optimizer、scheduler、step、完整配置和方法指纹。Stage 3 只保存 LoRA A/B、optimizer、scheduler、注入模块清单和完整配置，不重复保存 14B base。

## 6. 导出推理权重

```bash
python tools/export_lora.py \
  --checkpoint outputs/stage3/best.pt \
  --wan-model /models/Wan-AI/Wan2.2-I2V-A14B-Diffusers \
  --output outputs/stage3/merged
```

导出工具把 LoRA 合并到 `transformer_2`，其他 Wan 组件继续从原 base 目录加载。

## 7. 代码结构

```text
causalalign/
  data/                 manifest、确定性 split、record loader
  integrations/         LingBot-Map 与 Wan2.2 adapter
  models/teacher.py     Stage 1 structured 128D delta-token
  models/projector.py   Stage 2 layer-specific Q_l 与固定 Gamma
  models/lora.py        Stage 3 selected-layer LoRA
  training/stage1.py
  training/stage2.py
  training/stage3.py
tools/
  prepare_data.py
  extract_teacher_features.py
  cache_teacher_tokens.py
  cache_wan_inputs.py
  cache_wan_transitions.py
  validate_release.py
  export_lora.py
```

Stage 间通过 `paper_spec_sha256` 和 stage 编号严格校验，不能误接旧版 1k probe、GF probe 或其他 Stage7 checkpoint。

## 8. WAM 拓展实验

`wam/` 将冻结的 CausalAlign Wan2.2-TI2V-5B video expert 接入 Fast-WAM，包含
RoboCasa365 数据准备、Human300 预训练、Target100 微调以及闭环评测。这是三阶段
CausalAlign 训练完成后的拓展实验，不是第四个训练阶段；仓库未收录 ablation 代码。

该实现使用三路相机当前帧与当前 proprioceptive state。WAM 所需 LoRA 必须针对
Wan2.2-TI2V-5B 训练，不能直接使用本仓库默认 A14B recipe 的 Stage 3 权重。安装与
checkpoint 格式、训练配置和评测命令见 [`wam/README.md`](wam/README.md)：

```bash
pip install -e '.[wam,test]'
```

WAM 派生代码的上游来源与许可证见
[`third_party/NOTICES.md`](third_party/NOTICES.md)。
