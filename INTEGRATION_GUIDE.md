# HQS Flow Integration Guide

## Overview

This document describes the integration of the `hqs_pytorch` folder with the main training framework. The merge unifies model development while keeping both codebases available for reference.

The authoritative implementation is `models/hqs_flow.py`. It incorporates all critical bug fixes from `hqs_pytorch` while maintaining full compatibility with the configuration system, training pipeline, and metrics collection.

---

## Mathematical Formulation

Given two images $I_1, I_2$, optical flow $u$, and regularizer variable $v$:

$$
E(u, v) = D(u; I_1, I_2) + \frac{\mu}{2}\|u - v\|_2^2 + \lambda R(v)
$$

Unrolled HQS iterations alternate between:

$$
u^{k+1} = \arg\min_u \; D(u; I_1, I_2) + \frac{\mu^k}{2}\|u - v^k\|_2^2
$$

$$
v^{k+1} = \arg\min_v \; \frac{\mu^k}{2}\|u^{k+1} - v\|_2^2 + \lambda R(v)
$$

Each stage is realized by learned modules: a data/update network for the $u$-subproblem and a proximal/regularization network for the $v$-subproblem. The model is trained with a stage-weighted sequence loss.

---

## 1. Architecture and Authoritative Files

`models/hqs_flow.py` is the single source of truth.

Weight sharing modes (controlled by `model.weight_sharing`):

- `share_all`: all stages share weights
- `share_none`: each stage has independent weights
- `share_update`: shared update/data network, independent prox
- `share_prox`: shared prox network, independent update

The `graveyard/` folder retains deprecated implementations as reference material only.

---

## 2. Loss Functions

### SequenceLoss (default)

Supervised stage-weighted loss with geometric decay. The final stage receives the largest weight.

```yaml
loss:
  gamma: 0.85        # decay factor; lower = more weight on later stages
  max_flow: 400.0    # clamp GT flow magnitude before loss
  loss_fn: charbonnier
```

### SmoothnessLoss

Edge-aware spatial smoothness. Encourage gradients only in homogeneous regions.

```yaml
loss:
  smooth_weight: 0.05    # set > 0 to enable
```

### PhotometricLoss

Warping consistency for semi-supervised settings (no GT flow required).

```yaml
loss:
  photo_weight: 0.05    # set > 0 to enable
```

### OFCE Loss

Enforces the brightness constancy constraint $I_t + \nabla I \cdot u = 0$.

```yaml
loss:
  ofce_weight: 0.01    # set > 0 to enable
```

---

## 3. HSV Flow Visualization

**Module**: `utils/flow_visualization_hsv.py` — preferred over the legacy Middlebury visualizer.

```python
from utils import flow_to_hsv, flow_to_hsv_batch, create_flow_colorwheel

flow_hsv = flow_to_hsv(flow)          # (H, W, 2) → (H, W, 3) uint8 RGB
flows_hsv = flow_to_hsv_batch(batch)  # List of RGB images
wheel = create_flow_colorwheel()      # Reference colorwheel image
```

Hue encodes direction; saturation encodes magnitude; value is constant.

Both `flow_to_hsv` and the legacy `flow_to_color` remain importable.

---

## 4. Evaluation Scripts

### Comprehensive Evaluation (`evaluate_comprehensive.py`)

Produces flows, visualizations, error maps, and per-sample/aggregated metrics.

Saves predicted flows (`.flo`), HSV visualizations, error maps, per-sample metrics, and stage convergence plots.

```bash
python evaluate_comprehensive.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/eval_sintel
```

Output layout:

```text
results/eval_sintel/
├── flows/                    # Predicted flows (.flo)
├── flows_hsv/                # Predicted flows (HSV PNG)
├── gt_flows/                 # Ground truth flows (.flo)
├── gt_flows_hsv/             # GT flows (HSV PNG)
├── errors/                   # EPE error maps
├── intermediate_stages/      # Multi-stage convergence grids
├── stage_convergence/        # Convergence plots
├── metrics_detailed.json     # Per-sample metrics
├── metrics_summary.json      # Aggregated metrics
└── flow_colorwheel_reference.png
```

### Stage Progression Visualization (`visualize_stages.py`)

Shows flow evolution across all unrolled HQS stages. Useful for diagnosing slow convergence or divergence.

```bash
python visualize_stages.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/stages_sintel \
  --num_samples 10
```

Output per sample: `sample_XXXX_stages.png` (flow/error/magnitude grid) and `sample_XXXX_convergence.png` (EPE curve).

---

## 5. Trainer and Metrics

### Progress Display

During training:

```text
Epoch 1 | loss: 0.0234 | epe: 1.456 | f1: 5.3%
         s0_10: 0.892 | s10_40: 1.234 | s40+: 3.456
         d0: 1.123 | d0_10: 0.987 | d10_60: 1.456 | d60_140: 1.890 | d140+: 2.345
         smooth: 0.0032 | photo: 0.0045 | ofce: 0.0089 | lr: 4.00e-04
```

Distance-to-occlusion metrics (`d0` through `d140_plus`) are logged when the dataset provides occlusion masks. If masks are absent, a one-time warning is emitted and these keys are omitted.

### Resume Modes

Controlled by `training.resume_mode`:

- `full` (default): restores model, optimizer, scheduler, gradient scaler, and global step counter
- `weights_only`: loads model weights only, resets everything else — use for curriculum warm-starts

### Metrics Reference

| Key | Meaning |
| --- | ------- |
| `epe_all` | Mean endpoint error over all valid pixels |
| `epe_matched` | EPE on visible/matched pixels |
| `epe_unmatched` | EPE on occluded pixels |
| `f1` | KITTI-style outlier percentage |
| `s0_10`, `s10_40`, `s40_plus` | Speed-stratified EPE |
| `d0`, `d0_10`, `d10_60`, `d60_140`, `d140_plus` | Distance-to-occlusion-boundary EPE (Sintel) |

---

## 6. Configuration Reference

### Resume Mode

```yaml
training:
  checkpoint: checkpoints/hqs_flow_default/best.pth
  resume_mode: weights_only    # or: full
```

### Loss Terms

```yaml
loss:
  gamma: 0.85          # sequence loss geometric decay
  max_flow: 400.0      # GT flow clamp
  loss_fn: charbonnier
  smooth_weight: 0.05  # edge-aware smoothness (0 to disable)
  photo_weight: 0.0    # photometric consistency (0 to disable)
  ofce_weight: 0.01    # brightness constancy (0 to disable)
```

---

## 7. Command Cookbook

### Train from scratch

```bash
python train.py --config configs/default.yaml
```

### Resume full training state

```bash
python train.py --config configs/default.yaml \
  training.checkpoint=checkpoints/hqs_flow_default/last.pth \
  training.resume_mode=full
```

### Curriculum warm-start on Sintel

```bash
python train.py --config configs/sintel_ft.yaml
```

`sintel_ft.yaml` sets `resume_mode: weights_only` so the OneCycle schedule restarts from zero.

### Sintel-only training

```bash
python train.py --config configs/sintel_only.yaml
```

No mixed FlyingThings dataset; designed for final fine-tuning.

### Standard evaluation

```bash
python evaluate.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth
```

### Comprehensive evaluation

```bash
python evaluate_comprehensive.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/eval_sintel
```

### Stage progression analysis

```bash
python visualize_stages.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/stages_sintel \
  --num_samples 10
```

### Disable MLflow for local debugging

```bash
python train.py --config configs/default.yaml mlflow.enabled=false
```

---

## 8. Troubleshooting

**OFCE loss is NaN**: check image normalization (expected `[0, 1]` or `[-1, 1]`); start with `ofce_weight: 0.001`.

**Distance metrics missing**: dataset has no occlusion masks. A one-time warning is logged. Metrics are expected to be absent on non-Sintel datasets.

**Curriculum LR not resetting**: ensure `resume_mode: weights_only` is set in the fine-tuning config, not `full`.

**MLflow TLS warning at startup**: set `mlflow.insecure_tls: false` in `default.yaml`. The trainer clears the `MLFLOW_TRACKING_INSECURE_TLS` env var automatically when disabled.

---

## 9. File Reference

| File | Purpose |
| ---- | ------- |
| `models/hqs_flow.py` | Authoritative HQSFlow model |
| `engine/trainer.py` | Training loop, checkpoint I/O, MLflow |
| `data/augmentation.py` | Spatial/photometric augmentation |
| `losses/flow_loss.py` | SequenceLoss + auxiliary losses |
| `utils/flow_visualization_hsv.py` | HSV flow visualizer |
| `configs/default.yaml` | Base configuration |
| `configs/sintel_ft.yaml` | Sintel fine-tune (weights_only resume) |
| `configs/sintel_only.yaml` | Sintel-only training |
| `graveyard/` | Deprecated implementations (reference only) |
| `hqs_pytorch/AUDIT_REPORT.md` | Bug fix audit trail |
