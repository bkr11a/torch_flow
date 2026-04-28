# Scripts Quick Start

This document provides copy/paste-ready commands for common training, evaluation, and visualization workflows.

---

## Training

### Start training from scratch

```bash
python train.py --config configs/default.yaml
```

Trains from the base FlyingChairs configuration with default hyperparameters.

### Resume full training state

```bash
python train.py --config configs/default.yaml \
  training.checkpoint=checkpoints/hqs_flow_default/last.pth \
  training.resume_mode=full
```

Restores model, optimizer, scheduler, gradient scaler, and global step counter exactly.

### Curriculum warm-start on Sintel

```bash
python train.py --config configs/sintel_ft.yaml
```

`sintel_ft.yaml` is pre-configured with `resume_mode: weights_only`. Model weights are loaded from the configured checkpoint; optimizer, scheduler, and step counter all reset for a fresh OneCycle schedule.

### Sintel-only training

```bash
python train.py --config configs/sintel_only.yaml
```

No mixed FlyingThings dataset. Designed for final fine-tuning on Sintel alone.

### Override config values from CLI

```bash
# Change number of stages
python train.py --config configs/default.yaml model.model_backbone.num_hqs_iterations=8 run_name=ablate_stages_8

# Add OFCE loss
python train.py --config configs/default.yaml loss.ofce_weight=0.01

# Disable MLflow for local debugging
python train.py --config configs/default.yaml mlflow.enabled=false
```

OmegaConf dot-notation overrides take effect after the base config is loaded.

---

## Evaluation

### Standard evaluation

```bash
python evaluate.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth
```

Prints EPE, F1, speed-stratified, and distance-to-occlusion metrics.

### Comprehensive evaluation

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
├── flows_hsv/                # HSV visualizations (PNG)
├── gt_flows/                 # Ground truth flows (.flo)
├── gt_flows_hsv/             # GT flow visualizations (PNG)
├── errors/                   # EPE error maps (PNG)
├── intermediate_stages/      # Stage convergence grids
├── stage_convergence/        # EPE convergence plots
├── metrics_detailed.json     # Per-sample metrics
├── metrics_summary.json      # Aggregated metrics
└── flow_colorwheel_reference.png
```

### Check summary metrics

```bash
cat results/eval_sintel/metrics_summary.json
```

Example output:

```json
{
  "epe": 1.234,
  "f1": 5.3,
  "s0_10": 0.892,
  "s10_40": 1.456,
  "s40_plus": 3.789,
  "d0": 1.123,
  "d0_10": 0.987,
  "d10_60": 1.456,
  "d60_140": 1.890,
  "d140_plus": 2.345
}
```

Distance-to-occlusion keys (`d*`) are present only when the dataset provides occlusion masks (MPI-Sintel).

---

## Stage Progression Visualization

```bash
python visualize_stages.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/stages_sintel \
  --num_samples 10
```

Produces per-sample grids showing flow, error, and magnitude at each HQS stage, plus EPE convergence curves.

Interpretation:

- **Smooth EPE decrease**: healthy convergence
- **EPE plateaus early**: model capacity or feature quality issue
- **EPE increases at later stages**: potential numerical instability; check `hqs_beta`/`hqs_lambda` values

---

## HSV Flow Visualization

```python
from utils import flow_to_hsv, create_flow_colorwheel

flow_hsv = flow_to_hsv(flow)                    # (H, W, 2) → (H, W, 3) uint8 RGB
flow_hsv = flow_to_hsv(flow, max_magnitude=20.0) # explicit magnitude scale

wheel = create_flow_colorwheel()                 # reference colorwheel
```

For batch processing:

```python
from utils import flow_to_hsv_batch

rgb_list = flow_to_hsv_batch(flow_batch)         # List of (H, W, 3) arrays
```

---

## TensorBoard

```bash
tensorboard --logdir logs/
```

All loss components (`loss`, `epe`, `f1`, `smooth`, `photo`, `ofce`, speed-stratified, and distance-to-occlusion metrics) are logged per step.

---

## MLflow

If using a remote tracking server:

```yaml
# configs/default.yaml
mlflow:
  enabled: true
  tracking_uri: https://mlflow.yourdomain.home/
  insecure_tls: false    # require valid TLS cert
```

To run without MLflow:

```bash
python train.py --config configs/default.yaml mlflow.enabled=false
```

---

## Verification

```bash
python verify_integration.py    # full integration health check
python verify_model_build.py    # model construction only
python verify_data_loading.py   # dataset loading only
python verify_training.py       # single training step
```

Run after environment setup to confirm everything is wired correctly before a long training run.

---

Generated: 2026-04-23  
Version: 1.0
