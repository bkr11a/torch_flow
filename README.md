# HQSFlow (PyTorch)

HQSFlow is an optical flow training and evaluation framework built around an unrolled Half-Quadratic Splitting (HQS) solver.

This repository supports:

- multi-stage curriculum training
- dense and sparse optical flow datasets
- optional MLflow/TensorBoard logging
- robust metrics, including Sintel distance-to-occlusion metrics

## Mathematical Formulation

Given two images $I_1, I_2$, optical flow $u$, and regularizer variable $v$, we optimize:

$$
E(u, v) = D(u; I_1, I_2) + \frac{\mu}{2}\|u - v\|_2^2 + \lambda R(v)
$$

Where:

- $D(u; I_1, I_2)$ is the data term (photometric and/or learned matching consistency)
- $R(v)$ is the regularization prior
- $\mu$ is the HQS coupling parameter

Unrolled HQS iterations alternate:

$$
u^{k+1} = \arg\min_u \; D(u; I_1, I_2) + \frac{\mu^k}{2}\|u - v^k\|_2^2
$$

$$
v^{k+1} = \arg\min_v \; \frac{\mu^k}{2}\|u^{k+1} - v\|_2^2 + \lambda R(v)
$$

In practice, each stage is represented by learned modules:

- a data/update network approximating the $u$-subproblem
- a proximal/regularization network approximating the $v$-subproblem

The model predicts intermediate flows across stages and is trained with a stage-weighted sequence loss.

## Repository Layout

```text
torch_flow/
  configs/                  # Training/eval configs
  data/                     # Dataset loaders and augmentations
  engine/                   # Trainer and checkpoint logic
  losses/                   # Supervised and optional auxiliary losses
  models/                   # Main HQSFlow architecture
  utils/                    # Metrics, visualization, helpers
  evaluate.py               # Standard evaluation
  evaluate_comprehensive.py # Extended evaluation outputs
  train.py                  # Training entrypoint
  visualize_stages.py       # Stage progression visualization
```

## Configuration Notes

Training behavior is controlled through OmegaConf YAML files.

Important fields:

- `training.checkpoint`: checkpoint path to load
- `training.resume_mode`: `full` or `weights_only`
- `mlflow.insecure_tls`: should be `false` for verified TLS

Resume modes:

- `full`: resumes model, optimizer, scheduler, scaler, and global step
- `weights_only`: loads model weights only, resets optimizer/scheduler/step for curriculum warm-start

## Command Cookbook

The commands below are intentionally copy/paste-ready.

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

Use this once per environment.

### 2) Verify integration health

```bash
python verify_integration.py
```

Runs sanity checks for model/build/config/integration wiring.

### 3) Train from base config (Chairs default)

```bash
python train.py --config configs/default.yaml
```

Starts training with the default dataset and hyperparameters.

### 4) Curriculum stage with full resume

```bash
python train.py --config configs/default.yaml \
  training.checkpoint=checkpoints/hqs_flow_default/last.pth \
  training.resume_mode=full
```

Continues exact training state, including learning-rate schedule position.

### 5) Curriculum warm-start from best model (reset LR schedule)

```bash
python train.py --config configs/sintel_ft.yaml
```

`configs/sintel_ft.yaml` is configured for warm-start (`weights_only`) so steps and OneCycle schedule restart from zero.

### 6) Sintel-only training recipe

```bash
python train.py --config configs/sintel_only.yaml
```

Uses only Sintel data (no mixed Things dataset) with warm-start from checkpoint.

### 7) Disable MLflow quickly from CLI

```bash
python train.py --config configs/default.yaml mlflow.enabled=false
```

Useful when debugging local training without a tracking server.

### 8) Evaluate a checkpoint

```bash
python evaluate.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth
```

Runs standard evaluation metrics for the configured validation dataset.

### 9) Comprehensive evaluation artifacts

```bash
python evaluate_comprehensive.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/eval_sintel
```

Produces richer reports and saved outputs.

### 10) Visualize stage progression

```bash
python visualize_stages.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/stages_sintel \
  --num_samples 10
```

Helps inspect convergence behavior across unrolled HQS stages.

### 11) Run a quick ablation (number of stages)

```bash
python train.py --config configs/default.yaml model.model_backbone.num_hqs_iterations=8 run_name=ablate_stages_8
```

Overrides config values directly from CLI.

### 12) Toggle OFCE loss term

```bash
python train.py --config configs/default.yaml loss.ofce_weight=0.01
```

Adds physics-inspired regularization to training.

## Metrics and What They Mean

Common metrics include:

- `epe_matched`: endpoint error on matched/visible pixels
- `epe_unmatched`: endpoint error on unmatched/occluded pixels
- `epe_all`: combined endpoint error
- `f1`: KITTI-style outlier percentage
- `s0_10`, `s10_40`, `s40_plus`: speed-stratified EPE buckets
- `d0`, `d0_10`, `d10_60`, `d60_140`, `d140_plus`: Sintel distance-to-occlusion-boundary buckets

If occlusion masks are not available from the dataset, distance metrics can be absent or NaN.

## MLflow Setup Guidance

Recommended defaults:

- `mlflow.insecure_tls: false`
- valid CA chain installed on clients
- artifact storage path writable by MLflow server process

If you need local runs without server interaction, set `mlflow.enabled=false`.

## Typical Workflow

1. Install dependencies and verify integration.
2. Train base stage.
3. Warm-start next curriculum stage with `weights_only`.
4. Evaluate with `evaluate.py` or `evaluate_comprehensive.py`.
5. Inspect stage behavior with `visualize_stages.py`.

## Related Documentation

- `README_INTEGRATION.md`: integration details and migration notes
- `SCRIPTS_QUICKSTART.md`: broader script examples
- `INTEGRATION_GUIDE.md`: feature-level integration explanation
- `hqs_pytorch/README.md`: details about the TensorFlow-port module set
