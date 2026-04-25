# HQSFlow Integration README

This document explains what was integrated into the main training framework, why those integrations matter, and how to run the most common tasks.

## What Is Integrated

The codebase includes a unified path for:

- model construction through config
- loss composition (sequence, smoothness, photometric, OFCE)
- dataset + augmentation + mask handling
- checkpoint resume and warm-start behavior
- standard and comprehensive evaluation scripts

Key behavior now present:

- occlusion and invalid masks follow the same crop/flip/resize as flow and valid masks
- Sintel distance metrics are surfaced in training logs
- curriculum warm-start can reset step and scheduler with `training.resume_mode=weights_only`

## HQS Model Math (Implementation View)

The model approximates alternating minimization of:

$$
E(u, v) = D(u) + \frac{\mu}{2}\|u-v\|_2^2 + \lambda R(v)
$$

Per unrolled stage $k$:

$$
u^{k+1} \approx \mathcal{D}_{\theta_k}(u^k, v^k, I_1, I_2, \text{corr})
$$

$$
v^{k+1} \approx \mathcal{R}_{\phi_k}(u^{k+1}, I_1)
$$

Where:

- $\mathcal{D}_{\theta_k}$ is the learned data/update block
- $\mathcal{R}_{\phi_k}$ is the learned proximal/regularization block
- outputs across stages are supervised with weighted sequence loss

## Command Cookbook

### Validate the repository wiring

```bash
python verify_integration.py
```

Purpose: confirms imports, configs, and main integration points are healthy.

### Base training

```bash
python train.py --config configs/default.yaml
```

Purpose: run the default recipe with secure MLflow defaults.

### Warm-start curriculum stage (reset LR schedule)

```bash
python train.py --config configs/sintel_ft.yaml
```

Purpose: load model weights from checkpoint while resetting step/scheduler due to `resume_mode=weights_only`.

### Force full resume behavior

```bash
python train.py --config configs/default.yaml \
  training.checkpoint=checkpoints/hqs_flow_default/last.pth \
  training.resume_mode=full
```

Purpose: continue exact optimizer/scheduler/global-step state.

### Train Sintel-only setup

```bash
python train.py --config configs/sintel_only.yaml
```

Purpose: run pure Sintel training without mixed dataset composition.

### Evaluate checkpoint

```bash
python evaluate.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth
```

Purpose: run standard evaluation metrics.

### Produce comprehensive outputs

```bash
python evaluate_comprehensive.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/eval_sintel
```

Purpose: generate richer reports and artifacts.

### Visualize stage progression

```bash
python visualize_stages.py \
  --config configs/sintel_ft.yaml \
  --checkpoint checkpoints/hqs_flow_sintel/best.pth \
  --data_config configs/sintel_ft.yaml \
  --output_dir results/stages_sintel \
  --num_samples 10
```

Purpose: inspect how flow evolves through the HQS unrolled stages.

### Enable OFCE in training

```bash
python train.py --config configs/default.yaml loss.ofce_weight=0.01
```

Purpose: add physics-inspired constraint loss.

## Logging and Metrics Notes

What you should see during training and validation:

- core: `loss`, `epe_matched`, `epe_unmatched`, `epe_all`, `f1`
- speed buckets: `s0_10`, `s10_40`, `s40_plus`
- Sintel distance buckets: `d0`, `d0_10`, `d10_60`, `d60_140`, `d140_plus`

If the dataset does not provide occlusion masks, distance buckets may be absent/NaN and the trainer emits a warning.

## MLflow Notes

Recommended production settings:

- `mlflow.insecure_tls: false`
- valid CA trust chain installed on training hosts
- writable artifact destination configured server-side

Local/offline fallback:

```bash
python train.py --config configs/default.yaml mlflow.enabled=false
```

## Integration Checklist

1. Run `verify_integration.py`.
2. Start a short training run and confirm distance metrics/logging presence.
3. Warm-start from a checkpoint and confirm step resets when expected.
4. Run evaluation and stage visualization scripts.
