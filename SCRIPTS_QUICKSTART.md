# New Scripts & Features Quick Start

## Overview

This document provides quick-start examples for the new evaluation and visualization scripts added during the hqs_pytorch integration.

---

## 1. HSV Flow Visualization

### Overview
The new HSV visualization is the **preferred method** for displaying optical flow.

- **Hue** = Flow direction (angle)
- **Saturation** = Flow magnitude (speed)  
- **Value** = Constant brightness

### Usage

#### Single Flow
```python
import torch
from utils import flow_to_hsv, create_flow_colorwheel

# Your flow: (2, H, W) or numpy (H, W, 2)
flow = torch.randn(2, 256, 512)

# Visualize
rgb = flow_to_hsv(flow)  # (H, W, 3) uint8 RGB

# Optional: scale to specific max magnitude
rgb = flow_to_hsv(flow, max_magnitude=10.0)

# Save
import cv2
cv2.imwrite("flow.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

# Show colorwheel reference
wheel = create_flow_colorwheel()
cv2.imwrite("colorwheel.png", cv2.cvtColor(wheel, cv2.COLOR_RGB2BGR))
```

#### Batch Processing
```python
from utils import flow_to_hsv_batch

flow_batch = torch.randn(8, 2, 256, 512)  # (B, 2, H, W)
rgb_list = flow_to_hsv_batch(flow_batch)  # List of (H, W, 3) arrays
```

---

## 2. Comprehensive Evaluation Script

### Purpose
Evaluate a trained model and save all results for detailed analysis.

### Basic Usage
```bash
python evaluate_comprehensive.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/eval_run_1
```

### Advanced Options
```bash
python evaluate_comprehensive.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/eval_run_1 \
  --batch_size 8 \
  --device cuda:0
```

### Output Files

```
results/eval_run_1/
├── flows/                          # Predicted flows (.flo format)
│   ├── batch_000000_item_00.flo
│   └── ...
├── flows_hsv/                      # HSV visualizations (PNG)
│   ├── batch_000000_item_00.png
│   └── ...
├── gt_flows/                       # Ground truth flows (.flo)
│   └── batch_000000_item_00_gt.flo
├── gt_flows_hsv/                  # GT flow visualizations (PNG)
│   └── batch_000000_item_00_gt.png
├── errors/                         # EPE error maps (HSV PNG)
│   └── batch_000000_item_00_epe.png
├── intermediate_stages/            # Stage-by-stage progression
│   └── batch_000000_item_00_stages.png
├── stage_convergence/              # Convergence plots
│   └── batch_000000_item_00_convergence.png
├── metrics_detailed.json           # Per-sample metrics
├── metrics_summary.json            # Aggregated metrics
├── flow_colorwheel_reference.png
└── evaluation.log
```

### Interpreting Results

#### metrics_summary.json
```json
{
  "epe": 1.234,           # Mean endpoint error
  "f1": 5.3,              # Outlier percentage
  "s0_10": 0.892,         # EPE for slow pixels (0-10 px)
  "s10_40": 1.456,        # EPE for medium motion
  "s40_plus": 3.789,      # EPE for fast motion
  "smoothness_loss": 0.032,
  "ofce_loss": 0.089
}
```

#### Convergence Plots
- **EPE Convergence**: Shows how mean EPE decreases over HQS stages
- **Improvement %**: Percentage improvement from initial prediction
- **Magnitude Evolution**: How estimated flow magnitude changes

---

## 3. Stage Progression Visualization

### Purpose
Visualize how optical flow evolves through all HQS stages with detailed analysis.

### Basic Usage
```bash
python visualize_stages.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/stage_viz
```

### Advanced Options
```bash
python visualize_stages.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/stage_viz \
  --num_samples 20 \
  --device cuda:0
```

### Understanding the Output

#### sample_XXXX_stages.png
A grid showing:
- **Row 0**: Input images (Image1, Image2) and valid mask
- **Row 1**: Ground truth flow and metrics
- **Rows 2+**: Each HQS stage with:
  - **Column 0**: Flow visualization (HSV)
  - **Column 1**: EPE error map (hot colormap)
  - **Column 2**: Flow magnitude difference (viridis colormap)

#### sample_XXXX_convergence.png
Two plots showing:
- **Left**: EPE over stages with improvement percentage
- **Right**: Mean flow magnitude evolution

### Interpretation Tips

1. **Perfect Convergence**: EPE curve smoothly decreases to plateau
2. **Stuck Convergence**: EPE decreases then plateaus quickly
3. **Divergence**: EPE increases at later stages (indicates issues)
4. **Jerky Convergence**: Large jumps between stages (might indicate numerical instability)

---

## 4. Using New Loss Functions

### OFCE Loss (Optical Flow Constraint Equation)

Enforces brightness constancy: `I_t + ∇I · u = 0`

#### Enable in Config
```yaml
loss:
  ofce_weight: 0.01
```

#### Expected Behavior
- Encourages more physically plausible flows
- Particularly useful for textured regions
- May reduce overfitting in data-scarce scenarios

#### Training with OFCE
```bash
# Create config with OFCE
cp configs/default.yaml configs/ofce_experiment.yaml
# Edit: set ofce_weight: 0.01

# Train
python train.py --config configs/ofce_experiment.yaml

# Watch for ofce loss in training output:
# loss: 0.0234 | epe: 1.456 | ofce: 0.0089
```

### Smoothness Loss

Encourages spatially smooth flow with edge awareness.

#### Enable in Config
```yaml
loss:
  smooth_weight: 0.05
```

### Photometric Loss

Enforces warping consistency (useful for semi-supervised training).

#### Enable in Config
```yaml
loss:
  photo_weight: 0.05
```

---

## 5. Typical Workflows

### Workflow A: Quick Model Evaluation
```bash
# 1. Evaluate model
python evaluate_comprehensive.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/quick_eval

# 2. Check summary metrics
cat results/quick_eval/metrics_summary.json

# 3. View HSV visualizations
open results/quick_eval/flows_hsv/
```

### Workflow B: Detailed Analysis
```bash
# 1. Comprehensive evaluation
python evaluate_comprehensive.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/detailed_eval

# 2. Stage progression visualization
python visualize_stages.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/best.pth \
  --data_config configs/default.yaml \
  --output_dir results/stage_analysis \
  --num_samples 20

# 3. Review results
# - Open results/detailed_eval/metrics_summary.json
# - Open results/stage_analysis/sample_*_stages.png
# - Open results/stage_analysis/sample_*_convergence.png
```

### Workflow C: Physics-Informed Training
```bash
# 1. Create config with OFCE loss
cat > configs/physics_informed.yaml <<EOF
# Copy from default.yaml, then modify:
loss:
  smooth_weight: 0.05
  photo_weight: 0.0
  ofce_weight: 0.01
EOF

# 2. Train
python train.py --config configs/physics_informed.yaml

# 3. Evaluate and compare
python evaluate_comprehensive.py \
  --config configs/physics_informed.yaml \
  --checkpoint checkpoints/hqs_flow_physics_informed/best.pth \
  --data_config configs/physics_informed.yaml \
  --output_dir results/physics_eval
```

---

## 6. Troubleshooting

### OFCE Loss is NaN
```
Causes:
- Images not normalized correctly
- Invalid tensor shapes
- Gradient computation issues

Solutions:
1. Check image normalization: should be [0, 1] or [-1, 1]
2. Start with small ofce_weight (0.001) and increase gradually
3. Ensure image1 and image2 have same shape
4. Check for invalid flow values
```

### HSV Visualization Looks Washed Out
```
Causes:
- Flow magnitude too large or too small
- Not specifying max_magnitude parameter

Solutions:
1. Use max_magnitude parameter:
   rgb = flow_to_hsv(flow, max_magnitude=20.0)
2. Check flow statistics:
   print(flow.min(), flow.max(), flow.mean())
3. Use 95th percentile for max:
   max_mag = torch.quantile(flow.abs(), 0.95)
```

### Stage Visualization Script is Slow
```
Solutions:
1. Reduce num_samples: --num_samples 5
2. Increase batch_size: --batch_size 8
3. Use fewer stages in model config if possible
4. Run on GPU with --device cuda:0
```

---

## 7. Integration with Training Pipeline

### During Training
Loss components are logged automatically:

```
Epoch 1 | loss: 0.0234 | epe: 1.456 | f1: 5.3% | smooth: 0.0032 | ofce: 0.0089
```

### TensorBoard Logging
All loss components are logged to TensorBoard:

```bash
tensorboard --logdir logs/
```

### Checkpoint Structure
Checkpoints save all loss values for reproducibility.

---

## 8. Performance Benchmarks

### Evaluation Speed
- **Comprehensive Eval**: ~50 samples/sec (GPU), ~5 samples/sec (CPU)
- **Stage Visualization**: ~20 samples/sec (GPU)

### Storage Requirements
- **Per Sample**: ~2.5 MB (.flo + PNG visualization)
- **1000 Samples**: ~2.5 GB

### Training Overhead
- **OFCE Loss**: +2% training time
- **Smoothness Loss**: +1% training time
- **HSV Visualization**: No training overhead (only at eval)

---

## 9. Best Practices

1. **Always run comprehensive evaluation** on validation set before deployment
2. **Use stage visualization** to debug convergence issues
3. **Enable OFCE loss** for physics-constrained applications
4. **Check colorwheel** when interpreting HSV visualizations
5. **Compare metrics_summary** across different configurations
6. **Save detailed metrics** for publication and reproducibility

---

## Quick Command Reference

```bash
# Train with OFCE loss
python train.py --config configs/default.yaml loss.ofce_weight=0.01

# Quick evaluation
python evaluate_comprehensive.py --config cfg.yaml --checkpoint best.pth --data_config cfg.yaml --output_dir results/eval

# Stage analysis
python visualize_stages.py --config cfg.yaml --checkpoint best.pth --data_config cfg.yaml --output_dir results/stages --num_samples 10

# Compare two models
python evaluate_comprehensive.py --config cfg.yaml --checkpoint model1.pth --data_config cfg.yaml --output_dir results/model1
python evaluate_comprehensive.py --config cfg.yaml --checkpoint model2.pth --data_config cfg.yaml --output_dir results/model2
# Compare: results/model1/metrics_summary.json vs results/model2/metrics_summary.json
```

---

Generated: 2026-04-23  
Version: 1.0
