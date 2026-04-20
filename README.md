# Half-Quadratic Splitting Optical Flow (HQSFlow)

A PyTorch implementation of an **unrolled, learned Half-Quadratic Splitting (HQS)** network for optical flow estimation.

## Mathematical Formulation

The optical flow energy minimization:

$$E(u) = \sum_x \rho\bigl(I_1(x) - I_2(x + u(x))\bigr) + \lambda\,\phi(u)$$

HQS introduces auxiliary variable $v$ to decouple data and regularization:

$$E(u, v) = \underbrace{\sum_x \rho\bigl(I_1(x) - I_2(x+u)\bigr)}_{\text{data term}} + \frac{\mu}{2}\|u - v\|^2 + \underbrace{\lambda\,\phi(v)}_{\text{regularizer}}$$

Alternating minimization produces two subproblems per stage $k$:

1. **Data subproblem** — solved by a learned ConvGRU update network $\mathcal{D}_\theta$:
$$u^{k+1} = \mathcal{D}_\theta\!\left(f_1, f_2, \mathrm{CorrVol}(f_1, f_2, u^k),\; u^k,\; v^k,\; \mu^k\right)$$

2. **Proximal / regularisation subproblem** — solved by a learned proximal operator $\mathcal{R}_\theta$ (CNN denoiser):
$$v^{k+1} = \mathcal{R}_\theta\!\left(u^{k+1},\; \sqrt{\lambda/\mu^k}\right)$$

Parameters $\mu^k$ and $\lambda/\mu^k$ are **learnable per-stage scalars**, so the network adapts the penalty schedule through training.

## Architecture

```
ImagePair (H×W×3) ──►  FeaturePyramid  ──► {f1, f2}  (1/8 scale)
                    └►  ContextEncoder  ──► ctx        (1/8 scale)

{f1, f2, ctx, u0=0, v0=0}
      │
   ┌──┴──────────────────────────────────┐
   │  HQS Stage 1 … Stage K              │  (K configurable, default 12)
   │  ┌──────────────────────────────┐   │
   │  │  CorrBlock(f1, f2, u^k)      │   │
   │  │  DataUpdateNet (ConvGRU)    │   │
   │  │    → u^{k+1}, hidden state  │   │
   │  │  ProximalNet (CNN denoiser) │   │
   │  │    → v^{k+1}                │   │
   │  └──────────────────────────────┘   │
   └──────────────────────────────────────┘
         │
         ▼
   Flow predictions {u^1 … u^K}  ──► sequence loss
   Final upsampled flow (×8)
```

## Project Layout

```
torch_flow/
├── configs/
│   ├── default.yaml               # base hyperparams
│   ├── sintel_ft.yaml             # MPI-Sintel fine-tune
│   ├── spring_ft.yaml             # Spring fine-tune
│   ├── kitti_ft.yaml              # KITTI fine-tune
│   └── ablations/
│       ├── stages_04.yaml         # 4 HQS stages
│       ├── stages_08.yaml         # 8 HQS stages
│       ├── stages_16.yaml         # 16 HQS stages
│       ├── no_regularizer.yaml    # ablate proximal net
│       ├── local_corr.yaml        # local vs all-pairs corr
│       └── small_model.yaml       # reduced capacity
├── models/
│   ├── __init__.py
│   ├── hqs_flow.py                # top-level model
│   ├── encoders.py                # feature + context pyramid
│   ├── correlation.py             # all-pairs & local cost volumes
│   ├── update_net.py              # ConvGRU data subproblem solver
│   ├── reg_net.py                 # CNN proximal operator
│   └── warp.py                    # differentiable flow warping
├── data/
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── augmentation.py
│   ├── flyingchairs.py
│   ├── flyingthings.py
│   ├── sintel.py
│   ├── spring.py
│   └── kitti.py
├── losses/
│   ├── __init__.py
│   └── flow_loss.py               # sequence & photometric losses
├── utils/
│   ├── __init__.py
│   ├── flow_utils.py              # flow I/O, warping helpers
│   ├── metrics.py                 # EPE, Fl-all, etc.
│   └── visualization.py          # flow colorisation
├── engine/
│   ├── __init__.py
│   └── trainer.py
├── train.py
├── evaluate.py
└── requirements.txt
```

## Training Schedule

| Stage | Dataset | Iterations | LR |
|-------|---------|------------|----|
| 1 – Chairs | FlyingChairs | 100k | 4e-4 |
| 2 – Things | FlyingThings3D | 100k | 1.25e-4 |
| 3 – Sintel | Sintel Clean+Final + Things + HD1K | 100k | 1.25e-4 |
| 4 – KITTI  | KITTI-15 train | 50k  | 1.25e-4 |
| 5 – Spring | Spring train | 100k | 1.25e-4 |

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Pre-train on FlyingChairs
python train.py --config configs/default.yaml data.dataset=chairs

# Fine-tune on Sintel
python train.py --config configs/sintel_ft.yaml \
    training.checkpoint=checkpoints/things.pth

# Evaluate on Sintel
python evaluate.py --config configs/sintel_ft.yaml \
    --checkpoint checkpoints/sintel.pth --split final

# Ablation: 4 stages
python train.py --config configs/default.yaml \
    --overrides configs/ablations/stages_04.yaml
```

## Ablation Studies

The config system (OmegaConf) supports composing overrides:

```bash
# Vary number of HQS stages
for S in 4 8 12 16; do
  python train.py --config configs/default.yaml model.num_stages=$S \
      run_name=ablate_stages_$S
done

# Ablate proximal regularizer
python train.py --config configs/ablations/no_regularizer.yaml

# Compare correlation types
python train.py --config configs/ablations/local_corr.yaml
```
