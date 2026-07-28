# HQSCore DeepMatch Drop-in

## Scope

This patch upgrades the HQSCore measurement front end while preserving the
existing inverse solver:

```text
strong pair features
    -> distinct global correspondence modes
    -> cycle-consistent coarse initial state
    -> unchanged HQSCore data/proximal iterations
```

The default `hqs_core` configuration remains unchanged. The new path is enabled
only by `configs/dropins/13_hqs_core_deep_match.yaml`.

Base repository revision:

```text
ea88e909178c6db14b266d82887aa998dceae735
branch: pgma
```

## Implemented changes

### Stronger matching pyramid

- deeper shared Siamese encoder: `[2, 3, 4, 4]` residual blocks;
- feature widths `[48, 96, 144, 192]`;
- matching widths `[96, 128, 160, 192]`;
- bidirectional top-down and bottom-up multiscale fusion;
- four global symmetric self/cross-attention blocks at `1/16`;
- two alternating shifted-window self/cross-attention blocks at `1/8`;
- native-resolution window processing with no pooling fallback;
- source-only context projections remain outside cross-frame attention.

### Better correspondence decoding

- full all-pairs correlation at `1/16`;
- four spatially distinct modes selected by non-maximum suppression;
- local expectation inside each mode for sub-pixel decoding;
- reverse modes decoded from the transposed correlation matrix;
- forward modes reranked by forward-backward cycle support;
- selected confidence routes the initialization as full-flow or zero;
- the forward pass never multiplies a valid flow magnitude by confidence;
- the soft form of the gate is used only as a straight-through gradient.

### New diagnostics

The model output now includes:

```text
global_init_candidate_flow_xy
global_init_acceptance
global_init_soft_acceptance
global_init_selected_index
global_init_cycle_support
global_init_cycle_error
global_topk_flow_xy
global_topk_map_flow_xy
global_topk_probabilities
global_topk_cycle_support
global_topk_cycle_error
global_reverse_flow_xy
global_reverse_confidence
matching_transformer_blends
gmflow_reverse_flow_yx
```

`gmflow_init_flow_yx` is the raw selected matcher candidate for the DeepMatch
path. This keeps correspondence supervision active even when the solver router
rejects the candidate. `global_init_flow_xy` is the state actually admitted to
the HQS solver.

## Apply the patch

From the root of a clean checkout of the `pgma` branch:

```bash
git apply --check HQSCore-DeepMatch-pgma-ea88e90.patch
git apply HQSCore-DeepMatch-pgma-ea88e90.patch
```

The patch is intentionally based on commit `ea88e90`. If the live branch has
moved, run the check first and resolve only genuine overlapping edits.

## Verification

Run the focused unit tests:

```bash
python -m pytest -q tests/test_hqs_core.py
```

Run the repository smoke test:

```bash
python smoketest_hqs_core_deep_match.py
```

The smoke test checks:

- the ten-prediction `2+4+2+2` forward contract;
- top-four global modes and per-mode cycle support;
- full-resolution reverse-flow diagnostics;
- finite flow outputs;
- gradients through the pair-feature interaction branch.

## Training

### Short canary

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/13_hqs_core_deep_match.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_core_deep_match_canary \
  training.num_steps=20000 \
  training.val_every=2500 \
  training.save_every=5000 \
  training.log_every=100 \
  mlflow.enabled=false
```

### Full curriculum

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/13_hqs_core_deep_match.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_core_deep_match
```

## Required controlled comparisons

The implementation exposes independent feature and decoder switches:

| Variant | `matching_pyramid` | `global_decoder` | Purpose |
|---|---|---|---|
| C0 | `standard` | `soft_expectation` | Existing HQSCore control |
| M1 | `standard` | `multimodal_cycle` | Decoder-only effect |
| F1 | `deep_bidirectional` | `soft_expectation` | Feature-only effect |
| FM1 | `deep_bidirectional` | `multimodal_cycle` | Complete drop-in |

For M1, add the decoder options from overlay 13 to overlay 07 but retain the
original encoder widths. For F1, retain the DeepMatch feature settings and set:

```yaml
model:
  hqs_core:
    global_decoder: soft_expectation
    global_confidence_gated: true
    global_confidence_floor: 0.05
```

Keep the optimizer, curriculum, data order, augmentations, crop, batch size,
loss weights and seed fixed across the four variants.

## Decision metrics

Do not select the model from final EPE alone. Report:

- Recall@1 and Recall@4 within 1, 3 and 5 full-resolution pixels;
- top-1 and oracle top-4 correspondence EPE;
- initialization EPE and F1;
- cycle-support precision/recall;
- EPE versus accepted initialization coverage;
- final EPE/F1 by `s0-10`, `s10-40` and `s40+`;
- boundary EPE and high-frequency alignment/recovery;
- data EPE gain and proximal EPE gain.

The central success condition is that improved candidate recall is converted
into positive data-step gain and lower final EPE. If Recall@4 improves but the
data-step gain remains negative, the next bottleneck is the
measurement-to-data-update interface rather than feature extraction.

## Memory notes

At the canonical training crop, `1/16` attention is global and `1/8` attention
uses `8x8` shifted windows. Gradient checkpointing is enabled for the pair
transformers. If memory remains limiting:

1. reduce `matching_transformer_depth` from `[0,0,2,4]` to `[0,0,2,2]`;
2. reduce the effective batch size and preserve it with gradient accumulation;
3. reduce `match_channels`, but keep each attention width divisible by its
   configured head count.

Do not lower the input or matching-grid resolution as the first response: that
would directly weaken the boundary and thin-structure hypothesis being tested.
