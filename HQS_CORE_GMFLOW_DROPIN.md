# HQSCore GMFlow Correspondence Drop-in

This overlay replaces HQSCore's weak one-off global matching representation
with a dedicated GMFlow-style measurement front end while leaving the
four-scale HQS solver, analytic update, learned data residual, proximal
operator and `[2, 4, 2, 2]` recurrence schedule unchanged.

The implementation targets `pgma` commit
`77c2c5b9f0957d09816c03f69bd116db5b55c904`.

## Implemented correspondence path

At `1/8` resolution the new path is:

1. the single-scale GMFlow CNN backbone (`64 -> 96 -> 128` channels);
2. split-window two-dimensional sine positional encoding;
3. six alternating unshifted/shifted GMFlow Transformer blocks;
4. scaled dot-product all-pairs correlation,
   `C = F1^T F2 / sqrt(128)`;
5. row-wise softmax with temperature `1.0`;
6. expected target coordinate minus source coordinate;
7. source self-attention flow propagation.

The raw and propagated candidates both receive unit-weight direct flow
supervision.  This replaces the previous `global_init_weight: 0.05` auxiliary
signal.  During the first 10,000 steps, only the dedicated measurement front
end is trainable; joint HQS optimisation follows.

No pretrained GMFlow weights are loaded.  This is an architecture transplant
trained inside the HQSFlow curriculum.  It must therefore be treated as a new
experiment rather than resumed strictly from a standard HQSCore checkpoint.

## Bidirectional consistency and occlusion-aware routing

Forward and reverse soft matches are decoded from `C` and `C^T`.  The CNN,
Transformer and correlation are evaluated only once.  For the forward flow,
the consistency residual is

```text
e12(x) = ||u12(x) + warp(u21, u12)(x)||2.
```

The hard GMFlow/UnFlow decision is

```text
occluded(x) = e12(x) > alpha * (||u12(x)||2 + ||u21(x)||2) + beta
```

or true when the forward endpoint is outside the target image.  The default
values are `alpha=0.01` and `beta=0.5` full-resolution pixels.  Since the test
is performed on the native `1/8` grid, the implementation uses `beta/8`.

A differentiable version supplies measurement reliability.  Raw matching is
retained where this reliability is high; source self-attention propagation is
used where it is low:

```text
q0 = r_fb * q_match + (1 - r_fb) * q_propagated.
```

The same geometric reliability multiplies the learned validity only in the
`1/16` and `1/8` data operators.  It never enters the proximal operator.  The
proximal continues to receive the split states and source context only.

The model exposes both native and full-resolution diagnostics:

- `global_fb_reliability` and `global_fb_occlusion`;
- `global_fb_raw_*` and `global_fb_propagated_*`;
- `global_init_candidate_flow_xy`;
- `global_init_propagated_flow_xy`;
- `global_init_solver_candidate_flow_xy`;
- `global_reverse_flow_xy`.

The hard mask is a geometric occlusion/invalidity estimate, not a calibrated
probability.  `global_fb_reliability` is the soft data gate.

## Preflight

Run the focused tests:

```bash
python -m pytest -q \
  tests/test_hqs_core_gmflow.py \
  tests/test_hqs_core.py
```

Then run the end-to-end preflight:

```bash
python preflight_hqscore_gmflow.py --device cuda
```

Do not begin the curriculum unless the JSON result contains:

```json
"passed": true
```

The preflight checks the ten-prediction contract, native `1/8` matching,
bidirectional decoding, bounded consistency reliability, direct supervision
of both matcher outputs, finite gradients through the complete GMFlow front
end, synthetic occlusion detection and the source-only proximal boundary.

## Curriculum command

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/15_hqs_core_gmflow.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_core_gmflow_fb_2_4_2_2_50000 \
  training.num_steps=50000 \
  training.val_every=5000 \
  training.save_every=10000 \
  training.log_every=100 \
  mlflow.enabled=false
```

As with the previous command, `training.num_steps=50000` applies to each
curriculum stage.  The total is 200,000 steps over four stages.  The 10,000
step matcher-only phase also occurs at the beginning of each stage because
the curriculum transfers weights with `resume_mode: weights_only`.

The dedicated Transformer and the `1/8` all-pairs matrix increase memory use.
If batch size six exceeds device memory, apply:

```bash
data.batch_size=4 val_data.batch_size=1
```

## Required ablation

The first experiment should retain a controlled attribution matrix:

| Run | GMFlow features | Strong matcher loss | Propagation | FB data gate | Purpose |
|---|---:|---:|---:|---:|---|
| A | no | no | no | no | existing HQSCore control |
| B | yes | yes | no | no | feature/matching transplant |
| C | yes | yes | yes | no | GMFlow propagation contribution |
| D | yes | yes | yes | yes | complete proposed drop-in |

Run B from the complete overlay with:

```bash
model.hqs_core.global_use_flow_propagation=false \
model.hqs_core.global_fb_data_gate=false
```

Run C with:

```bash
model.hqs_core.global_fb_data_gate=false \
model.hqs_core.global_propagation_routing=propagated
```

The primary diagnostic remains raw `1/8` candidate EPE, especially `s40+`,
alongside entropy, endpoint rank, propagated candidate EPE, FB reliability
accuracy, final EPE, boundary EPE and high-frequency recovery.  If the raw
candidate remains centre-seeking after the matcher-only phase, the transplant
has not learned GMFlow-quality logits and the full curriculum should stop.
