# HQS-Field-OF drop-in

## Purpose

`HQSFieldOpticalFlow` implements a revised inverse-problem formulation for
optical flow while preserving the project hypothesis that a trainable,
finite-depth HQS solver can organise learned inference around explicit
operators.

The model is added as `model_type: hqs_field_of`. The existing `hqs_core` and
`hqs_lm_of` paths remain unchanged so completed negative and baseline runs
remain reproducible.

## Operator structure

The forward path is:

1. extract a four-scale source/target matching pyramid and an independent
   source-only context pyramid;
2. apply gated self/cross-transformer enhancement to the matching branch;
3. retain the strongest global or local correlation hypotheses instead of
   reducing the cost volume to one soft-argmax vector;
4. recurrently calibrate each hypothesis, its mixture weight, matchability and
   full symmetric `2 x 2` precision;
5. update the HQS data state through the closed-form quadratic-majorisation
   system in `solve_correlation_mixture_hqs_increment`;
6. complete the auxiliary flow field with a positive-weight graph solve whose
   affinities depend only on the source frame;
7. convexly upsample the final `1/2`-resolution field.

The canonical path deliberately omits:

- the feature-constancy Gauss-Newton/LM residual;
- photometric or OFCE terms in `model.forward`;
- a target-conditioned flow vector after the analytic data solve;
- a free learned flow-residual head in the proximal operator;
- ground-truth visibility inputs at inference.

## Changed and added files

- `hqs_pytorch/customML/customModels/HQSFieldOpticalFlow.py`
- `models/hqs_field_components.py`
- `models/hqs_core_components.py`
- `models/hqs_flow.py`
- `models/__init__.py`
- `hqs_pytorch/__init__.py`
- `hqs_pytorch/customML/customModels/__init__.py`
- `losses/flow_loss.py`
- `configs/dropins/10_hqs_field_of.yaml`
- `smoketest_hqs_field.py`
- `tests/test_hqs_field.py`
- `tests/test_import_and_logging.py`

## Verification

Run the targeted tests:

```bash
python -m pytest -q \
  tests/test_hqs_field.py \
  tests/test_hqs_lm.py \
  tests/test_hqs_core.py \
  tests/test_import_and_logging.py
```

Run the reduced forward/backward smoke test:

```bash
python smoketest_hqs_field.py --device cuda
```

Run the complete configured forward test only after the reduced test passes:

```bash
python smoketest_hqs_field.py \
  --device cuda \
  --full-config
```

## Training sequence

Do not resume an HQS-LM or HQSCore checkpoint. The observation decoder and
proximal operator are structurally different.

First use a 2,000-step numerical and memory gate:

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/10_hqs_field_of.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_field_of_stability_gate \
  training.num_steps=2000 \
  training.val_every=1000 \
  training.save_every=2000 \
  training.log_every=20 \
  training.checkpoint=null \
  mlflow.enabled=false
```

If it remains finite, run the matched 30,000-step Chairs decision pilot:

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/10_hqs_field_of.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_field_of_2_4_2_2_pilot \
  training.num_steps=30000 \
  training.val_every=2500 \
  training.save_every=5000 \
  training.log_every=100 \
  training.checkpoint=null \
  mlflow.enabled=false
```

## Required decision diagnostics

For each scale and iteration, compare:

- `hypothesis_proposal_lows` against ground truth;
- `hypothesis_responsibility_lows` and
  `responsibility_entropy_lows`;
- `measurement_support_lows` in matched, unmatched and boundary regions;
- `data_flow_low` against `flow_low`;
- `delta_match_low` against `delta_prior_low`;
- `cycle_support_lows`;
- `lm_inverse_trace_lows` and `lm_condition_lows`.

Define the operator contributions:

```text
Delta_data  = EPE(previous proximal) - EPE(post data solve)
Delta_field = EPE(post data solve)   - EPE(post field proximal)
```

The formulation is supported only if the data solve improves visible
correspondence regions and the field proximal improves unmatched or
low-support regions without degrading motion boundaries.

## Evidence boundary

This drop-in is an implemented research hypothesis. It has not been trained or
evaluated in this workspace. Static syntax, configuration invariants and
independent numerical checks do not establish benchmark accuracy,
generalisation, robustness or publication readiness.
