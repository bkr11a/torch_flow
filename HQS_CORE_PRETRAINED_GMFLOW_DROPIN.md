# HQSCore with the released GMFlow correspondence front end

This drop-in imports the correspondence-producing portion of the official
single-scale GMFlow model:

- 1/8 CNN backbone;
- six-layer Swin-style feature Transformer;
- global dot-product correspondence readout;
- feature-flow propagation parameters (loaded, but not routed initially).

The official convex upsampler is intentionally excluded. HQSCore retains its
own source-conditioned recurrent solver and final upsampler.

## 1. Download the official checkpoint

From the repository root:

```bash
python tools/download_gmflow_pretrained.py
```

This downloads the checkpoint distributed by the official GMFlow repository,
extracts `gmflow_things-e9887eda.pth`, validates characteristic state-dict
keys and writes it to:

```text
pretrained/gmflow_things-e9887eda.pth
```

If Google Drive changes its confirmation page, download the official archive
manually from:

<https://drive.google.com/file/d/1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2/view?usp=sharing>

The path can be overridden without editing YAML:

```bash
export GMFLOW_CHECKPOINT=/absolute/path/to/gmflow_things-e9887eda.pth
```

## 2. Run the acceptance gate

```bash
python preflight_pretrained_gmflow.py --device cuda
```

The preflight fails unless:

- every expected backbone, Transformer and propagation tensor loads;
- all shapes match the single-scale architecture;
- the official full-model upsampler is the only deliberately omitted module;
- the pretrained matcher is frozen and receives no gradients;
- the HQS solver receives finite gradients;
- raw routing is active and the forward/backward data gate is disabled.

Loading uses `torch.load(..., weights_only=True)`. A partial
`strict=False` import is not used.

## 3. Run the controlled Chairs experiment

```bash
python train_curriculum.py \
  --config configs/default.yaml \
  --override configs/dropins/16_hqs_core_pretrained_gmflow.yaml \
  --curriculum configs/curriculum/universal_replay_curriculum.yaml \
  --base-run-name hqs_core_pretrained_gmflow_raw_2_4_2_2 \
  --stop-after-stage u01_chairs_baseline \
  training.num_steps=50000 \
  training.val_every=5000 \
  training.save_every=10000 \
  mlflow.enabled=false
```

The overlay deliberately overrides the stage-one learning rate to `1e-4` and
disables AMP. CLI values still take final precedence.

## Initial routing decision

The first experiment uses:

```yaml
global_use_flow_propagation: false
global_propagation_routing: raw
global_fb_data_gate: false
global_fb_visibility_weight: 0.0
```

This isolates the pretrained correspondence measurement from the failed
forward/backward routing observed in the scratch-trained run. Bidirectional
decoding remains enabled so consistency diagnostics can still be logged.

The raw measurement is decoded at 1/8. It initializes the 1/16 state by a
flow-aware resize, after which the native 1/8 candidate is restored with the
coarse HQS correction using `native_residual` transition mode.

## Fine-tuning after the frozen experiment

Do not fine-tune the matcher in the first run. Once the frozen-front-end solver
is stable, resume its weights with:

```text
model.hqs_core.gmflow_freeze_pretrained=false
training.global_matcher_lr=1.0e-5
training.lr=1.0e-4
training.resume_mode=weights_only
training.checkpoint=/path/to/frozen/run/best.pth
```

`Trainer` creates a separate matcher parameter group when
`training.global_matcher_lr` is provided, and supplies both maximum learning
rates to `OneCycleLR`. When the matcher is frozen, its parameters are omitted
from the optimizer and the warm-up freeze controller cannot re-enable them.

## Compatibility notes

- Existing runs produced by `15_hqs_core_gmflow.yaml` retain the historical
  local residual-block graph. The released graph is enabled only by
  `gmflow_released_weights_compatible: true`.
- Input tensors in this repository are already in `[0,1]`; HQSCore applies the
  same ImageNet mean/std normalization used by GMFlow exactly once.
- A two-scale `gmflow_with_refine_*` checkpoint is rejected. This drop-in
  expects the basic single-scale checkpoint.
- A subsequent HQSCore curriculum checkpoint loads after the official weights
  and therefore correctly restores any locally fine-tuned matcher state.

## Source and licence

The weight layout and released computation graph correspond to the official
Apache-2.0 GMFlow implementation:

<https://github.com/haofeixu/gmflow>
