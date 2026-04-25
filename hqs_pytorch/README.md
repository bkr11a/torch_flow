# hqs_pytorch Module Notes

This directory preserves the TensorFlow-port lineage and supporting custom modules used during integration and audits.

Use the main repository implementation in `models/` and `engine/` for current training runs.

## Mathematical Context

The ported HQS-style design follows the same alternating structure:

$$
E(u, v) = D(u) + \frac{\mu}{2}\|u-v\|_2^2 + \lambda R(v)
$$

With iterative updates:

$$
u^{k+1} \leftarrow \text{data/update block}(u^k, v^k, I_1, I_2)
$$

$$
v^{k+1} \leftarrow \text{proximal block}(u^{k+1}, I_1)
$$

Several custom layers in this directory mirror those algorithmic pieces, including gradient estimation, warping, correlation, and iterative solver utilities.

## What Lives Here

- `customML/customLayers/`: lower-level building blocks (correlation, warping, HQS iteration helpers)
- `customML/customLosses/`: loss variants (AEPE, OFCE, angular, etc.)
- `customML/customModels/`: reference model variants from the port effort
- `AUDIT_REPORT.md`, `IMPLEMENTATION_FIXES.md`: issue history and fixes

## When To Use This Directory

Use this directory when you need:

- historical parity checks against the TF-port logic
- reference implementations from the integration/audit period
- low-level custom layer experimentation

For normal training/evaluation in this repository, use top-level scripts and configs.

## Cookbook Commands

### Inspect available custom layers

```bash
ls hqs_pytorch/customML/customLayers
```

Purpose: quick view of module inventory.

### Read audit and fix history

```bash
sed -n '1,220p' hqs_pytorch/AUDIT_REPORT.md
sed -n '1,220p' hqs_pytorch/IMPLEMENTATION_FIXES.md
```

Purpose: understand bug history and rationale of fixes.

### Run the main trainer (recommended path)

```bash
python train.py --config configs/default.yaml
```

Purpose: execute the maintained training path rather than direct use of legacy port-only model wrappers.

### Evaluate with current framework

```bash
python evaluate.py \
  --config configs/default.yaml \
  --checkpoint checkpoints/hqs_flow_default/best.pth
```

Purpose: validate checkpoints with current metrics pipeline.

## Notes on Flow Conventions

The broader project uses consistent flow conventions in current `models/` code paths and metric utilities.
If comparing against historical artifacts in this directory, always verify channel order assumptions before reusing old snippets.
