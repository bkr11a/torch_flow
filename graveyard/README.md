# Graveyard

This folder stores deprecated or superseded code kept only for reference.

It is intentionally not part of the primary training and evaluation path.

## Purpose

Use this directory for:

- historical context
- migration reference
- checking prior implementations when debugging regressions

Do not build new features on graveyard code unless you are intentionally reviving archived behavior.

## Cookbook Commands

### List archived assets

```bash
ls -la graveyard
```

Purpose: see what is archived.

### Compare archived vs active implementation

```bash
diff -u graveyard/README.md README.md | sed -n '1,200p'
```

Purpose: quickly compare documentation intent and current direction.

### Search for references to graveyard code

```bash
rg "graveyard|hqs_pytorch_original" -n
```

Purpose: ensure active paths do not accidentally depend on deprecated modules.

## Migration Guidance

1. Identify the required behavior from archived code.
2. Re-implement in active modules under `models/`, `data/`, `engine/`, or `utils/`.
3. Add tests/verification in the active pipeline.
4. Keep graveyard files unchanged except for documentation.
