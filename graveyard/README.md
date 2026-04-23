# Graveyard 🪦

This folder contains deprecated code, experimental implementations, and superseded components that are no longer part of the primary codebase.

## Contents

### hqs_pytorch_original/
The original TensorFlow-ported HQSFlowModel from `hqs_pytorch/` before integration with the main training framework. This implementation, while containing important bug fixes, has been superseded by the unified `models/hqs_flow.py` which incorporates:
- Full configuration system compatibility
- Integrated weight-sharing mechanisms (share_all, share_none, etc.)
- Compatible with the standard training pipeline and metrics
- Bug fixes from hqs_pytorch backported where applicable

**Why deprecated**: The main `models/hqs_flow.py` is now the authoritative implementation with all necessary features and bug fixes applied, maintained in a single location for consistency.

---

## Migration Notes

If you need to reference or restore code from the graveyard:

1. **Bug fixes from hqs_pytorch**: Check `models/hqs_flow.py` for issues #4, #5, #10 implementations
2. **Original layers**: If you need specific layer implementations, check graveyard backup
3. **Historical context**: AUDIT_REPORT.md documents all fixes that were applied

---

Generated: 2026-04-23
