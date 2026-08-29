#!/usr/bin/env python3
"""Idempotently register OF-A4 and OF-B model types in the repository."""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Registration anchor {label!r} expected exactly once, found {count}"
        )
    return text.replace(old, new, 1)


def patch_hqs_flow(path: Path) -> None:
    text = path.read_text()

    old_doc = '''                    ``hqs_core_recurrent_control`` (equal-capacity generic\n                    recurrent control for the OF-A ablation), ``hqs_lm_of``\n'''
    new_doc = '''                    ``hqs_core_recurrent_control`` (equal-capacity generic\n                    recurrent control for the OF-A ablation),\n                    ``hqs_core_single_state`` (OF-A4 no-split-state control),\n                    ``hqs_core_operator_ablation`` (OF-B operator controls),\n                    ``hqs_lm_of``\n'''
    text = replace_once(text, old_doc, new_doc, label="hqs_flow docstring")

    anchor = '''        return HQSCoreRecurrentControl(cfg.model)\n    if model_type in {"hqs_lm_of", "hqslm_of", "hqs-lm-of"}:\n'''
    replacement = '''        return HQSCoreRecurrentControl(cfg.model)\n    if model_type in {\n        "hqs_core_single_state",\n        "hqs_single_state",\n        "single_state",\n    }:\n        from hqs_pytorch.customML.customModels.HQSCoreSingleState import (\n            HQSCoreSingleState,\n        )\n\n        return HQSCoreSingleState(cfg.model)\n    if model_type in {\n        "hqs_core_operator_ablation",\n        "hqs_operator_ablation",\n        "operator_ablation",\n    }:\n        from hqs_pytorch.customML.customModels.HQSCoreOperatorAblation import (\n            HQSCoreOperatorAblation,\n        )\n\n        return HQSCoreOperatorAblation(cfg.model)\n    if model_type in {"hqs_lm_of", "hqslm_of", "hqs-lm-of"}:\n'''
    text = replace_once(text, anchor, replacement, label="hqs_flow factory")

    old_error = '''        "hqs_core | hqscore | core | hqs_core_recurrent_control | "\n        "hqs_lm_of | hqs_field_of | "\n'''
    new_error = '''        "hqs_core | hqscore | core | hqs_core_recurrent_control | "\n        "hqs_core_single_state | hqs_core_operator_ablation | "\n        "hqs_lm_of | hqs_field_of | "\n'''
    text = replace_once(text, old_error, new_error, label="hqs_flow error list")
    path.write_text(text)


def patch_models_init(path: Path) -> None:
    text = path.read_text()
    old_all = '''    "HQSFlow", "HQSCore", "HQSCoreRecurrentControl", "HQSLMOpticalFlow",\n    "HQSFieldOpticalFlow",\n'''
    new_all = '''    "HQSFlow", "HQSCore", "HQSCoreRecurrentControl",\n    "HQSCoreSingleState", "HQSCoreOperatorAblation", "HQSLMOpticalFlow",\n    "HQSFieldOpticalFlow",\n'''
    text = replace_once(text, old_all, new_all, label="models __all__")

    old_map = '''        "HQSCoreRecurrentControl": (\n            "hqs_pytorch.customML.customModels.HQSCoreRecurrentControl"\n        ),\n        "HQSLMOpticalFlow": (\n'''
    new_map = '''        "HQSCoreRecurrentControl": (\n            "hqs_pytorch.customML.customModels.HQSCoreRecurrentControl"\n        ),\n        "HQSCoreSingleState": (\n            "hqs_pytorch.customML.customModels.HQSCoreSingleState"\n        ),\n        "HQSCoreOperatorAblation": (\n            "hqs_pytorch.customML.customModels.HQSCoreOperatorAblation"\n        ),\n        "HQSLMOpticalFlow": (\n'''
    text = replace_once(text, old_map, new_map, label="models lazy map")
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_root", type=Path)
    args = parser.parse_args()
    repo = args.repo_root.expanduser().resolve()
    patch_hqs_flow(repo / "models" / "hqs_flow.py")
    patch_models_init(repo / "models" / "__init__.py")
    print("Registered HQSCoreSingleState and HQSCoreOperatorAblation.")


if __name__ == "__main__":
    main()
