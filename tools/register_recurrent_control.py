#!/usr/bin/env python3
"""Idempotently register HQSCoreRecurrentControl with torch_flow's model API.

Run from anywhere; --repo points at the torch_flow repository root.
The script is deliberately anchor-based and fails if the expected pgma layout
has drifted rather than silently editing an unexpected file.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"Could not find expected anchor while editing {label}")
    return text.replace(old, new, 1)


def patch_hqs_flow(path: Path) -> bool:
    text = path.read_text()
    original = text

    text = replace_once(
        text,
        '                    ``hqs_lm_of`` (learned probabilistic correspondence and\n',
        '                    ``hqs_core_recurrent_control`` (equal-capacity generic\n'
        '                    recurrent control for the OF-A ablation), ``hqs_lm_of``\n'
        '                    (learned probabilistic correspondence and\n',
        label=str(path),
    )

    anchor = '''    if model_type in {"hqs_core", "hqscore", "core"}:\n        from hqs_pytorch.customML.customModels.HQSCore import HQSCore\n\n        return HQSCore(cfg.model)\n'''
    replacement = anchor + '''    if model_type in {\n        "hqs_core_recurrent_control",\n        "hqs_recurrent_control",\n        "recurrent_control",\n    }:\n        from hqs_pytorch.customML.customModels.HQSCoreRecurrentControl import (\n            HQSCoreRecurrentControl,\n        )\n\n        return HQSCoreRecurrentControl(cfg.model)\n'''
    text = replace_once(text, anchor, replacement, label=str(path))

    text = replace_once(
        text,
        '        "hqs_core | hqscore | core | hqs_lm_of | hqs_field_of | "\n',
        '        "hqs_core | hqscore | core | hqs_core_recurrent_control | "\n'
        '        "hqs_lm_of | hqs_field_of | "\n',
        label=str(path),
    )

    if text != original:
        path.write_text(text)
        return True
    return False


def patch_models_init(path: Path) -> bool:
    text = path.read_text()
    original = text

    text = replace_once(
        text,
        '    "HQSFlow", "HQSCore", "HQSLMOpticalFlow", "HQSFieldOpticalFlow",\n',
        '    "HQSFlow", "HQSCore", "HQSCoreRecurrentControl", "HQSLMOpticalFlow",\n'
        '    "HQSFieldOpticalFlow",\n',
        label=str(path),
    )

    anchor = '        "HQSCore": "hqs_pytorch.customML.customModels.HQSCore",\n'
    replacement = anchor + '''        "HQSCoreRecurrentControl": (\n            "hqs_pytorch.customML.customModels.HQSCoreRecurrentControl"\n        ),\n'''
    text = replace_once(text, anchor, replacement, label=str(path))

    if text != original:
        path.write_text(text)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="torch_flow repository root")
    args = parser.parse_args()

    repo = Path(args.repo).expanduser().resolve()
    if not (repo / "train_curriculum.py").is_file():
        raise SystemExit(f"Not a torch_flow repository root: {repo}")

    changed_flow = patch_hqs_flow(repo / "models" / "hqs_flow.py")
    changed_init = patch_models_init(repo / "models" / "__init__.py")

    if changed_flow or changed_init:
        print("Registered HQSCoreRecurrentControl.")
    else:
        print("HQSCoreRecurrentControl was already registered; no changes needed.")


if __name__ == "__main__":
    main()
