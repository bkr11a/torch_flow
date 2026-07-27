from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from train import setup_logging


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_and_leaf_imports_do_not_cycle():
    code = "\n".join(
        (
            "from hqs_pytorch.customML.customModels.occlusion_geometry import FlowGeometry",
            "from hqs_pytorch import HQSFlowModel",
            "from hqs_pytorch.customML.customModels import HQSCore, HQSLMOpticalFlow, HQSFieldOpticalFlow, HQSOTOpticalFlow, HQSFieldOpticalFlowV2, HQSLMSceneFlow",
            "from models import build_model",
            "from losses import HQSFlowLoss, HQSSceneFlowLoss",
            "assert all(x is not None for x in (FlowGeometry, HQSFlowModel, HQSCore, HQSLMOpticalFlow, HQSFieldOpticalFlow, HQSOTOpticalFlow, HQSFieldOpticalFlowV2, HQSLMSceneFlow, build_model, HQSFlowLoss, HQSSceneFlowLoss))",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_hierarchical_run_name_creates_log_parent(tmp_path):
    setup_logging(str(tmp_path), "dropins/01_causal_prior")
    assert (tmp_path / "dropins").is_dir()
