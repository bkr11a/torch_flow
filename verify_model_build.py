__author__ = "Brad Rice"
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import os
import sys

# PyPI packages
import torch

from torchinfo import summary

from loguru import logger
from omegaconf import OmegaConf

# Custom modules
from models import build_model

# ---------------------------------------------------------------------------

def main():

    # Create the new logger instance and configure it to write to a file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "verify_model_build.log"), rotation="1 MB")

    logger.info("#"*60)
    logger.info("Starting model build verification...")
    logger.info("#"*60)

    cfg = OmegaConf.load("configs/default.yaml")
    logger.info("Loaded config:\n{}", OmegaConf.to_yaml(cfg))
    model = build_model(cfg)
    logger.info("Built model:\n{}", model)
    out = model(
        torch.randn(1, 3, 256, 256),  # img1
        torch.randn(1, 3, 256, 256),  # img2
    )

    logger.info("Output keys: {}", out.keys())
    logger.info("Output \'flow_preds\' length: {}", len(out['flow_preds']))
    expected_iters = cfg.model.model_backbone.num_hqs_iterations

    assert len(out['flow_preds']) == expected_iters, (
        f"Expected {expected_iters} flow predictions, "
        f"but got {len(out['flow_preds'])}."
    )
    logger.success("Output flow_preds length matches model_backbone.num_hqs_iterations.")

    logger.info("Final Predicted Flow Shape: {}", out['flow_preds'][-1].shape)
    logger.success("Model build successful!")

    model_summary = summary(model, input_data=[
        torch.randn(1, 3, 436, 1024),  # img1
        torch.randn(1, 3, 436, 1024),  # img2
    ])
    logger.info("Model summary:\n{}", model_summary)

    logger.info(f"Total Parameters: {model_summary.total_params:,}")
    logger.info(f"Trainable Parameters: {model_summary.trainable_params:,}")
    logger.info(f"Non-trainable Parameters: {model_summary.total_params - model_summary.trainable_params:,}")

    logger.info(f"Model Parameter Counts: {model.param_count()}")

    logger.success("All checks passed. Model build verification complete.")

if __name__ == "__main__":
    main()
