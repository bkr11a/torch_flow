__author__ = "Brad Rice"
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

# Standard library
import os
import sys

# PyPI packages
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary
from loguru import logger
from omegaconf import OmegaConf

# Custom modules
from models import build_model
from data import build_dataset, SintelDataset
from utils import flow_to_color, flow_to_hsv

# ---------------------------------------------------------------------------

def load_and_verify_flyingchairs():
    logger.info("Loading FlyingChairs dataset...")

    cfg = OmegaConf.create({'name':'chairs','root':'./data/benchmark_data/FlyingChairs_release', 'batch_size':1,'num_workers':0,'crop_size':[384,512]})
    dataset = build_dataset(cfg, split="train")
    logger.info("Dataset loaded with {} samples.", len(dataset))

    sample = dataset[0]
    logger.info("Sample keys: {}", sample.keys())
    logger.info("Image 1 shape: {}, Image 2 shape: {}, Flow shape: {}, Valid shape: {}",
                sample['image1'].shape, sample['image2'].shape, sample['flow'].shape, sample['valid'].shape)

    # Plot the first image pair, flow (as hsv (with quiver plot)) and valid mask for visual verification
    # We need to convert the flow to HSV for visualization. The valid mask can be shown as a binary image.
    # We also need to convert the flow from (H, W, 2) to (H, W, 3) for HSV visualization, where the third channel is the magnitude of the flow.
    flow = sample['flow'].numpy().transpose(1, 2, 0)
    valid = sample['valid'].numpy() * 255

    # Also need to convert the images from (3, H, W) to (H, W, 3) for visualization
    sample['image1'] = sample['image1'].numpy().transpose(1, 2, 0)
    sample['image2'] = sample['image2'].numpy().transpose(1, 2, 0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(sample['image1'])
    axes[0].set_title("Image 1")
    axes[0].axis('off')
    axes[1].imshow(sample['image2'])
    axes[1].set_title("Image 2")
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(flow_to_hsv(flow), cv2.COLOR_BGR2RGB))
    axes[2].set_title("Flow (HSV)")
    axes[2].axis('off')
    axes[3].imshow(valid, cmap='gray')
    axes[3].set_title("Valid")
    axes[3].axis('off')
    plt.show()

    logger.success("FlyingChairs training data split loading verification successful!")
    logger.info("Loading FlyingChairs validation split...")
    dataset_val = build_dataset(cfg, split="val")
    logger.info("Validation dataset loaded with {} samples.", len(dataset_val))
    sample_val = dataset_val[0]
    logger.info("Sample keys: {}", sample_val.keys())
    logger.info("Image 1 shape: {}, Image 2 shape: {}, Flow shape: {}, Valid shape: {}",
                sample_val['image1'].shape, sample_val['image2'].shape, sample_val['flow'].shape, sample_val['valid'].shape)

    # Plot the first validation sample as well
    flow_val = sample_val['flow'].numpy().transpose(1, 2, 0)
    valid_val = sample_val['valid'].numpy() * 255
    sample_val['image1'] = sample_val['image1'].numpy().transpose(1, 2, 0)
    sample_val['image2'] = sample_val['image2'].numpy().transpose(1, 2, 0)
    fig, axes = plt.subplots(1, 4, figsize=(20, 15))
    axes[0].imshow(sample_val['image1'])
    axes[0].set_title("Val Image 1")
    axes[0].axis('off')
    axes[1].imshow(sample_val['image2'])
    axes[1].set_title("Val Image 2")
    axes[1].axis('off')
    axes[2].imshow(cv2.cvtColor(flow_to_hsv(flow_val), cv2.COLOR_BGR2RGB))
    axes[2].set_title("Val Flow (HSV)")
    axes[2].axis('off')
    axes[3].imshow(valid_val, cmap='gray')
    axes[3].set_title("Val Valid")
    axes[3].axis('off')
    plt.show()

    logger.success("FlyingChairs validation data split loading verification successful!")

    logger.success("FlyingChairs data loading verification successful!")

def load_and_verify_sintel():
    logger.info("Loading Sintel dataset...")

    dstypes = ["clean", "final"]
    for dstype in dstypes:
        logger.info("Loading Sintel training split for {}...", dstype)
        dataset = SintelDataset(root="./data/benchmark_data/Sintel/MPI-Sintel-complete", split="train", dstype=dstype, use_occlusions=True, use_invalid=True)
        logger.info("Dataset loaded with {} samples.", len(dataset))
        sample = dataset[0]
        logger.info("Sample keys: {}", sample.keys())

        # Plot the first image pair, flow (as hsv (with quiver plot)) and valid mask for visual verification as well as occlusion and invalid masks if they exist
        flow = sample['flow'].numpy().transpose(1, 2, 0)
        valid = sample['valid'].numpy() * 255
        sample['image1'] = sample['image1'].numpy().transpose(1, 2, 0)
        sample['image2'] = sample['image2'].numpy().transpose(1, 2, 0)
        fig, axes = plt.subplots(3, 2, figsize=(25, 15))
        axes[0, 0].imshow(sample['image1'])
        axes[0, 0].set_title("Image 1")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(sample['image2'])
        axes[0, 1].set_title("Image 2")
        axes[0, 1].axis('off')
        axes[1, 0].imshow(cv2.cvtColor(flow_to_hsv(flow), cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Flow (HSV)")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(valid, cmap='gray')
        axes[1, 1].set_title("Valid")
        axes[1, 1].axis('off')
        if sample.get('occlusion') is not None:
            occlusion = sample['occlusion'].numpy() * 255
            axes[2, 0].imshow(occlusion, cmap='gray')
            axes[2, 0].set_title("Occlusion")
            axes[2, 0].axis('off')
        if sample.get('invalid') is not None:
            invalid = sample['invalid'].numpy() * 255
            axes[2, 1].imshow(invalid, cmap='gray')
            axes[2, 1].set_title("Invalid")
            axes[2, 1].axis('off')
        plt.tight_layout()
        plt.show()

        logger.success("Sintel training data split loading verification successful!")
        logger.info("Loading Sintel validation split for {}...", dstype)
        dataset_val = SintelDataset(root="./data/benchmark_data/Sintel/MPI-Sintel-complete", split="val", dstype=dstype, use_occlusions=True, use_invalid=True)

        logger.info("Validation dataset loaded with {} samples.", len(dataset_val))
        sample_val = dataset_val[0]
        logger.info("Sample keys: {}", sample_val.keys())
        flow_val = sample_val['flow'].numpy().transpose(1, 2, 0)
        valid_val = sample_val['valid'].numpy() * 255
        sample_val['image1'] = sample_val['image1'].numpy().transpose(1, 2, 0)
        sample_val['image2'] = sample_val['image2'].numpy().transpose(1, 2, 0)
        fig, axes = plt.subplots(3, 2, figsize=(25, 15))
        axes[0, 0].imshow(sample_val['image1'])
        axes[0, 0].set_title("Val Image 1")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(sample_val['image2'])
        axes[0, 1].set_title("Val Image 2")
        axes[0, 1].axis('off')
        axes[1, 0].imshow(cv2.cvtColor(flow_to_hsv(flow_val), cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Val Flow (HSV)")
        axes[1, 0].axis('off')
        axes[1, 1].imshow(valid_val, cmap='gray')
        axes[1, 1].set_title("Val Valid")
        axes[1, 1].axis('off')
        if sample_val.get('occlusion') is not None:
            occlusion_val = sample_val['occlusion'].numpy() * 255
            axes[2, 0].imshow(occlusion_val, cmap='gray')
            axes[2, 0].set_title("Val Occlusion")
            axes[2, 0].axis('off')
        if sample_val.get('invalid') is not None:
            invalid_val = sample_val['invalid'].numpy() * 255
            axes[2, 1].imshow(invalid_val, cmap='gray')
            axes[2, 1].set_title("Val Invalid")
            axes[2, 1].axis('off')
        plt.tight_layout()
        plt.show()

        logger.info("Loading Sintel test split for test...")
        dataset_test = SintelDataset(root="./data/benchmark_data/Sintel/MPI-Sintel-complete", split="test", dstype=dstype, use_occlusions=True, use_invalid=True)
        logger.info("Test dataset loaded with {} samples.", len(dataset_test))
        sample_test = dataset_test[0]
        logger.info("Sample keys: {}", sample_test.keys())
        sample_test['image1'] = sample_test['image1'].numpy().transpose(1, 2, 0)
        sample_test['image2'] = sample_test['image2'].numpy().transpose(1, 2, 0)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes[0].imshow(sample_test['image1'])
        axes[0].set_title("Test Image 1")
        axes[0].axis('off')
        axes[1].imshow(sample_test['image2'])
        axes[1].set_title("Test Image 2")
        axes[1].axis('off')
        plt.tight_layout()
        plt.show()

        logger.success("Sintel test data split loading verification successful!")

    logger.success("Sintel data loading verification successful!")

def load_and_verify_kitti():
    logger.info("Loading KITTI dataset...")

def main():
    # TODO: Add some sample data loading and visualization here to verify the dataset code.
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "verify_data_loading.log"), rotation="1 MB")
    logger.info("#"*60)
    logger.info("Starting data loading verification...")
    logger.info("#"*60)

    load_and_verify_flyingchairs()
    load_and_verify_sintel()
    load_and_verify_kitti()
    logger.success("Data loading verification successful!")

if __name__ == "__main__":
    main()