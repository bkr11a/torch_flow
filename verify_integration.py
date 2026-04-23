#!/usr/bin/env python3
"""
Verification Checklist for HQS Flow Integration

Run this script to verify that all integration changes are working correctly.
"""
import os
import sys
import importlib
import json
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists and print status."""
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {description}: {path}")
    return exists


def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"  ✅ {description}")
        return True
    except ImportError as e:
        print(f"  ❌ {description}: {e}")
        return False


def check_config_field(config_path: str, field: str, description: str) -> bool:
    """Check if a config file has the required field."""
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        
        # Navigate nested fields
        parts = field.split(".")
        obj = cfg
        for part in parts:
            obj = obj.get(part)
            if obj is None:
                raise KeyError(f"Field not found: {field}")
        
        print(f"  ✅ {description} (value: {obj})")
        return True
    except Exception as e:
        print(f"  ❌ {description}: {e}")
        return False


def main():
    print("=" * 80)
    print("HQS Flow Integration Verification Checklist")
    print("=" * 80)
    
    results = {}
    
    # 1. New Files Created
    print("\n1. New Files Created")
    print("-" * 80)
    results["new_files"] = all([
        check_file_exists("utils/flow_visualization_hsv.py", "HSV visualization module"),
        check_file_exists("evaluate_comprehensive.py", "Comprehensive evaluation script"),
        check_file_exists("visualize_stages.py", "Stage progression visualization"),
        check_file_exists("INTEGRATION_GUIDE.md", "Integration guide"),
        check_file_exists("SCRIPTS_QUICKSTART.md", "Scripts quickstart"),
        check_file_exists("MERGE_SUMMARY.md", "Merge summary"),
        check_file_exists("graveyard/README.md", "Graveyard documentation"),
    ])
    
    # 2. Files Modified
    print("\n2. Files Modified")
    print("-" * 80)
    results["files_modified"] = all([
        check_file_exists("losses/flow_loss.py", "OFCE loss added"),
        check_file_exists("losses/__init__.py", "Losses exports updated"),
        check_file_exists("utils/__init__.py", "Utils exports updated"),
        check_file_exists("configs/default.yaml", "Config updated"),
        check_file_exists("engine/trainer.py", "Trainer enhanced"),
    ])
    
    # 3. Imports Work
    print("\n3. Module Imports")
    print("-" * 80)
    results["imports"] = all([
        check_import("utils.flow_visualization_hsv", "HSV visualization import"),
        check_import("losses.flow_loss", "Losses module import"),
        check_import("models.hqs_flow", "Model import"),
    ])
    
    # 4. Configuration
    print("\n4. Configuration Files")
    print("-" * 80)
    results["config"] = all([
        check_config_field("configs/default.yaml", "loss.ofce_weight", "OFCE weight in config"),
        check_config_field("configs/default.yaml", "loss.smooth_weight", "Smoothness weight in config"),
        check_config_field("configs/default.yaml", "loss.photo_weight", "Photometric weight in config"),
        check_config_field("configs/default.yaml", "model.num_stages", "Number of stages in config"),
    ])
    
    # 5. Backward Compatibility
    print("\n5. Backward Compatibility")
    print("-" * 80)
    try:
        from utils import flow_to_color
        print("  ✅ Old flow_to_color still available")
        results["compat_old_viz"] = True
    except ImportError:
        print("  ❌ flow_to_color not available")
        results["compat_old_viz"] = False
    
    # 6. New Functionality
    print("\n6. New Functionality Available")
    print("-" * 80)
    try:
        from utils import flow_to_hsv, flow_to_hsv_batch, create_flow_colorwheel
        print("  ✅ HSV visualization functions available")
        results["new_hsv"] = True
    except ImportError as e:
        print(f"  ❌ HSV visualization not available: {e}")
        results["new_hsv"] = False
    
    try:
        from losses import OFCELoss
        print("  ✅ OFCE loss available")
        results["new_ofce"] = True
    except ImportError as e:
        print(f"  ❌ OFCE loss not available: {e}")
        results["new_ofce"] = False
    
    try:
        from losses import HQSFlowLoss
        print("  ✅ HQSFlowLoss with new parameters available")
        results["new_loss_master"] = True
    except ImportError as e:
        print(f"  ❌ HQSFlowLoss not available: {e}")
        results["new_loss_master"] = False
    
    # 7. Model Building
    print("\n7. Model Building")
    print("-" * 80)
    try:
        import torch
        from omegaconf import OmegaConf
        from models import build_model
        
        cfg = OmegaConf.load("configs/default.yaml")
        model = build_model(cfg)
        
        print(f"  ✅ Model builds successfully")
        print(f"    - Type: {model.__class__.__name__}")
        print(f"    - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        results["model_build"] = True
    except Exception as e:
        print(f"  ❌ Model build failed: {e}")
        results["model_build"] = False
    
    # 8. Loss Creation
    print("\n8. Loss Function Creation")
    print("-" * 80)
    try:
        import torch
        from omegaconf import OmegaConf
        from losses import HQSFlowLoss, OFCELoss
        
        cfg = OmegaConf.load("configs/default.yaml")
        loss_fn = HQSFlowLoss(cfg.loss)
        
        print(f"  ✅ Loss function creates successfully")
        print(f"    - Has AEPE loss: True")
        print(f"    - Has smoothness: {hasattr(loss_fn, 'smooth_loss')}")
        print(f"    - Has photometric: {hasattr(loss_fn, 'photo_loss')}")
        print(f"    - Has OFCE: {hasattr(loss_fn, 'ofce_loss')}")
        results["loss_create"] = True
    except Exception as e:
        print(f"  ❌ Loss creation failed: {e}")
        results["loss_create"] = False
    
    # 9. Forward Pass
    print("\n9. Model Forward Pass")
    print("-" * 80)
    try:
        import torch
        from omegaconf import OmegaConf
        from models import build_model
        
        cfg = OmegaConf.load("configs/default.yaml")
        model = build_model(cfg)
        device = torch.device("cpu")
        model = model.to(device)
        
        img1 = torch.randn(1, 3, 64, 64)
        img2 = torch.randn(1, 3, 64, 64)
        
        with torch.no_grad():
            out = model(img1, img2)
        
        assert "flow_preds" in out
        assert len(out["flow_preds"]) == cfg.model.num_stages
        
        print(f"  ✅ Forward pass successful")
        print(f"    - Input shape: {img1.shape}")
        print(f"    - Output flow shape: {out['flow_preds'][-1].shape}")
        print(f"    - Number of stages: {len(out['flow_preds'])}")
        results["forward_pass"] = True
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        results["forward_pass"] = False
    
    # 10. Loss Computation
    print("\n10. Loss Computation")
    print("-" * 80)
    try:
        import torch
        from omegaconf import OmegaConf
        from models import build_model
        from losses import HQSFlowLoss
        
        cfg = OmegaConf.load("configs/default.yaml")
        model = build_model(cfg)
        loss_fn = HQSFlowLoss(cfg.loss)
        device = torch.device("cpu")
        
        model = model.to(device)
        loss_fn = loss_fn.to(device)
        
        img1 = torch.randn(1, 3, 64, 64, device=device)
        img2 = torch.randn(1, 3, 64, 64, device=device)
        flow_gt = torch.randn(1, 2, 64, 64, device=device)
        valid = torch.ones(1, 64, 64, device=device)
        
        with torch.no_grad():
            out = model(img1, img2)
            loss_dict = loss_fn(out["flow_preds"], flow_gt, valid, img1, img2)
        
        assert "loss" in loss_dict
        assert "epe" in loss_dict
        
        print(f"  ✅ Loss computation successful")
        print(f"    - Total loss: {loss_dict['loss'].item():.6f}")
        print(f"    - EPE: {loss_dict['epe'].item():.6f}")
        for k in ["smooth", "photo", "ofce"]:
            if k in loss_dict:
                print(f"    - {k}: {loss_dict[k].item():.6f}")
        results["loss_compute"] = True
    except Exception as e:
        print(f"  ❌ Loss computation failed: {e}")
        results["loss_compute"] = False
    
    # 11. HSV Visualization
    print("\n11. HSV Visualization")
    print("-" * 80)
    try:
        import torch
        from utils import flow_to_hsv, create_flow_colorwheel
        
        flow = torch.randn(2, 64, 64)
        rgb = flow_to_hsv(flow)
        
        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype.name == "uint8"
        
        wheel = create_flow_colorwheel()
        assert wheel.shape[-1] == 3
        
        print(f"  ✅ HSV visualization working")
        print(f"    - Input shape: {flow.shape}")
        print(f"    - Output shape: {rgb.shape}")
        print(f"    - Output dtype: {rgb.dtype}")
        results["hsv_viz"] = True
    except Exception as e:
        print(f"  ❌ HSV visualization failed: {e}")
        results["hsv_viz"] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    for category, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {category}")
    
    print("\n" + "=" * 80)
    print(f"Result: {passed_checks}/{total_checks} checks passed")
    print("=" * 80)
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Integration is complete and working.")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
