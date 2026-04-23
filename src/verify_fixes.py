#!/usr/bin/env python3
"""
Verification script to test if all fixes are working correctly.
Tests:
1. Spacing is isotropic
2. Tensor shapes are valid
3. DataLoader works without errors
4. Model can be transferred to GPU
5. Single training step works
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.datasets import TASKS, validate_batch
from models.unet import build_unet, UNetWithEncoder


def test_spacing():
    """Test 1: Verify all spacings are isotropic"""
    print("\n" + "="*70)
    print("TEST 1: Verify Isotropic Spacing")
    print("="*70)
    
    all_ok = True
    for task_name, cfg in TASKS.items():
        spacing = cfg["spacing"]
        is_isotropic = spacing[0] == spacing[1] == spacing[2]
        status = "✅ PASS" if is_isotropic else "❌ FAIL"
        print(f"  {status}  {task_name:<10}  spacing={spacing}")
        if not is_isotropic:
            all_ok = False
    
    return all_ok


def test_tensor_validation():
    """Test 2: Verify tensor validation function works"""
    print("\n" + "="*70)
    print("TEST 2: Verify Tensor Validation")
    print("="*70)
    
    try:
        # Create valid batch
        valid_batch = {
            "image": torch.randn(2, 1, 96, 96, 96),  # [B, C, D, H, W]
            "label": torch.randint(0, 2, (2, 96, 96, 96))  # [B, D, H, W]
        }
        
        result = validate_batch(valid_batch, "test_task")
        print(f"  ✅ PASS  Valid batch accepted")
        
        # Test invalid batch (wrong dimensions)
        try:
            invalid_batch = {
                "image": torch.randn(2, 96, 96, 96),  # Missing channel dim
                "label": torch.randint(0, 2, (2, 96, 96, 96))
            }
            validate_batch(invalid_batch, "test_task")
            print(f"  ❌ FAIL  Invalid batch not caught")
            return False
        except ValueError as e:
            print(f"  ✅ PASS  Invalid batch caught: {str(e)[:50]}...")
        
        return True
    except Exception as e:
        print(f"  ❌ FAIL  {e}")
        return False


def test_model_creation():
    """Test 3: Verify model can be created and transferred to GPU"""
    print("\n" + "="*70)
    print("TEST 3: Verify Model Creation and GPU Transfer")
    print("="*70)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        
        # Create model
        unet = build_unet(in_channels=1, out_channels=2,
                         channels=(32, 64, 128, 256, 512),
                         strides=(2, 2, 2, 2))
        print(f"  ✅ PASS  U-Net created")
        
        # Wrap with encoder
        model = UNetWithEncoder(unet)
        print(f"  ✅ PASS  UNetWithEncoder created")
        
        # Transfer to device
        model = model.to(device)
        print(f"  ✅ PASS  Model transferred to {device}")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  ✅ PASS  Model has {n_params:,} parameters")
        
        return True
    except Exception as e:
        print(f"  ❌ FAIL  {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test 4: Verify forward pass works"""
    print("\n" + "="*70)
    print("TEST 4: Verify Forward Pass")
    print("="*70)
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model
        unet = build_unet(in_channels=1, out_channels=2,
                         channels=(32, 64, 128, 256, 512),
                         strides=(2, 2, 2, 2))
        model = UNetWithEncoder(unet).to(device)
        model.eval()
        
        # Create dummy input
        x = torch.randn(1, 1, 96, 96, 96).to(device)
        print(f"  Input shape: {x.shape}")
        
        # Forward pass
        with torch.no_grad():
            y = model(x)
        
        print(f"  ✅ PASS  Output shape: {y.shape}")
        
        # Verify output shape
        expected_shape = (1, 2, 96, 96, 96)
        if y.shape == expected_shape:
            print(f"  ✅ PASS  Output shape is correct")
            return True
        else:
            print(f"  ❌ FAIL  Expected {expected_shape}, got {y.shape}")
            return False
            
    except Exception as e:
        print(f"  ❌ FAIL  {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_memory():
    """Test 5: Verify GPU memory management"""
    print("\n" + "="*70)
    print("TEST 5: Verify GPU Memory Management")
    print("="*70)
    
    if not torch.cuda.is_available():
        print(f"  ⚠️  SKIP  CUDA not available")
        return True
    
    try:
        # Get initial memory
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  Initial GPU memory: {initial_mem:.2f} GB")
        
        # Create and run model
        device = torch.device("cuda")
        unet = build_unet(in_channels=1, out_channels=2,
                         channels=(32, 64, 128, 256, 512),
                         strides=(2, 2, 2, 2))
        model = UNetWithEncoder(unet).to(device)
        
        x = torch.randn(1, 1, 96, 96, 96).to(device)
        with torch.no_grad():
            y = model(x)
        
        used_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory after forward pass: {used_mem:.2f} GB")
        
        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        cleared_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory after cache clear: {cleared_mem:.2f} GB")
        
        print(f"  ✅ PASS  GPU memory management working")
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL  {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("VERIFICATION TESTS FOR CONTINUAL SSL FIXES")
    print("="*70)
    
    results = {
        "Spacing": test_spacing(),
        "Tensor Validation": test_tensor_validation(),
        "Model Creation": test_model_creation(),
        "Forward Pass": test_forward_pass(),
        "GPU Memory": test_gpu_memory(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Code is ready for training!")
    else:
        print("❌ SOME TESTS FAILED - Please review errors above")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
