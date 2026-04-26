#!/usr/bin/env python3
"""
Phase 1 Validation Script — Quick Test of SSL+KD Pipeline

Run this BEFORE the full 3-fold CV experiment to verify:
1. Data loads correctly
2. Models train without crashing  
3. KD loss is properly scaled
4. Results look reasonable

Usage:
    python phase1_validation.py

Expected runtime: ~30 minutes on GPU
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add src to path
REPO_DIR = Path(__file__).parent
SRC_DIR = REPO_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Quick config for validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
VALIDATION_EPOCHS = 5  # Just 5 epochs to test

def set_seed():
    """Set all random seeds for reproducibility."""
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_data_loading():
    """Test that heart dataset loads correctly."""
    print("🔍 Testing data loading...")
    
    from data.datasets import build_task_roots, verify_datasets, get_file_list
    
    # Use local data path (adjust as needed)
    task_roots = build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon"))
    
    if not verify_datasets(task_roots):
        print("❌ Dataset verification failed")
        return False
    
    train_files, val_files = get_file_list(task_roots, "heart")
    print(f"✅ Found {len(train_files)} train, {len(val_files)} val files")
    
    if len(train_files) < 10:
        print("⚠️  Warning: Very few training files, results may be unstable")
    
    return True

def test_model_creation():
    """Test model creation and pretrained weight loading."""
    print("🔍 Testing model creation...")
    
    from models.unet import build_unet, UNetWithEncoder
    
    # Test basic U-Net creation
    unet = build_unet(in_channels=1, out_channels=2,
                      channels=(32, 64, 128, 256, 512),
                      strides=(2, 2, 2, 2))
    model = UNetWithEncoder(unet).to(DEVICE)
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 96, 96, 96).to(DEVICE)
    with torch.inference_mode():
        output = model(dummy_input)
    
    expected_shape = (1, 2, 96, 96, 96)
    if output.shape == expected_shape:
        print(f"✅ Model forward pass: {output.shape}")
        return True
    else:
        print(f"❌ Wrong output shape: {output.shape}, expected {expected_shape}")
        return False

def test_kd_loss_scaling():
    """Test that KD loss is properly scaled relative to DiceCE."""
    print("🔍 Testing KD loss scaling...")
    
    import torch.nn.functional as F
    from monai.losses import DiceCELoss
    
    # Create dummy predictions and labels
    batch_size = 2
    preds = torch.randn(batch_size, 2, 96, 96, 96).to(DEVICE)
    labels = torch.randint(0, 2, (batch_size, 1, 96, 96, 96)).to(DEVICE).long()
    
    # Compute DiceCE loss
    dice_ce = DiceCELoss(to_onehot_y=True, softmax=True)
    dice_loss = dice_ce(preds, labels)
    
    # Compute KD loss (student vs teacher, both same predictions for test)
    T = 2.0
    teacher_soft = F.softmax(preds.detach() / T, dim=1)
    student_log = F.log_softmax(preds / T, dim=1)
    
    # Test old (wrong) scaling
    kd_loss_wrong = F.kl_div(student_log, teacher_soft, reduction="mean") * (T ** 2)
    
    # Test new (correct) scaling  
    n_voxels = preds.shape[2] * preds.shape[3] * preds.shape[4]
    kd_loss_correct = F.kl_div(student_log, teacher_soft, reduction="sum") / (batch_size * n_voxels)
    kd_loss_correct = kd_loss_correct * (T ** 2)
    
    print(f"  DiceCE loss:     {dice_loss.item():.6f}")
    print(f"  KD loss (wrong): {kd_loss_wrong.item():.6f}")
    print(f"  KD loss (fixed): {kd_loss_correct.item():.6f}")
    
    # Check if scaling is reasonable
    ratio_wrong = kd_loss_wrong.item() / dice_loss.item()
    ratio_correct = kd_loss_correct.item() / dice_loss.item()
    
    print(f"  Ratio wrong:     {ratio_wrong:.6f} (should be ~0.000001)")
    print(f"  Ratio correct:   {ratio_correct:.6f} (should be ~0.1-1.0)")
    
    if 0.01 <= ratio_correct <= 10.0:
        print("✅ KD loss scaling looks reasonable")
        return True
    else:
        print("❌ KD loss scaling may be wrong")
        return False

def test_training_loop():
    """Test a few training steps to ensure no crashes."""
    print("🔍 Testing training loop...")
    
    from models.unet import build_unet, UNetWithEncoder
    from monai.losses import DiceCELoss
    from data.datasets import build_task_roots, get_loaders
    import torch.nn.functional as F
    
    # Get data
    task_roots = build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon"))
    train_loader, val_loader = get_loaders(
        task_roots, "heart", batch_size=1, num_workers=0, cache_rate=0.1)
    
    # Create model
    unet = build_unet(in_channels=1, out_channels=2,
                      channels=(32, 64, 128, 256, 512),
                      strides=(2, 2, 2, 2))
    model = UNetWithEncoder(unet).to(DEVICE)
    
    # Training setup
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Test a few training steps
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for i, batch in enumerate(train_loader):
        if i >= 3:  # Just test 3 batches
            break
            
        imgs = batch["image"].to(DEVICE)
        labels = batch["label"].to(DEVICE)
        if labels.dim() == 4:
            labels = labels.unsqueeze(1)
        labels = labels.long()
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
            print(f"  Batch {i+1}: loss = {loss.item():.4f}")
        else:
            print(f"  Batch {i+1}: ❌ Non-finite loss!")
            return False
    
    if n_batches > 0:
        avg_loss = total_loss / n_batches
        print(f"✅ Training loop works, avg loss = {avg_loss:.4f}")
        return True
    else:
        print("❌ No successful training steps")
        return False

def test_evaluation():
    """Test evaluation pipeline."""
    print("🔍 Testing evaluation...")
    
    from models.unet import build_unet, UNetWithEncoder
    from data.datasets import build_task_roots, get_loaders
    from evaluation.metrics import SegmentationEvaluator
    from monai.inferers import sliding_window_inference
    
    # Get data
    task_roots = build_task_roots(os.environ.get("DATA_ROOT", "/data/decathlon"))
    _, val_loader = get_loaders(
        task_roots, "heart", batch_size=1, num_workers=0, cache_rate=0.1)
    
    # Create model
    unet = build_unet(in_channels=1, out_channels=2,
                      channels=(32, 64, 128, 256, 512),
                      strides=(2, 2, 2, 2))
    model = UNetWithEncoder(unet).to(DEVICE)
    
    # Test evaluation on one batch
    model.eval()
    evaluator = SegmentationEvaluator(num_classes=2)
    
    with torch.inference_mode():
        for i, batch in enumerate(val_loader):
            if i >= 1:  # Just test 1 batch
                break
                
            img = batch["image"].to(DEVICE)
            pred = sliding_window_inference(
                img, (96, 96, 96), sw_batch_size=1, predictor=model, overlap=0.25)
            evaluator.update(pred.cpu(), batch["label"].cpu())
    
    metrics = evaluator.aggregate()
    print(f"✅ Evaluation works: DSC = {metrics['dice']:.4f}, HD95 = {metrics['hd95']:.1f}")
    return True

def main():
    """Run all validation tests."""
    print("🚀 Phase 1 Validation — SSL+KD Pipeline Test")
    print("=" * 50)
    
    set_seed()
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation), 
        ("KD Loss Scaling", test_kd_loss_scaling),
        ("Training Loop", test_training_loop),
        ("Evaluation", test_evaluation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
        print()
    
    # Summary
    print("=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Ready to run full 3-fold CV experiment")
        print("📝 Expected paper outcome: Workshop/conference submission")
    else:
        print("⚠️  SOME TESTS FAILED!")
        print("🔧 Fix issues before running full experiment")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)