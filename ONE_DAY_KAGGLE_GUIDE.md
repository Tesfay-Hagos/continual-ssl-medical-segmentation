# 🚀 One-Day Kaggle Run Guide

## ⏰ **Time Budget (12-16 hours total)**

| Component | Time | Notes |
|-----------|------|-------|
| Environment setup | 10 min | Git clone, pip install |
| SparK pretraining | 2-3 hours | 100 epochs on all volumes |
| 3-fold CV experiments | 9-12 hours | 4 experiments × 3 folds × 50 epochs |
| Results analysis | 30 min | Statistics, plots |

## 🎯 **Optimizations Applied**

### Training Speed
- **Epochs:** 300 → 50 (6x faster)
- **Patience:** 50 → 20 (earlier stopping)
- **Learning rate:** 1e-4 → 2e-4 (faster convergence)
- **Warmup:** 10 → 5 epochs

### Expected Performance Impact
- **DSC drop:** ~0.01-0.02 (minimal with early stopping)
- **Still publishable:** 50 epochs sufficient for few-shot learning
- **Statistical validity:** 3-fold CV maintains rigor

## 📋 **Kaggle Setup Checklist**

### Before Starting
- [ ] Add dataset: `vivekprajapati2048/medical-segmentation-decathlon-heart`
- [ ] Enable GPU (P100 or T4 x1)
- [ ] Set accelerator to GPU
- [ ] Add Kaggle secret: `WANDB_API_KEY` (optional)
- [ ] Verify 30 hours GPU quota available

### Notebook Settings
```python
# In first cell, verify:
print("GPU:", torch.cuda.get_device_name(0))
print("Memory:", torch.cuda.get_device_properties(0).total_memory // 1e9, "GB")
```

## ⚡ **Runtime Estimates**

### Per Experiment (50 epochs)
- **Baseline:** ~45 min (random init, fast)
- **SSL only:** ~60 min (pretrained init)  
- **SSL + KD:** ~75 min (teacher inference overhead)
- **Upper bound:** ~90 min (more training data)

### Total Timeline
```
Hour 0:   Start notebook
Hour 0.5: Pretraining begins
Hour 3:   Fold 1 experiments start
Hour 6:   Fold 2 experiments start  
Hour 9:   Fold 3 experiments start
Hour 12:  Results analysis
Hour 13:  Complete ✅
```

## 🔧 **If Running Behind Schedule**

### Emergency Optimizations (if needed)
1. **Reduce to 2-fold CV:** Change `N_FOLDS = 2`
2. **Skip upper bound:** Comment out Experiment D
3. **Reduce epochs further:** Set `epochs = 30`

### Critical vs Optional
- **Critical:** Baseline, SSL only, SSL+KD (core comparison)
- **Optional:** Upper bound (nice to have)
- **Critical:** At least 2 folds (statistical validity)

## 📊 **Expected Results (50 epochs)**

Based on medical SSL literature with reduced training:

| Method | Expected DSC | 95% CI |
|--------|--------------|--------|
| Baseline | 0.64 ± 0.05 | [0.59, 0.69] |
| SSL only | 0.70 ± 0.04 | [0.66, 0.74] |
| SSL + KD | 0.73 ± 0.04 | [0.69, 0.77] |
| Upper bound | 0.82 ± 0.03 | [0.79, 0.85] |

**Key Metrics:**
- SSL gain: +0.06 DSC (publishable)
- KD gain: +0.03 DSC (meaningful)
- Gap closure: ~35% (reasonable)

## 🚨 **Monitoring During Run**

### Check Every 2 Hours
```python
# Monitor progress
print(f"Current fold: {fold+1}/3")
print(f"Completed experiments: {list(cv_results.keys())}")
print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

### Warning Signs
- **Loss not decreasing:** Check learning rate
- **Memory errors:** Reduce batch size to 1
- **Very slow progress:** Verify GPU is active
- **NaN losses:** Check data preprocessing

## 📝 **Results Validation**

### Minimum Success Criteria (50 epochs)
- SSL gain ≥ +0.04 DSC
- KD gain ≥ +0.015 DSC  
- All experiments complete
- CV std ≤ 0.06 (higher tolerance for shorter training)

### If Results Are Weak
- **Still publishable** as "preliminary results"
- **Workshop paper** focus on methodology
- **Future work** mentions longer training

## 🎯 **Paper Positioning (50 epochs)**

### Title Options
1. "SparK-based SSL with Knowledge Distillation for Few-Shot Medical Segmentation"
2. "Efficient Self-Supervised Pretraining for Data-Scarce Medical Image Segmentation"

### Key Claims
1. **SSL helps** in extreme few-shot scenarios (1 labeled volume)
2. **KD provides additional benefit** over SSL alone
3. **Efficient training** achieves good results in limited time
4. **Reproducible framework** with public dataset

### Venue Strategy
- **Primary:** MICCAI LABELS Workshop (few-shot focus)
- **Secondary:** IEEE ISBI (medical imaging methods)
- **Backup:** MICCAI poster session

## 🔄 **Backup Plan**

### If Kaggle Times Out
1. **Save intermediate results:** JSON files auto-saved
2. **Resume capability:** Built-in checkpoint system
3. **Partial results:** Even 2 folds publishable as preliminary

### If Results Are Disappointing
1. **Focus on SSL only:** Remove KD, emphasize pretraining
2. **Methodology paper:** Emphasize reproducible framework
3. **Negative results:** "When does SSL not help?" angle

## ✅ **Success Definition**

**Minimum viable paper:** 
- 3-fold CV complete
- SSL shows positive gain
- Statistical analysis done
- Reproducible code available

**Stretch goals:**
- All 4 experiments complete
- Strong statistical significance  
- Clear KD benefit demonstrated
- Ready for conference submission

---

**Start Time:** Record when you begin
**Expected Completion:** 12-16 hours later
**Backup Plan:** Activated if >75% through GPU quota