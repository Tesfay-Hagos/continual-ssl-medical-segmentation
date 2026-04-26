# SSL + Knowledge Distillation for Few-Shot Medical Image Segmentation

> **Phase 1 Experiment:** SparK-based self-supervised pretraining with knowledge distillation for heart segmentation using only 1 labeled training volume.

## Quick Start on Kaggle

1. Add dataset: `vivekprajapati2048/medical-segmentation-decathlon-heart`
2. Enable GPU (P100 or T4)
3. Add Kaggle secret: `WANDB_API_KEY` (optional)
4. Open `src/notebooks/SSL_KD.py`
5. Run all cells — takes ~12-18 hours total

## Research Question

**Can SSL pretraining + Knowledge Distillation significantly improve segmentation performance when labeled data is extremely scarce?**

With only 1 labeled training volume (out of 20 total), we compare:
- Baseline: Random initialization + DiceCE loss
- SSL only: SparK pretrained encoder + DiceCE loss  
- SSL + KD: SparK pretrained encoder + DiceCE + Knowledge Distillation
- Upper bound: Supervised training on all available labels

## Method Overview

### 1. SparK Self-Supervised Pretraining
- **Input:** All 20 heart volumes (unlabeled)
- **Method:** Sparse masked image modeling (75% masking ratio)
- **Architecture:** U-Net encoder with lightweight reconstruction head
- **Output:** Pretrained encoder weights

### 2. Knowledge Distillation Pipeline
- **Teacher:** SSL-pretrained model fine-tuned on 1 labeled volume
- **Student:** Fresh SSL-pretrained model + KD loss from teacher
- **KD Loss:** Temperature-scaled KL divergence with per-voxel normalization
- **Total Loss:** `DiceCE + α × KL_div(student || teacher)`

### 3. Experimental Design
- **Dataset:** Medical Segmentation Decathlon Task02 (Heart MRI)
- **Validation:** 3-fold cross-validation (13 train / 7 val per fold)
- **Metrics:** Dice Similarity Coefficient (DSC), Hausdorff Distance 95% (HD95)
- **Statistics:** Mean ± std across folds

## Expected Results

Based on similar studies in medical imaging:

| Method | Expected DSC | Rationale |
|--------|--------------|-----------|
| Baseline (random) | 0.65 ± 0.08 | Typical few-shot performance |
| SSL only | 0.72 ± 0.06 | +0.07 from pretraining |
| SSL + KD | 0.76 ± 0.05 | +0.04 from distillation |
| Upper bound | 0.85 ± 0.03 | All labels available |

**Target:** SSL+KD closes ~50% of gap between baseline and upper bound.

## Phase 1 vs Phase 2 Plan

### 🎯 Phase 1 (Current) — Proof of Concept
**Timeline:** 2-3 weeks  
**Goal:** Demonstrate SSL+KD works, get initial results  
**Scope:** Single task (heart), 3-fold CV, core baselines  
**Outcome:** Workshop paper submission ready

**Experiments:**
- [x] Baseline (random init)
- [x] SSL only (SparK pretrained)  
- [x] SSL + KD (teacher-student)
- [x] Supervised upper bound
- [x] 3-fold cross-validation
- [x] Statistical analysis

### 🚀 Phase 2 (Extension) — Full Paper
**Timeline:** +4-6 weeks  
**Goal:** Comprehensive evaluation, conference-quality paper  
**Scope:** Multi-task, more baselines, ablations  
**Outcome:** Main conference submission

**Additional Experiments:**
- [ ] Multi-organ evaluation (liver, pancreas)
- [ ] SSL baseline comparison (SimCLR, random crops)
- [ ] KD hyperparameter ablation (temperature, alpha)
- [ ] Computational efficiency analysis
- [ ] Comparison to recent medical SSL methods
- [ ] Statistical significance testing

## Technical Implementation

### Architecture
```
U-Net Encoder: [32, 64, 128, 256, 512] channels
Decoder: Standard U-Net upsampling path
Input: 1-channel MRI, Output: 2-class segmentation
Patch size: 96³ voxels
```

### Training Configuration
```yaml
SSL Pretraining:
  epochs: 100
  batch_size: 2
  lr: 1e-4
  mask_ratio: 0.75

Fine-tuning:
  epochs: 300
  batch_size: 2  
  lr: 1e-4
  patience: 50
  
Knowledge Distillation:
  alpha: 1.0
  temperature: 2.0
```

### Key Implementation Details
- **Reproducibility:** Fixed seed (42) for all random operations
- **Resume Logic:** Automatic checkpoint recovery on Kaggle interruptions
- **KD Loss Scaling:** Per-voxel normalization to match DiceCE scale
- **Evaluation:** `torch.inference_mode()` for memory efficiency
- **Cross-validation:** Stratified 3-fold split with consistent validation sets

## File Structure

```
src/notebooks/SSL_KD.py          # Main experiment notebook
src/pretraining/
├── pretrain.py                  # SparK pretraining loop
└── spark.py                     # Masked image modeling implementation
src/models/unet.py               # U-Net architecture
src/data/datasets.py             # Medical Decathlon data loaders
src/evaluation/metrics.py        # DSC, HD95 computation
src/continual/lwf.py            # Knowledge distillation utilities
```

## Results Format

The experiment produces publication-ready statistics:

```
Heart segmentation — Task02 (3-fold cross-validation)
======================================================================
Method                         DSC (mean ± std)     HD95 (mean ± std)
----------------------------------------------------------------------
Supervised UB (all labels)     0.XXX ± 0.XXX       XX.X ± XX.X
Baseline (random, 1 label)     0.XXX ± 0.XXX       XX.X ± XX.X  
SSL only (SparK, 1 label)      0.XXX ± 0.XXX       XX.X ± XX.X
SSL + KD (SparK, 1 label)      0.XXX ± 0.XXX       XX.X ± XX.X
======================================================================
NOTE: nnU-Net reports 0.923 DSC on official test server (different split)

SSL gain over baseline   :  +0.XXX DSC
KD  gain over SSL-only   :  +0.XXX DSC  
SSL+KD closes XX.X% of gap to supervised UB
```

## Publication Strategy

### Phase 1 Target Venues
- **MICCAI LABELS Workshop** (Sept 2024) — Few-shot learning focus
- **IEEE ISBI** (May 2025) — Medical imaging methods
- **MICCAI Poster Session** — If results are strong

### Phase 2 Target Venues  
- **MICCAI Main Conference** — With multi-organ results
- **Medical Image Analysis** — Journal submission
- **IEEE TMI** — If computational efficiency story is strong

## Key Contributions

1. **Empirical Validation:** First systematic study of SparK+KD for medical few-shot segmentation
2. **Proper Evaluation:** 3-fold CV with statistical analysis on public dataset
3. **Reproducible Implementation:** Clean, documented code with automatic resume
4. **Practical Impact:** Addresses real clinical scenario (limited labeled data)

## Limitations & Future Work

**Current Limitations:**
- Single organ (heart) evaluation
- Limited baseline comparisons  
- No hyperparameter optimization
- Small dataset (20 volumes total)

**Phase 2 Extensions:**
- Multi-organ generalization study
- Comparison to meta-learning approaches
- Analysis of pretraining data requirements
- Integration with active learning strategies

## Dependencies

```bash
# Core ML stack
torch>=1.12.0
monai[all]>=1.0.0
scikit-learn>=1.0.0

# Data handling  
nibabel>=3.2.0
scipy>=1.7.0
scikit-image>=0.19.0

# Experiment tracking
wandb>=0.13.0
pyyaml>=6.0

# Kaggle environment
kaggle>=1.5.0
```

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{your-paper-2024,
  title={SparK-based Self-Supervised Pretraining with Knowledge Distillation 
         for Few-Shot Medical Image Segmentation},
  author={[Your Names]},
  booktitle={[Conference/Workshop]},
  year={2024}
}
```

## Acknowledgments

- **SparK:** Tian et al., "Designing BERT for Convolutional Networks", ICLR 2023
- **Medical Decathlon:** Simpson et al., Nature Communications 2022  
- **MONAI:** Project MONAI Consortium
- **nnU-Net:** Isensee et al., Nature Methods 2021 (reference baseline)

---

**Contact:** [Your Email]  
**Code:** [GitHub Repository]  
**Data:** [Medical Segmentation Decathlon](http://medicaldecathlon.com/)