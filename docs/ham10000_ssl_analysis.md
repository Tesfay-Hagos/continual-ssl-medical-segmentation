# Semi-Supervised Skin Lesion Classification via SSL Pretraining and Mean Teacher
## Research Analysis — HAM10000 Experiment

---

## 1. Research Question

> **Can domain-specific self-supervised pretraining combined with Mean Teacher
> semi-supervised fine-tuning close the gap to fully-supervised performance
> when only 1–20 % of skin lesion images are labeled?**

Secondary questions:
- At what label fraction does SSL pretraining provide the largest relative gain?
- Does Mean Teacher consistently add value on top of SSL pretraining, or only at
  specific label scarcity levels?
- Are the improvements visible in gradient saliency maps (model attention on
  clinically relevant lesion features)?

---

## 2. Dataset: HAM10000

**Citation:** Tschandl P. et al., *The HAM10000 dataset, a large collection of
multi-source dermatoscopic images of common pigmented skin lesions*, Scientific
Data, 2018. https://doi.org/10.1038/sdata.2018.161

| Property | Value |
|---|---|
| Full name | Human Against Machine with 10000 training images |
| Total images | 10,015 RGB dermoscopy photographs |
| Resolution | 450 × 600 px (JPEG) |
| Classes | 7 (see table below) |
| Label confirmation | >50 % confirmed by pathology or follow-up |
| Kaggle slug | `kmader/skin-cancer-mnist-ham10000` |

### Class distribution

| Code | Condition | Count | % |
|---|---|---|---|
| nv | Melanocytic nevi | 6,705 | 66.9 % |
| mel | Melanoma | 1,113 | 11.1 % |
| bkl | Benign keratosis-like lesions | 1,099 | 11.0 % |
| bcc | Basal cell carcinoma | 514 | 5.1 % |
| akiec | Actinic keratoses / intraepithelial carcinoma | 327 | 3.3 % |
| vasc | Vascular lesions | 142 | 1.4 % |
| df | Dermatofibroma | 115 | 1.1 % |

**Key challenge:** Severe class imbalance (nv = 66.9 % vs df = 1.1 %).
Primary metric is **balanced accuracy** (macro-averaged per-class accuracy),
not overall accuracy, so that minority classes contribute equally.

### Experimental label splits

| Label fraction | # Labeled | # Unlabeled | Regime |
|---|---|---|---|
| 1 % | ~67 | ~5,945 | Extreme few-shot |
| 5 % | ~335 | ~5,677 | Very low-label |
| 10 % | ~669 | ~5,343 | Low-label (main experiment) |
| 20 % | ~1,338 | ~4,674 | Semi-supervised |
| 100 % | ~6,677 | 0 | Supervised upper bound |

All splits are **stratified** (each class proportionally represented in the
labeled subset). 3-fold cross-validation on the full 10,015 images.

---

## 3. Methodology

### 3.1 Backbone: EfficientNet-B3

- **Source:** `torchvision.models.efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)`
- **Feature dimension:** 1,536
- **Classifier head:** replaced with `Dropout(0.3) → Linear(1536, 7)`
- **Reason:** Strong ImageNet pretrained features as starting point; B3 balances
  accuracy and memory on Kaggle T4 (16 GB).

### 3.2 Stage 1 — SimCLR Self-Supervised Pretraining

SimCLR (Chen et al., ICML 2020) learns visual representations without labels
by maximising agreement between two augmented views of the same image.

**Augmentation pipeline (SimCLR-style):**
```
RandomResizedCrop(224, scale=(0.2, 1.0))
RandomHorizontalFlip(p=0.5)
ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)  p=0.8
RandomGrayscale(p=0.2)
GaussianBlur(kernel_size=23)  p=0.5
ToTensor + Normalize(ImageNet mean/std)
```

**Loss:** NT-Xent (normalised temperature-scaled cross-entropy), temperature τ = 0.5

**Why domain-specific pretraining matters:**
- ImageNet features (dogs, cars, furniture) differ from dermoscopy features
  (lesion borders, colour asymmetry, texture patterns).
- Pretraining on the *same unlabeled images* that will later be fine-tuned on
  forces the encoder to learn skin-specific features.
- Prior work (Barlow Twins on HAM10000, arXiv:2401.00692) confirms SSL
  pretraining outperforms ImageNet transfer at low label fractions.

### 3.3 Stage 2 — Mean Teacher Semi-Supervised Fine-tuning

Mean Teacher (Tarvainen & Valpola, NeurIPS 2017) maintains two networks:
- **Student**: updated by gradient descent on labeled + unlabeled loss.
- **Teacher**: exponential moving average (EMA) of student weights — always
  slightly ahead of the student, producing better soft predictions.

```
θ_teacher ← α · θ_teacher + (1−α) · θ_student      α = 0.999

Loss = L_supervised(labeled) + λ · L_consistency(unlabeled)

L_supervised    = CrossEntropy(student(x_l), y_l)
L_consistency   = MSE(student(x_u + strong_aug),
                       teacher(x_u + weak_aug).detach())
```

**Why Mean Teacher beats KD here:**
| | KD (previous approach) | Mean Teacher |
|---|---|---|
| Teacher quality | Trained on 1 label → weak | EMA of student → always better |
| Labeled data required for teacher | Yes (separate training run) | No |
| Stability | High variance (seen in results) | Consistent gains |
| Unlabeled data use | Indirect (soft labels) | Direct (consistency loss) |

### 3.4 Experiment Matrix

Four methods × four label fractions × 3-fold CV = 48 runs.

| Method | Backbone init | Training data | Loss |
|---|---|---|---|
| Baseline | ImageNet | Labeled only | CrossEntropy |
| SSL-only | SimCLR pretrained | Labeled only | CrossEntropy |
| SSL + Mean Teacher | SimCLR pretrained | Labeled + Unlabeled | CE + Consistency |
| Upper bound | ImageNet | All labeled | CrossEntropy |

---

## 4. Visualizations (Section 8 of notebook)

All figures are saved as PNG/PDF and uploaded to WandB as artifacts.

| Figure | Tool | What it shows |
|---|---|---|
| Class distribution bar chart | matplotlib | Dataset imbalance |
| Sample image grid per class | `torchvision.utils.make_grid` | Visual data quality |
| Training curves (loss, balanced acc) | WandB native | Learning dynamics |
| Confusion matrix | seaborn heatmap | Per-class errors |
| GradCAM overlays | manual hooks + matplotlib | What drives predictions |
| t-SNE of learned features | sklearn TSNE + matplotlib | Feature space quality |
| FlashTorch saliency maps | flashtorch.saliency | Pixel-level attribution |
| Method comparison bar chart | matplotlib | Main result figure |
| Label fraction ablation curve | matplotlib | SSL gain vs. label scarcity |

---

## 5. Related Work and Positioning

### 5.1 Must-cite papers

| # | Reference | Year | Venue | Relation to your work |
|---|---|---|---|---|
| 1 | Tschandl et al. — HAM10000 dataset | 2018 | Scientific Data | Dataset |
| 2 | Chen et al. — SimCLR | 2020 | ICML | Your SSL method |
| 3 | Tarvainen & Valpola — Mean Teacher | 2017 | NeurIPS | Your semi-supervised method |
| 4 | Tan & Le — EfficientNet | 2019 | ICML | Your backbone |
| 5 | Sohn et al. — FixMatch | 2020 | NeurIPS | Semi-supervised baseline |

### 5.2 Direct competition — papers you must beat or match

| Paper | Method | Metric | Result |
|---|---|---|---|
| FixMatch-LS (Biomedical Signal Processing 2023) | FixMatch + label smoothing | AUC | 91.63–95.44 % |
| ReFixMatch-LS (Med&Bio Eng 2022) | Pseudo-label reuse | AUC | Improves FixMatch-LS |
| Barlow Twins SSL (arXiv 2401.00692, 2024) | SSL pretraining only | Accuracy | Beats supervised TL at low labels |
| Online KD (arXiv 2508.11511, 2024) | Ensemble KD semi-supervised | Accuracy | Comparable to FixMatch |

### 5.3 Your differentiation

No existing paper combines **domain-specific SimCLR pretraining** with **Mean Teacher**
on HAM10000 and evaluates systematically across label fractions (1 %–20 %).

- FixMatch-LS does not use SSL pretraining as initialisation.
- Barlow Twins uses SSL but no semi-supervised fine-tuning step.
- Online KD does not use SSL pretraining.

Your contribution: **the combination** + **systematic label-fraction ablation** +
**gradient visualisation confirming clinical relevance**.

### 5.4 Additional references

| Paper | Year | Venue | Cite for |
|---|---|---|---|
| Self-supervised curricular learning for skin lesions (arXiv 2112.12086) | 2021 | arXiv | SSL curriculum comparison |
| ECL contrastive learning for long-tailed skin lesion (MICCAI 2023) | 2023 | MICCAI | Class imbalance handling |
| DermViT (PMC 2025) | 2025 | PMC | ViT baseline comparison |
| CSSL-ETL ensemble (PubMed 2025) | 2025 | PubMed | Upper bound comparison |
| SSL pretraining survey for medical imaging (BMC 2024) | 2024 | BMC Med Imaging | SSL background |
| Comparative supervised vs SSL on imbalanced medical data (Sci Rep 2025) | 2025 | Scientific Reports | Validation of SSL benefit |

---

## 6. Expected Results

### 6.1 Main result table (10 % labeled, 3-fold CV)

| Method | Balanced Acc | AUC (macro) |
|---|---|---|
| Baseline (ImageNet, 10 % labels) | ~65–68 % | ~0.88–0.90 |
| SSL-only (SimCLR, 10 % labels) | ~72–76 % | ~0.91–0.93 |
| SSL + Mean Teacher (10 % labels) | **~80–84 %** | **~0.94–0.96** |
| Fully supervised upper bound | ~87–89 % | ~0.97–0.98 |

### 6.2 Label fraction ablation (SSL + Mean Teacher)

| Label % | Expected balanced acc | Gap to upper bound |
|---|---|---|
| 1 % | ~62–66 % | Large |
| 5 % | ~72–76 % | Moderate |
| 10 % | ~80–84 % | Small |
| 20 % | ~84–87 % | Very small |

**Key narrative:** At 5 % labels, SSL + Mean Teacher closes ~75 % of the gap to
full supervision. This is the strongest selling point of the paper.

---

## 7. Target Venues

| Venue | Type | Deadline (approx) | Fit |
|---|---|---|---|
| **ISBI 2026** | IEEE conference | Jan 2026 | ★★★ Exact fit |
| **MIDL 2026** | Conference | Feb 2026 | ★★★ Exact fit |
| MICCAI 2026 workshops | Workshop | Apr 2026 | ★★ Good fit |
| IEEE JBHI | Journal | Rolling | ★★ Good fit |

**Recommended target: ISBI 2026** (4-page short paper format, ideal scope).

---

## 8. Paper Outline

```
Title: Semi-Supervised Skin Lesion Classification via Domain-Specific
       Self-Supervised Pretraining and Mean Teacher

Abstract (150 words)

1. Introduction
   - Skin cancer detection, annotation cost, semi-supervised motivation
   - Contribution summary: SimCLR + Mean Teacher + HAM10000 + label ablation

2. Related Work
   - SSL for medical imaging
   - Semi-supervised classification
   - Skin lesion analysis

3. Methodology
   3.1 SimCLR pretraining (domain-specific)
   3.2 Mean Teacher fine-tuning
   3.3 Implementation details

4. Experiments
   4.1 Dataset and evaluation protocol
   4.2 Main results (10% labeled)
   4.3 Label fraction ablation

5. Visualizations and Interpretation
   5.1 GradCAM analysis
   5.2 t-SNE feature space
   5.3 Confusion matrix analysis

6. Conclusion

References (~20 citations)
```

---

## 9. Implementation Notes

- **Kaggle dataset:** `kmader/skin-cancer-mnist-ham10000`
- **WandB project:** `ham10000-ssl`
- **WandB secret name:** `HAM10000_WANDB`
- **Code location:** `src/2d-notebook/ham10000.py`
- **Reused modules:** `src/utils/storage.py` (checkpoint save/restore)
- **New modules (self-contained in ham10000.py):**
  - `HAM10000Dataset` — PIL-based loader with stratified split helpers
  - `SimCLRPairDataset` — dual-view augmentation for SSL
  - `SimCLRModel` — EfficientNet backbone + projection head
  - `MeanTeacherTrainer` — EMA update + consistency loss
  - `GradCAM` — hook-based gradient class activation maps
  - All visualisation functions

---

*Document version: 1.0 — created 2026-05-07*
