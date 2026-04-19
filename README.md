# Continual Self-Supervised Learning for Medical Image Segmentation

> A simplified, centralized framework combining SparK-based masked pretraining with
> EWC, LwF, and Experience Replay for sequential multi-organ segmentation —
> without federated infrastructure.

## Quick start on Kaggle

1. Add [Medical Segmentation Decathlon](http://medicaldecathlon.com/) as a dataset input
2. Open `src/notebooks/kaggle_run.ipynb`
3. Run all cells — clones this repo, installs dependencies, runs all experiments

## Project structure

```
src/
├── data/           MONAI-based loaders for Liver / Pancreas / Heart (Decathlon)
├── models/         U-Net encoder-decoder (MONAI backbone)
├── pretraining/    SparK sparse masked autoencoding for CNN pretraining
├── continual/      EWC, LwF, Experience Replay regularizers
├── evaluation/     DSC, HD95, BWT, FWT, Forgetting Measure
├── configs/        YAML configs for each experiment
├── scripts/        CLI entry points
└── notebooks/      Kaggle starter notebook
docs/               LaTeX paper (introduction + related work + methodology)
```

## Reference papers

| Paper | DOI | Citations |
|---|---|---|
| FedCSL (Zhang 2025) | 10.1109/TNNLS.2024.3469962 | 13 |
| SparK (Tian 2023) | arXiv:2301.03580 | 145 |
| EWC (Kirkpatrick 2017) | 10.1073/pnas.1611835114 | 9,523 |
| LwF (Li 2016) | 10.1007/978-3-319-46493-0_37 | 5,411 |
| U-Net (Ronneberger 2015) | 10.1007/978-3-319-24574-4_28 | 93,789 |
| nnU-Net (Isensee 2021) | 10.1038/s41592-020-01008-z | 8,295 |
| Medical Decathlon (Simpson 2022) | 10.1038/s41467-022-30695-9 | — |

## Project Overview

This project implements a continual self-supervised learning framework that combines:
- **Self-supervised pre-training** using masked image modeling (MIM)
- **Continual learning** strategies to prevent catastrophic forgetting
- **Knowledge distillation** for efficient deployment (optional)

The framework is designed for single-institution medical image segmentation scenarios where labeled data is scarce and models need to learn multiple organ segmentation tasks sequentially.

## Project Structure

```
continual_self_supervised_learning/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation and LaTeX files
│   └── introduction.tex         # Project introduction and planning document
├── src/                         # Source code (to be created)
│   ├── data/                    # Data loading and preprocessing
│   ├── models/                  # Model architectures
│   ├── ssl/                     # Self-supervised learning methods
│   ├── continual/               # Continual learning strategies
│   ├── distillation/            # Knowledge distillation
│   └── utils/                   # Utility functions
├── configs/                     # Configuration files (to be created)
├── experiments/                 # Experiment scripts (to be created)
└── results/                     # Experimental results (to be created)
```

## Key Features

### 1. Self-Supervised Pre-training
- Masked image modeling (MIM) for learning robust representations
- Contrastive learning (optional)
- Pre-training on unlabeled medical images

### 2. Continual Learning
- Multiple strategies: EWC, LwF, Experience Replay, Progressive Networks
- Prevention of catastrophic forgetting
- Sequential training on multiple organ segmentation tasks

### 3. Knowledge Distillation
- Teacher-student framework
- Lightweight models for deployment
- Efficiency-accuracy trade-off analysis

## Datasets

The project uses publicly available medical image segmentation datasets:

1. **Medical Segmentation Decathlon** - 10 organ segmentation tasks
2. **ACDC** - Cardiac MRI segmentation
3. **Synapse** - Multi-organ abdominal CT segmentation
4. **BraTS** - Brain tumor segmentation

## Methodologies (APAI Requirements)

This project combines three APAI-approved methodologies:

1. ✅ **Self-Supervised Learning** - Masked image modeling for pre-training
2. ✅ **Continual Learning** - Sequential task learning without forgetting
3. ✅ **Knowledge Distillation** - Efficient model compression (optional)

## Resources

### GitHub Repositories

- **SSL4MIS**: https://github.com/HiLab-git/SSL4MIS (2.4k stars)
  - Self-supervised learning methods for medical imaging
  
- **Avalanche**: https://github.com/ContinualAI/avalanche (1.7k stars)
  - Continual learning framework
  
- **MONAI**: https://github.com/Project-MONAI/MONAI (5.8k stars)
  - Medical imaging deep learning framework

### Key Papers

1. **FedCSL** (Primary Reference):
   - Fan Zhang et al., "Federated Cross-Incremental Self-Supervised Learning for Medical Image Segmentation"
   - IEEE TNNLS 2024
   - DOI: 10.1109/TNNLS.2024.3469962

2. **Masked Autoencoders**:
   - He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022

3. **Elastic Weight Consolidation**:
   - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks", PNAS 2017

4. **Learning Without Forgetting**:
   - Li and Hoiem, "Learning without Forgetting", ECCV 2016

## Implementation Plan

### Phase 1: Setup and Baseline (Weeks 1-2)
- Download and preprocess datasets
- Implement data loading pipelines
- Train supervised baselines

### Phase 2: Self-Supervised Pre-training (Weeks 3-4)
- Implement masked image modeling
- Pre-train encoder on unlabeled data
- Evaluate learned representations

### Phase 3: Continual Learning (Weeks 5-7)
- Implement continual learning strategies
- Train sequentially on multiple tasks
- Measure catastrophic forgetting

### Phase 4: Knowledge Distillation (Week 8)
- Train teacher model
- Distill to student model
- Evaluate efficiency-accuracy trade-off

### Phase 5: Evaluation and Analysis (Weeks 9-10)
- Comprehensive experiments
- Statistical analysis
- Ablation studies

### Phase 6: Paper Writing (Weeks 11-12)
- Write all sections
- Create figures and tables
- Prepare code repository

## Getting Started

### Prerequisites

```bash
# Python 3.8+
# PyTorch 1.12+
# CUDA 11.3+ (for GPU support)
```

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd continual_self_supervised_learning

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Download datasets (to be implemented)
python scripts/download_datasets.py

# 2. Pre-train with self-supervised learning (to be implemented)
python experiments/pretrain.py --config configs/pretrain_mim.yaml

# 3. Train with continual learning (to be implemented)
python experiments/continual_train.py --config configs/continual_ewc.yaml

# 4. Evaluate (to be implemented)
python experiments/evaluate.py --checkpoint results/model_best.pth
```

## Evaluation Metrics

### Segmentation Performance
- Dice Similarity Coefficient (DSC)
- Hausdorff Distance (HD95)
- Intersection over Union (IoU)

### Continual Learning Metrics
- Average Accuracy
- Backward Transfer (BWT)
- Forward Transfer (FWT)
- Forgetting Measure

## Expected Contributions

1. **Simplified Framework**: Remove federated learning complexity while maintaining benefits
2. **Comprehensive Evaluation**: Systematic comparison across multiple organs and strategies
3. **Efficient Deployment**: Knowledge distillation for lightweight models
4. **Open-Source Implementation**: Reproducible code with clear documentation

## Comparison with FedCSL

| Component | FedCSL | Our Approach |
|-----------|--------|--------------|
| Federated Learning | ✅ | ❌ |
| Self-Supervised Pre-training | ✅ | ✅ |
| Continual Learning | ✅ | ✅ |
| Knowledge Distillation | ✅ | ✅ (optional) |
| Multi-client Training | ✅ | ❌ |
| Secure Computation | ✅ | ❌ |
| **Complexity** | Very High | Medium |
| **Implementation Time** | 3-4 months | 2-2.5 months |

## Documentation

- **Introduction Document**: `docs/introduction.tex` - Comprehensive project introduction with methodologies, datasets, and implementation plan
- **Paper Search Analysis**: `../title_selection/` - Analysis of related papers and selection process

## Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-2 | Setup & Baseline | Data ready, baselines trained |
| 3-4 | Self-Supervised | Pre-trained encoder |
| 5-7 | Continual Learning | CL models trained |
| 8 | Knowledge Distillation | Student models |
| 9-10 | Evaluation | All experiments done |
| 11-12 | Paper Writing | Final paper |

## Team

- [Team Member 1] - [Responsibilities]
- [Team Member 2] - [Responsibilities]
- [Team Member 3] - [Responsibilities]

## License

[To be determined]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{[your-citation-key],
  title={Continual Self-Supervised Learning for Medical Image Segmentation: A Simplified Framework},
  author={[Your Names]},
  journal={[Conference/Journal]},
  year={2024}
}
```

## Acknowledgments

- Original FedCSL paper by Fan Zhang et al.
- SSL4MIS repository by HiLab-git
- Avalanche continual learning library by ContinualAI
- MONAI framework by Project MONAI

## Contact

For questions or issues, please contact:
- [Your Email]
- [Your GitHub]

## References

See `docs/introduction.tex` for complete list of references.
