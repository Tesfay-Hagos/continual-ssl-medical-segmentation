# 📊 Publication Figure Generation Guide

## Generated Figures

The SSL_KD.py experiment automatically generates **publication-ready figures** in both PNG (300 DPI) and PDF (vector) formats:

### 📈 Figure 1: Method Comparison (`method_comparison.png`)
- **Type:** Dual bar chart with error bars
- **Content:** DSC and HD95 performance across all 4 methods
- **Features:** 
  - Mean ± std values displayed on bars
  - Color-coded methods (baseline=red, SSL=green, SSL+KD=blue, upper bound=purple)
  - Professional styling with grid lines
- **Usage:** Main results figure for paper

### 📦 Figure 2: Cross-Validation Analysis (`cv_boxplot.png`)
- **Type:** Box plot with individual data points
- **Content:** Individual fold results for statistical transparency
- **Features:**
  - Shows distribution across 3 folds
  - Individual points overlaid on boxes
  - Demonstrates statistical rigor
- **Usage:** Supplementary material or methods validation

### 📊 Figure 3: Improvement Breakdown (`improvement_analysis.png`)
- **Type:** Bar chart showing gains
- **Content:** SSL gain, KD gain, and total improvement
- **Features:**
  - Quantifies individual contributions
  - Positive/negative change visualization
  - Clear contribution attribution
- **Usage:** Discussion section, ablation analysis

## Figure Quality Standards

### Resolution & Format
- **PNG:** 300 DPI (publication quality)
- **PDF:** Vector format (scalable, journal preferred)
- **Size:** Optimized for 2-column journal layout

### Styling
- **Font sizes:** 12pt body, 14pt titles (readable in print)
- **Colors:** Colorblind-friendly palette
- **Grid:** Subtle background grid for readability
- **Labels:** Bold value annotations on bars

## File Locations

```
/kaggle/working/checkpoints/figures/  (Kaggle)
/tmp/ssl_kd_ckpts/figures/           (Local)
├── method_comparison.png
├── method_comparison.pdf
├── cv_boxplot.png
├── cv_boxplot.pdf
├── improvement_analysis.png
├── improvement_analysis.pdf
└── figure_summary.json
```

## LaTeX Integration

### Figure 1 (Main Results)
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/method_comparison.pdf}
\caption{Heart segmentation performance comparison. (Left) Dice Similarity Coefficient showing SSL pretraining improves baseline by X.XXX and knowledge distillation provides additional X.XXX gain. (Right) Hausdorff Distance 95\% demonstrating improved boundary accuracy. Error bars show standard deviation across 3-fold cross-validation.}
\label{fig:main_results}
\end{figure}
```

### Figure 2 (Cross-Validation)
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.8\textwidth]{figures/cv_boxplot.pdf}
\caption{Cross-validation results showing individual fold performance. Box plots display median, quartiles, and outliers across 3 folds, with individual data points overlaid. Consistent improvement of SSL+KD across all folds demonstrates robustness.}
\label{fig:cv_analysis}
\end{figure}
```

### Figure 3 (Ablation)
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.7\textwidth]{figures/improvement_analysis.pdf}
\caption{Contribution analysis of SSL pretraining and knowledge distillation. SSL pretraining provides the majority of improvement (+X.XXX DSC), while KD adds meaningful additional gain (+X.XXX DSC), resulting in total improvement of +X.XXX DSC over random initialization.}
\label{fig:ablation}
\end{figure}
```

## Expected Results Preview

Based on medical SSL literature, figures should show:

### Method Comparison
- **Baseline:** ~0.65 ± 0.08 DSC
- **SSL Only:** ~0.72 ± 0.06 DSC  
- **SSL + KD:** ~0.76 ± 0.05 DSC
- **Upper Bound:** ~0.85 ± 0.03 DSC

### Key Insights Visualized
1. **SSL Impact:** Clear jump from baseline to SSL-only
2. **KD Benefit:** Additional improvement from SSL+KD
3. **Statistical Significance:** Error bars show non-overlapping confidence
4. **Gap Analysis:** SSL+KD closes ~50% of gap to upper bound

## Troubleshooting

### Missing Figures
If figures aren't generated:
```python
# Check if matplotlib backend is available
import matplotlib
print(matplotlib.get_backend())

# Force non-interactive backend on Kaggle
matplotlib.use('Agg')
```

### Style Issues
```python
# Fallback if seaborn style fails
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
```

### Memory Issues
```python
# Clear figures after saving
plt.close('all')
import gc
gc.collect()
```

## Paper Integration Checklist

- [ ] Figures saved in both PNG and PDF formats
- [ ] Figure captions written with specific numerical results
- [ ] Statistical significance mentioned in captions
- [ ] Figures referenced in results section
- [ ] Color scheme consistent across all figures
- [ ] Font sizes readable when printed
- [ ] Vector formats used for journal submission

## WandB Integration

Figures are automatically uploaded to WandB artifacts:
- `figure-method_comparison`
- `figure-cv_boxplot` 
- `figure-improvement_analysis`

Access via WandB dashboard for sharing and collaboration.

---

**Note:** All figures are generated automatically at the end of the SSL_KD.py experiment. No manual intervention required.