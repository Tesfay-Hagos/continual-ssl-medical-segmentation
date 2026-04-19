# Quick Reference Guide

## 🚀 Three Commands to Get Started

```bash
# 1. Install (one-time setup)
pip install requests

# 2. Search for papers (takes ~15 minutes)
python example_searches.py

# 3. Analyze results
python analyze_papers.py search_results/self_supervised_medical_seg.json --top 10
```

## 📁 What You Have

```
Your Toolkit:
├── 🔍 paper_search_tool.py    → Search arXiv, Semantic Scholar, PubMed
├── 📊 analyze_papers.py        → Rank papers by improvement potential
├── 🎯 example_searches.py      → Run 8 pre-configured searches
├── 📚 README.md                → Complete documentation
├── 🎓 GETTING_STARTED.md       → Step-by-step tutorial
├── 📋 PROJECT_SUMMARY.md       → Overview and workflow
└── ⚡ QUICK_REFERENCE.md       → This file
```

## 🎯 Common Commands

### Search for Papers

```bash
# Medical + Self-Supervised
python paper_search_tool.py \
  --keywords "medical segmentation" "self-supervised" \
  --year-start 2023 --year-end 2024

# Medical + Knowledge Distillation
python paper_search_tool.py \
  --keywords "medical imaging" "knowledge distillation" \
  --year-start 2022 --year-end 2024

# Medical + Meta-Learning
python paper_search_tool.py \
  --keywords "meta learning" "few-shot" "medical" \
  --year-start 2023 --year-end 2024
```

### Analyze Results

```bash
# Show top 10 papers
python analyze_papers.py papers_results.json --top 10

# Generate detailed report
python analyze_papers.py papers_results.json --report analysis.txt

# Show papers by methodology
python analyze_papers.py papers_results.json --by-methodology
```

### Run Multiple Searches

```bash
# Python version (recommended)
python example_searches.py

# Bash version (Linux/Mac)
bash example_searches.sh
```

### Verify DOIs

```bash
# Verify all DOIs in results
python verify_dois.py papers_results.json

# Verify top 10 papers only
python verify_dois.py papers_results.json --top 10

# Generate verification report
python verify_dois.py papers_results.json --report doi_report.txt
```

## 📊 Understanding the Output

### Search Results (CSV columns)
- **Title**: Paper title
- **Authors**: Paper authors
- **Year**: Publication year
- **Venue**: Conference/Journal
- **Citations**: Citation count
- **Source**: arXiv/Semantic Scholar/PubMed
- **Keywords Found**: Which of your keywords matched
- **URL**: Link to paper
- **Abstract**: Paper abstract

### Analysis Scores
- **Overall Score**: Combined score (higher = better)
  - Future Work Score (40%)
  - Recency (30%)
  - Citations (20%)
  - Resources (10%)
- **Future Work Indicators**: Count of improvement keywords
- **Methodologies**: Detected methodologies
- **Resources Available**: Code/dataset availability

## 🎓 APAI Requirements Checklist

### Methodology Must Be One Of:
- [ ] Self-supervised learning
- [ ] Knowledge distillation
- [ ] Continual learning
- [ ] Meta-learning
- [ ] Vision-Language Models (CLIP, BLIP, etc.)

### Paper Must Include:
- [ ] Abstract
- [ ] Introduction
- [ ] Method (with architecture diagram)
- [ ] Results (with baselines)
- [ ] Conclusion
- [ ] Contributions section
- [ ] 6-8 pages (excluding references)
- [ ] Code URL in abstract

### Before Starting:
- [ ] Teacher approved your choice
- [ ] Dataset is accessible
- [ ] Code available or implementable
- [ ] Computational resources sufficient

## 🔍 What to Look For in Papers

### ✅ Good Signs
- "Future work" section
- "Limitations" mentioned
- "Can be extended to..."
- "Further research needed"
- Code available on GitHub
- Public datasets used
- Recent (2023-2024)
- 10-50 citations

### ❌ Red Flags
- No limitations mentioned
- Claims "perfect" results
- Proprietary data only
- No code available
- Too simple (basic CNN)
- Too complex (can't implement)
- Very old (pre-2020)

## 📈 Scoring Guide

### Future Work Score
- **0-1**: Avoid (no improvement opportunities)
- **2-3**: Consider (some opportunities)
- **4+**: Excellent (many opportunities)

### Citation Count (for 2023-2024 papers)
- **0-5**: Very new (risky but cutting-edge)
- **5-20**: Good (validated but not saturated)
- **20-50**: Popular (well-validated)
- **50+**: Mature (might be saturated)

### Year
- **2024**: Cutting-edge (best choice)
- **2023**: Recent (good choice)
- **2022**: Acceptable
- **2021 or older**: Avoid (too mature)

## 🎯 Example Improvement Strategies

### 1. Cross-Domain Transfer
**Original**: Brain MRI segmentation
**Your Work**: Apply to lung CT scans
**Contribution**: Demonstrate generalizability

### 2. Methodology Combination
**Original**: Self-supervised learning
**Your Work**: Add knowledge distillation
**Contribution**: Efficient lightweight model

### 3. Limitation Addressing
**Original**: Requires many labeled examples
**Your Work**: Add few-shot meta-learning
**Contribution**: Reduce annotation requirements

### 4. Future Work Implementation
**Original**: "Future work: multi-task learning"
**Your Work**: Implement multi-task version
**Contribution**: Validate proposed direction

### 5. Comprehensive Analysis
**Original**: Limited ablation studies
**Your Work**: Extensive ablation analysis
**Contribution**: Deeper understanding

## 🐛 Troubleshooting

### No papers found?
```bash
# Try broader keywords
python paper_search_tool.py --keywords "medical" "deep learning" --year-start 2020 --year-end 2024
```

### Too many papers?
```bash
# Add more specific keywords
python paper_search_tool.py --keywords "medical" "segmentation" "self-supervised" "transformer" --year-start 2023 --year-end 2024
```

### Low analysis scores?
```bash
# Try different search (papers might be too mature)
python paper_search_tool.py --keywords "medical" "self-supervised" --year-start 2024 --year-end 2024
```

### Can't access paper?
- Check university library
- Try Sci-Hub (if legal)
- Email authors
- Check arXiv version

## 📞 Getting Help

### For Tool Issues:
1. Check README.md
2. Check GETTING_STARTED.md
3. Run with `--help` flag

### For Paper Selection:
1. Review PROJECT_SUMMARY.md
2. Discuss with team
3. Consult teacher

### For APAI Requirements:
1. Read "APAI Exam Instructions.pdf"
2. Check CVPR proceedings examples
3. Ask teacher

## ⏱️ Time Estimates

- **Setup**: 5 minutes
- **Running searches**: 15-20 minutes
- **Analyzing results**: 1-2 hours
- **Reading papers**: 2-3 days
- **Selection**: 1 week
- **Implementation**: 2-3 months

## 🎯 Success Criteria

Your paper choice is good if:
- ✅ Methodology matches APAI requirements
- ✅ Clear improvement opportunities exist
- ✅ Resources (code/data) are available
- ✅ Implementation is feasible
- ✅ Teacher has approved it
- ✅ You understand it well
- ✅ Timeline is realistic

## 📚 Useful Links

- **arXiv**: https://arxiv.org/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/
- **Papers With Code**: https://paperswithcode.com/
- **CVPR Proceedings**: https://openaccess.thecvf.com/
- **Medical Datasets**: http://medicaldecathlon.com/
- **Grand Challenge**: https://grand-challenge.org/

## 🚀 Next Steps

1. **Today**: Run `python example_searches.py`
2. **This Week**: Analyze results, read top 10 papers
3. **Next Week**: Select top 3, write proposals
4. **Week 3**: Get teacher approval
5. **Week 4+**: Start implementation

---

**Remember**: The goal is to find a paper with clear improvement opportunities that you can implement within your timeline. Don't aim for perfection—aim for feasibility and clear contribution!

Good luck! 🍀
