#!/bin/bash

# Example searches for APAI Project
# Medical Image Segmentation with various methodologies

echo "=========================================="
echo "APAI Project - Paper Search Examples"
echo "=========================================="
echo ""

# Create output directory
mkdir -p search_results

# 1. Medical Image Segmentation + Self-Supervised Learning
echo "1. Searching: Medical Image Segmentation + Self-Supervised Learning..."
python paper_search_tool.py \
  --keywords "medical image segmentation" "self-supervised learning" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/self_supervised_medical_seg.json \
  --output-csv search_results/self_supervised_medical_seg.csv

echo ""
echo "=========================================="
echo ""

# 2. Medical Imaging + Knowledge Distillation
echo "2. Searching: Medical Imaging + Knowledge Distillation..."
python paper_search_tool.py \
  --keywords "medical imaging" "knowledge distillation" "segmentation" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/knowledge_distillation_medical.json \
  --output-csv search_results/knowledge_distillation_medical.csv

echo ""
echo "=========================================="
echo ""

# 3. Continual Learning + Medical Diagnosis
echo "3. Searching: Continual Learning + Medical Diagnosis..."
python paper_search_tool.py \
  --keywords "continual learning" "medical diagnosis" "deep learning" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/continual_learning_medical.json \
  --output-csv search_results/continual_learning_medical.csv

echo ""
echo "=========================================="
echo ""

# 4. Meta-Learning + Few-Shot Medical Imaging
echo "4. Searching: Meta-Learning + Few-Shot Medical Imaging..."
python paper_search_tool.py \
  --keywords "meta learning" "few-shot" "medical imaging" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/meta_learning_medical.json \
  --output-csv search_results/meta_learning_medical.csv

echo ""
echo "=========================================="
echo ""

# 5. Vision-Language Models + Medical Applications
echo "5. Searching: Vision-Language Models + Medical Applications..."
python paper_search_tool.py \
  --keywords "vision language model" "medical" "CLIP" "multimodal" \
  --year-start 2023 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/vlm_medical.json \
  --output-csv search_results/vlm_medical.csv

echo ""
echo "=========================================="
echo ""

# 6. Semantic Segmentation + Healthcare
echo "6. Searching: Semantic Segmentation + Healthcare..."
python paper_search_tool.py \
  --keywords "semantic segmentation" "healthcare" "deep learning" "CNN" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 50 \
  --output-json search_results/semantic_seg_healthcare.json \
  --output-csv search_results/semantic_seg_healthcare.csv

echo ""
echo "=========================================="
echo "All searches complete!"
echo "Results saved in: search_results/"
echo "=========================================="
