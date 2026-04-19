#!/usr/bin/env python3
"""
Example searches for APAI Project
Runs multiple pre-configured searches for medical image segmentation papers
"""

import os
import subprocess
from pathlib import Path


def run_search(name, keywords, year_start, year_end, max_results=50):
    """Run a paper search with given parameters"""
    print(f"\n{'='*80}")
    print(f"🔍 {name}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path("search_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create safe filename
    safe_name = name.lower().replace(" ", "_").replace("+", "").replace("-", "_")
    json_file = output_dir / f"{safe_name}.json"
    csv_file = output_dir / f"{safe_name}.csv"
    
    # Build command
    cmd = [
        "python", "paper_search_tool.py",
        "--keywords", *keywords,
        "--year-start", str(year_start),
        "--year-end", str(year_end),
        "--max-results", str(max_results),
        "--output-json", str(json_file),
        "--output-csv", str(csv_file)
    ]
    
    # Run search
    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ Search complete! Results saved to:")
        print(f"   - {json_file}")
        print(f"   - {csv_file}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Search failed: {e}")
    except KeyboardInterrupt:
        print(f"\n⚠️  Search interrupted by user")
        return False
    
    return True


def main():
    """Run all example searches"""
    print(f"{'='*80}")
    print(f"🔬 APAI PROJECT - PAPER SEARCH EXAMPLES")
    print(f"{'='*80}")
    print(f"\nThis script will run multiple searches for papers suitable for your APAI project.")
    print(f"Each search focuses on a different methodology combined with medical imaging.\n")
    
    searches = [
        {
            "name": "Medical Image Segmentation + Self-Supervised Learning",
            "keywords": ["medical image segmentation", "self-supervised learning"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Medical Imaging + Knowledge Distillation",
            "keywords": ["medical imaging", "knowledge distillation", "segmentation"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Continual Learning + Medical Diagnosis",
            "keywords": ["continual learning", "medical diagnosis", "deep learning"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Meta-Learning + Few-Shot Medical Imaging",
            "keywords": ["meta learning", "few-shot", "medical imaging"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Vision-Language Models + Medical Applications",
            "keywords": ["vision language model", "medical", "CLIP", "multimodal"],
            "year_start": 2023,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Semantic Segmentation + Healthcare",
            "keywords": ["semantic segmentation", "healthcare", "deep learning", "CNN"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Transfer Learning + Medical Image Analysis",
            "keywords": ["transfer learning", "medical image analysis", "pre-training"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        },
        {
            "name": "Attention Mechanisms + Medical Segmentation",
            "keywords": ["attention mechanism", "medical segmentation", "transformer"],
            "year_start": 2022,
            "year_end": 2024,
            "max_results": 50
        }
    ]
    
    print(f"Total searches to run: {len(searches)}\n")
    input("Press Enter to start searching (or Ctrl+C to cancel)...")
    
    completed = 0
    for i, search in enumerate(searches, 1):
        print(f"\n{'#'*80}")
        print(f"# Search {i}/{len(searches)}")
        print(f"{'#'*80}")
        
        success = run_search(
            name=search["name"],
            keywords=search["keywords"],
            year_start=search["year_start"],
            year_end=search["year_end"],
            max_results=search["max_results"]
        )
        
        if not success:
            break
        
        completed += 1
    
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    print(f"Completed: {completed}/{len(searches)} searches")
    print(f"Results saved in: search_results/")
    print(f"\n💡 Next steps:")
    print(f"   1. Review the CSV files in search_results/")
    print(f"   2. Identify papers with interesting future work sections")
    print(f"   3. Read the full papers for promising candidates")
    print(f"   4. Check if datasets and code are available")
    print(f"   5. Discuss your choice with your teacher")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Interrupted by user. Exiting...")
