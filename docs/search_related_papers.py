#!/usr/bin/env python3
"""
Targeted paper search for:
  "Continual Self-Supervised Learning for Medical Image Segmentation:
   A Simplified Framework Without Federated Components"

Methodology: U-Net encoder + SparK/MIM pretraining + EWC/LwF/Replay CL strategies
Datasets:    Medical Segmentation Decathlon (Liver CT, Pancreas CT, Heart MRI)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'title_selection'))

from paper_search_tool import PaperSearcher, Paper
import json
import csv
from typing import List, Dict
from datetime import datetime

# ── Search clusters ────────────────────────────────────────────────────────────
# Each entry: (label, query_string, keyword_filters, year_start, year_end)
SEARCH_CLUSTERS = [
    (
        "core_topic",
        "continual self-supervised learning medical image segmentation",
        ["continual learning", "self-supervised", "medical image", "segmentation"],
        2020, 2026
    ),
    (
        "masked_image_modeling_medical",
        "masked image modeling medical image segmentation pretraining",
        ["masked", "medical", "segmentation"],
        2021, 2026
    ),
    (
        "sparse_masked_cnn_pretraining",
        "sparse masked autoencoder CNN U-Net pretraining self-supervised",
        ["masked", "encoder", "pretraining"],
        2021, 2026
    ),
    (
        "catastrophic_forgetting_medical",
        "catastrophic forgetting continual learning medical imaging organ segmentation",
        ["catastrophic forgetting", "continual", "medical"],
        2019, 2026
    ),
    (
        "ewc_lwf_replay_segmentation",
        "elastic weight consolidation learning without forgetting experience replay segmentation",
        ["segmentation", "continual"],
        2019, 2026
    ),
    (
        "sequential_organ_segmentation",
        "sequential task learning organ segmentation incremental continual",
        ["organ", "segmentation", "sequential"],
        2020, 2026
    ),
    (
        "unet_self_supervised_medical",
        "U-Net self-supervised representation learning medical image segmentation",
        ["u-net", "self-supervised", "segmentation"],
        2020, 2026
    ),
    (
        "knowledge_distillation_medical_segmentation",
        "knowledge distillation teacher student medical image segmentation efficient",
        ["knowledge distillation", "medical", "segmentation"],
        2020, 2026
    ),
    (
        "federated_ssl_medical_forgetting",
        "federated learning self-supervised continual learning medical segmentation forgetting",
        ["federated", "continual", "medical"],
        2021, 2026
    ),
    (
        "medical_decathlon_benchmark_ssl",
        "medical segmentation decathlon benchmark self-supervised continual learning",
        ["decathlon", "segmentation"],
        2019, 2026
    ),
]


def run_all_searches(max_per_source: int = 80) -> List[Dict]:
    """Run all search clusters and merge deduplicated results."""
    all_papers: Dict[str, dict] = {}  # keyed by normalised title

    for label, query, keywords, yr_start, yr_end in SEARCH_CLUSTERS:
        print(f"\n{'='*70}")
        print(f"CLUSTER: {label}  ({yr_start}–{yr_end})")
        print(f"Query   : {query}")
        print(f"Filters : {keywords}")
        print(f"{'='*70}")

        searcher = PaperSearcher(yr_start, yr_end, keywords)
        papers = searcher.search_all(query, max_per_source)

        for p in papers:
            key = p.title.lower().strip()
            if key not in all_papers:
                d = p.to_dict()
                d['cluster'] = label
                all_papers[key] = d
            else:
                # add cluster tag if paper appears in multiple clusters
                existing = all_papers[key]['cluster']
                if label not in existing:
                    all_papers[key]['cluster'] = existing + ', ' + label

    return list(all_papers.values())


def score_paper(p: dict) -> float:
    """
    Relevance score tuned to our specific paper:
    - Higher weight for papers mentioning continual + SSL + medical together
    - Recency bonus (2023-2026)
    - Citation signal (capped)
    """
    text = (p.get('title', '') + ' ' + p.get('abstract', '')).lower()

    # Core relevance signals
    core_hits = sum([
        ('continual' in text or 'incremental' in text or 'lifelong' in text),
        ('self-supervised' in text or 'self supervised' in text or 'masked' in text),
        ('segment' in text),
        ('medical' in text or 'clinical' in text or 'radiol' in text),
        ('organ' in text or 'liver' in text or 'pancreas' in text or 'cardiac' in text),
    ])

    # CL strategy signals
    cl_hits = sum([
        'elastic weight' in text or 'ewc' in text,
        'learning without forgetting' in text or 'lwf' in text,
        'experience replay' in text or 'replay buffer' in text,
        'catastrophic forgetting' in text,
        'backward transfer' in text or 'forward transfer' in text,
    ])

    # SSL / architecture signals
    ssl_hits = sum([
        'u-net' in text or 'unet' in text,
        'encoder' in text,
        'sparse' in text and 'mask' in text,
        'mae' in text or 'masked autoencoder' in text,
        'pretraining' in text or 'pre-training' in text or 'pre-train' in text,
    ])

    # Recency (2023-2026 preferred)
    year = p.get('year', 2019)
    recency = max(0, year - 2019) * 2.0

    # Citation bonus (log-scaled, capped)
    cit = p.get('citations', 0)
    cit_score = min(cit / 50, 6.0)

    # Future work / limitations indicator (good for motivation section)
    fw_hits = sum(kw in text for kw in [
        'limitation', 'future work', 'future direction', 'remain',
        'challenge', 'open problem', 'can be extended',
    ])

    return core_hits * 5.0 + cl_hits * 4.0 + ssl_hits * 3.0 + recency + cit_score + fw_hits * 1.5


def tag_section(p: dict) -> str:
    """Suggest which paper-writing section this reference belongs to."""
    text = (p.get('title', '') + ' ' + p.get('abstract', '')).lower()
    tags = []
    if 'continual' in text or 'incremental' in text or 'catastrophic' in text:
        tags.append('RW:ContinualLearning')
    if 'self-supervised' in text or 'masked' in text or 'contrastive' in text or 'pretext' in text:
        tags.append('RW:SelfSupervisedLearning')
    if 'u-net' in text or 'unet' in text or 'encoder-decoder' in text:
        tags.append('RW:Architecture')
    if 'knowledge distil' in text or 'teacher' in text:
        tags.append('RW:KnowledgeDistillation')
    if 'medical' in text and 'segment' in text:
        tags.append('RW:MedicalSegmentation')
    if 'federated' in text:
        tags.append('RW:FederatedLearning')
    if 'decathlon' in text or 'acdc' in text or 'brats' in text or 'synapse' in text:
        tags.append('RW:Dataset')
    return ', '.join(tags) if tags else 'General'


def save_results(papers: List[dict], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # JSON — full data
    json_path = os.path.join(out_dir, 'related_papers.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    # CSV — for quick review
    csv_path = os.path.join(out_dir, 'related_papers.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Rank', 'RelevanceScore', 'SuggestedSection',
            'Title', 'Authors', 'Year', 'Venue', 'Citations',
            'Source', 'Cluster', 'DOI', 'URL', 'Abstract'
        ])
        for i, p in enumerate(papers, 1):
            writer.writerow([
                i,
                f"{p['_score']:.1f}",
                p['_section'],
                p['title'],
                '; '.join(p.get('authors', [])[:4]),
                p.get('year', ''),
                p.get('venue', ''),
                p.get('citations', 0),
                p.get('source', ''),
                p.get('cluster', ''),
                p.get('doi') or 'N/A',
                p.get('url', ''),
                (p.get('abstract', '')[:400] + '...') if len(p.get('abstract', '')) > 400 else p.get('abstract', '')
            ])

    print(f"\n  JSON  -> {json_path}")
    print(f"  CSV   -> {csv_path}")


def print_curated_list(papers: List[dict], top_n: int = 40):
    """Print a structured, section-tagged reading list."""
    print(f"\n{'='*80}")
    print(f"  CURATED RELATED-WORK LIST  (top {top_n} by relevance)")
    print(f"  Paper: Continual SSL for Medical Image Segmentation")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}\n")

    sections: Dict[str, List[dict]] = {}
    for p in papers[:top_n]:
        for tag in p['_section'].split(', '):
            sections.setdefault(tag, []).append(p)

    section_order = [
        'RW:MedicalSegmentation',
        'RW:SelfSupervisedLearning',
        'RW:ContinualLearning',
        'RW:Architecture',
        'RW:KnowledgeDistillation',
        'RW:FederatedLearning',
        'RW:Dataset',
        'General',
    ]

    seen = set()
    for sec in section_order:
        if sec not in sections:
            continue
        label = sec.replace('RW:', '').replace('ContinualLearning', 'Continual Learning') \
                   .replace('SelfSupervisedLearning', 'Self-Supervised Learning') \
                   .replace('MedicalSegmentation', 'Medical Image Segmentation') \
                   .replace('KnowledgeDistillation', 'Knowledge Distillation') \
                   .replace('FederatedLearning', 'Federated Learning (context/contrast)')
        print(f"\n── {label} {'─'*(60 - len(label))}")
        for p in sections[sec]:
            key = p['title']
            if key in seen:
                continue
            seen.add(key)
            authors = p.get('authors', [])
            first_author = authors[0].split()[-1] if authors else 'Unknown'
            et_al = ' et al.' if len(authors) > 1 else ''
            print(f"  [{p['year']}] {first_author}{et_al} — {p['title'][:72]}")
            print(f"         Venue: {p.get('venue','?'):<30}  Cit: {p.get('citations',0):<6}  Score: {p['_score']:.1f}")
            print(f"         DOI/URL: {p.get('doi') or p.get('url','N/A')}")
            print()


def main():
    print("Targeted Literature Search")
    print("Paper: Continual SSL for Medical Image Segmentation (U-Net + SparK + EWC/LwF/Replay)")
    print(f"Date : {datetime.now().strftime('%Y-%m-%d')}\n")

    papers = run_all_searches(max_per_source=80)
    print(f"\nTotal unique papers collected: {len(papers)}")

    # Score and sort
    for p in papers:
        p['_score'] = score_paper(p)
        p['_section'] = tag_section(p)
    papers.sort(key=lambda p: p['_score'], reverse=True)

    # Print curated list
    print_curated_list(papers, top_n=50)

    # Save
    out_dir = os.path.join(os.path.dirname(__file__), 'related_papers')
    save_results(papers, out_dir)

    print(f"\nTotal papers saved: {len(papers)}")
    print("Done. Use related_papers/related_papers.csv for paper writing.")


if __name__ == '__main__':
    main()
