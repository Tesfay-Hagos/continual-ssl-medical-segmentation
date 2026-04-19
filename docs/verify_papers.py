#!/usr/bin/env python3
"""
Paper DOI Verification Tool — Cross-checks DOIs against CrossRef and Semantic Scholar.
For each paper:
  1. Resolves DOI via CrossRef API (title, authors, year, journal, publisher)
  2. Fetches citation count via Semantic Scholar
  3. Cross-checks stored title vs. CrossRef title (detects hallucinated papers)
  4. Produces a confidence verdict: VERIFIED / MISMATCH / NO_DOI / UNRESOLVABLE

Usage:
    python verify_papers.py                        # verifies our curated priority list
    python verify_papers.py --all                  # verifies all 219 from related_papers.json
    python verify_papers.py --doi 10.1109/...      # verify a single DOI
"""

import json
import time
import re
import argparse
import csv
import os
import sys
from typing import Optional
import requests

CROSSREF_API  = "https://api.crossref.org/works"
SEMSCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
HEADERS = {"User-Agent": "ContinualSSL-PaperVerifier/1.0 (tesfay.hagos1421@gmail.com)"}

# ── Priority list: papers we actually plan to cite ────────────────────────────
# Format: (label, doi_or_arxiv, stored_title)
PRIORITY_PAPERS = [
    # ── Primary reference ──
    ("FedCSL [PRIMARY]",
     "10.1109/TNNLS.2024.3469962",
     "Federated Cross-Incremental Self-Supervised Learning for Medical Image Segmentation"),

    # ── SSL / MIM pretraining ──
    ("MAE (He 2022)",
     "10.48550/arXiv.2111.06377",
     "Masked Autoencoders Are Scalable Vision Learners"),

    ("SparK (Tian 2023)",
     "10.48550/arXiv.2301.03580",
     "Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling"),

    ("GMIM (Qi 2024)",
     "10.1016/j.compbiomed.2024.108547",
     "GMIM: Self-supervised pre-training for 3D medical image segmentation with grouped masked image modeling"),

    ("Hybrid MIM 3D (Xing 2024)",
     "10.1109/JBHI.2024.3360239",
     "Hybrid Masked Image Modeling for 3D Medical Image Segmentation"),

    ("Rethinking MIM Medical (Xie 2024)",
     "10.1016/j.media.2024.103304",
     "Rethinking masked image modelling for medical image representation"),

    ("Masked Deformation MRI (Lyu 2025)",
     "10.1109/TMI.2024.3510922",
     "Masked Deformation Modeling for Volumetric Brain MRI Self-Supervised Pre-training"),

    ("Hi-End-MAE (Tang 2026)",
     "10.1016/j.media.2025.103770",
     "Hi-End-MAE: Hierarchical encoder-driven masked autoencoders are stronger vision learners for medical image segmentation"),

    # ── Continual Learning (seminal) ──
    ("EWC (Kirkpatrick 2017)",
     "10.1073/pnas.1611835114",
     "Overcoming catastrophic forgetting in neural networks"),

    ("LwF (Li & Hoiem 2016)",
     "10.1007/978-3-319-46493-0_37",
     "Learning without Forgetting"),

    ("GEM (Lopez-Paz 2017)",
     "10.48550/arXiv.1706.08840",
     "Gradient Episodic Memory for Continual Learning"),

    # ── CL in medical segmentation ──
    ("MOSInversion (Kim 2025)",
     "10.1016/j.compbiomed.2025.111272",
     "MOSInversion: Knowledge distillation-based incremental learning in organ segmentation"),

    ("Continual Domain Adapt NPC (Yang 2025)",
     "10.1016/j.neunet.2025.107869",
     "Continual source-free active domain adaptation for nasopharyngeal carcinoma tumor segmentation"),

    # ── Architecture / baseline ──
    ("U-Net (Ronneberger 2015)",
     "10.1007/978-3-319-24574-4_28",
     "U-Net: Convolutional Networks for Biomedical Image Segmentation"),

    ("nnU-Net (Isensee 2021)",
     "10.1038/s41592-020-01008-z",
     "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation"),

    # ── Dataset ──
    ("Medical Decathlon (Simpson 2019)",
     "10.48550/arXiv.1902.09063",
     "A large annotated medical image dataset for the development and evaluation of segmentation algorithms"),
]


# ── CrossRef lookup ───────────────────────────────────────────────────────────

def crossref_lookup(doi: str) -> Optional[dict]:
    """Return CrossRef metadata dict or None on failure."""
    url = f"{CROSSREF_API}/{doi}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json().get("message", {})
        return None
    except Exception:
        return None


def semscholar_lookup(doi: str) -> Optional[dict]:
    """Return Semantic Scholar metadata or None."""
    url = f"{SEMSCHOLAR_API}/DOI:{doi}"
    params = {"fields": "title,citationCount,year,venue,authors,externalIds"}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def title_similarity(a: str, b: str) -> float:
    """Rough word-overlap similarity (0–1)."""
    wa = set(re.sub(r"[^a-z0-9 ]", "", a.lower()).split())
    wb = set(re.sub(r"[^a-z0-9 ]", "", b.lower()).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(len(wa), len(wb))


def journal_tier(venue: str) -> str:
    """Rough journal tier label based on known venues."""
    v = venue.lower()
    tier1 = ["nature", "ieee transactions on medical imaging", "medical image analysis",
             "miccai", "cvpr", "iccv", "eccv", "neurips", "iclr", "icml",
             "ieee tnnls", "tpami", "pnas", "science"]
    tier2 = ["ieee jbhi", "computers in biology", "scientific reports", "frontiers",
              "neuroimage", "radiology", "npj", "physics in medicine"]
    for t in tier1:
        if t in v:
            return "Tier-1"
    for t in tier2:
        if t in v:
            return "Tier-2"
    if "arxiv" in v:
        return "Preprint"
    return "Other"


# ── Main verifier ─────────────────────────────────────────────────────────────

def verify_one(label: str, doi: str, stored_title: str) -> dict:
    """Verify a single paper. Returns a result dict."""
    result = {
        "label":         label,
        "doi":           doi,
        "stored_title":  stored_title,
        "cr_title":      None,
        "cr_journal":    None,
        "cr_publisher":  None,
        "cr_year":       None,
        "cr_type":       None,
        "ss_citations":  None,
        "ss_venue":      None,
        "title_sim":     None,
        "tier":          None,
        "verdict":       None,
        "notes":         [],
    }

    # ── CrossRef ──
    cr = crossref_lookup(doi)
    if cr:
        # Title
        titles = cr.get("title", [])
        result["cr_title"] = titles[0] if titles else None
        # Journal
        containers = cr.get("container-title", [])
        result["cr_journal"] = containers[0] if containers else cr.get("institution", [None])[0]
        result["cr_publisher"] = cr.get("publisher")
        result["cr_type"] = cr.get("type")
        # Year
        dp = cr.get("published", cr.get("published-print", cr.get("published-online", {})))
        date_parts = dp.get("date-parts", [[None]])[0]
        result["cr_year"] = date_parts[0] if date_parts else None
        # Title similarity
        if result["cr_title"]:
            result["title_sim"] = title_similarity(stored_title, result["cr_title"])
    else:
        result["notes"].append("CrossRef returned no data")

    # ── Semantic Scholar ──
    ss = semscholar_lookup(doi)
    if ss:
        result["ss_citations"] = ss.get("citationCount")
        result["ss_venue"]     = ss.get("venue") or ss.get("publicationVenue", {})
        if isinstance(result["ss_venue"], dict):
            result["ss_venue"] = result["ss_venue"].get("name", "")

    venue_str = (result["cr_journal"] or result["ss_venue"] or "")
    result["tier"] = journal_tier(venue_str)

    # ── Verdict ──
    if not cr and not ss:
        result["verdict"] = "UNRESOLVABLE"
        result["notes"].append("DOI not found in CrossRef or Semantic Scholar")
    elif result["title_sim"] is None:
        result["verdict"] = "NO_TITLE_CHECK"
        result["notes"].append("CrossRef had no title to compare")
    elif result["title_sim"] >= 0.50:
        result["verdict"] = "VERIFIED"
    elif result["title_sim"] >= 0.25:
        result["verdict"] = "WEAK_MATCH"
        result["notes"].append(f"Title similarity only {result['title_sim']:.0%} — review manually")
    else:
        result["verdict"] = "MISMATCH"
        result["notes"].append("Title does not match DOI — possible hallucination or wrong DOI")

    return result


def print_result(r: dict, idx: int):
    verdict_icon = {
        "VERIFIED":       "✅",
        "WEAK_MATCH":     "⚠️ ",
        "MISMATCH":       "❌",
        "UNRESOLVABLE":   "🔴",
        "NO_TITLE_CHECK": "🔵",
    }.get(r["verdict"], "❓")

    print(f"\n{'─'*78}")
    print(f"{idx:>2}. {r['label']}")
    print(f"    DOI      : {r['doi']}")
    print(f"    Stored   : {r['stored_title'][:72]}")
    if r["cr_title"]:
        print(f"    CrossRef : {r['cr_title'][:72]}")
    print(f"    Year     : {r['cr_year']}   Journal: {r['cr_journal'] or r['ss_venue'] or '—'}")
    print(f"    Publisher: {r['cr_publisher'] or '—'}")
    print(f"    Citations: {r['ss_citations'] if r['ss_citations'] is not None else '—'}")
    print(f"    Tier     : {r['tier'] or '—'}")
    sim_str = f"{r['title_sim']:.0%}" if r['title_sim'] is not None else "—"
    print(f"    TitleSim : {sim_str}")
    print(f"    Verdict  : {verdict_icon} {r['verdict']}")
    if r["notes"]:
        for n in r["notes"]:
            print(f"    NOTE     : {n}")


def save_report(results: list, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Rank", "Label", "Verdict", "TitleSimilarity", "Citations", "Tier",
            "Year", "Journal", "Publisher", "CrossRefTitle", "StoredTitle", "DOI", "Notes"
        ])
        for i, r in enumerate(results, 1):
            writer.writerow([
                i,
                r["label"],
                r["verdict"],
                f"{r['title_sim']:.0%}" if r["title_sim"] is not None else "—",
                r["ss_citations"] if r["ss_citations"] is not None else "—",
                r["tier"],
                r["cr_year"],
                r["cr_journal"] or r["ss_venue"] or "—",
                r["cr_publisher"] or "—",
                r["cr_title"] or "—",
                r["stored_title"],
                r["doi"],
                "; ".join(r["notes"]),
            ])
    print(f"\n  Report saved → {path}")


def print_summary(results: list):
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1

    total = len(results)
    print(f"\n{'='*78}")
    print(f"  VERIFICATION SUMMARY  ({total} papers)")
    print(f"{'='*78}")
    for verdict, icon in [
        ("VERIFIED",       "✅ VERIFIED      "),
        ("WEAK_MATCH",     "⚠️  WEAK MATCH   "),
        ("NO_TITLE_CHECK", "🔵 NO TITLE CHK "),
        ("MISMATCH",       "❌ MISMATCH      "),
        ("UNRESOLVABLE",   "🔴 UNRESOLVABLE  "),
    ]:
        n = counts.get(verdict, 0)
        bar = "█" * n
        print(f"  {icon}: {n:>3}  {bar}")

    print(f"\n  Tier breakdown (Verified papers only):")
    tier_counts = {}
    for r in results:
        if r["verdict"] in ("VERIFIED", "NO_TITLE_CHECK"):
            t = r["tier"] or "Unknown"
            tier_counts[t] = tier_counts.get(t, 0) + 1
    for tier, n in sorted(tier_counts.items()):
        print(f"    {tier:<12}: {n}")

    flagged = [r for r in results if r["verdict"] in ("MISMATCH", "UNRESOLVABLE")]
    if flagged:
        print(f"\n  ⚠️  Papers to REMOVE from your reference list:")
        for r in flagged:
            print(f"    - {r['label']}  ({r['doi']})")

    print(f"{'='*78}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Verify paper DOIs via CrossRef + Semantic Scholar")
    ap.add_argument("--all",  action="store_true",
                    help="Verify all papers in related_papers/related_papers.json")
    ap.add_argument("--doi",  type=str,
                    help="Verify a single DOI interactively")
    ap.add_argument("--top",  type=int, default=None,
                    help="Limit --all mode to top N papers by relevance score")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "related_papers")
    os.makedirs(out_dir, exist_ok=True)

    if args.doi:
        # Single DOI mode
        r = verify_one("(manual)", args.doi, "")
        print_result(r, 1)
        return

    if args.all:
        json_path = os.path.join(out_dir, "related_papers.json")
        with open(json_path, encoding="utf-8") as f:
            all_papers = json.load(f)
        if args.top:
            all_papers = all_papers[:args.top]
        targets = [
            (p.get("title", "")[:40], p.get("doi") or "", p.get("title", ""))
            for p in all_papers if p.get("doi") and p["doi"] != "N/A"
        ]
        report_path = os.path.join(out_dir, "doi_verification_all.csv")
    else:
        # Default: verify priority list
        targets = [(label, doi, title) for label, doi, title in PRIORITY_PAPERS]
        report_path = os.path.join(out_dir, "doi_verification_priority.csv")

    print("=" * 78)
    print("  PAPER DOI VERIFICATION — CrossRef + Semantic Scholar")
    print(f"  Checking {len(targets)} papers")
    print("=" * 78)

    results = []
    for i, (label, doi, stored_title) in enumerate(targets, 1):
        print(f"  [{i}/{len(targets)}] {label[:55]}...", end="\r", flush=True)
        r = verify_one(label, doi, stored_title)
        results.append(r)
        print_result(r, i)
        time.sleep(0.6)   # respect rate limits

    print_summary(results)
    save_report(results, report_path)


if __name__ == "__main__":
    main()
