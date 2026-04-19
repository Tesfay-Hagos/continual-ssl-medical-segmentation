# DOI Verification Guide

## What is a DOI?

A **DOI (Digital Object Identifier)** is a unique alphanumeric string assigned to academic papers, datasets, and other digital objects. It provides a permanent link to the resource and helps verify the authenticity of a paper.

### DOI Format Examples:
- `10.1109/JBHI.2023.3340956`
- `10.1007/s10278-024-01319-z`
- `10.1109/TNNLS.2024.3469962`

The format is typically: `10.XXXX/identifier`

## Why DOI is Important

### 1. **Authenticity Verification**
- ✅ Papers with DOI are officially published
- ✅ DOI confirms the paper is peer-reviewed
- ✅ Helps distinguish real papers from preprints or fake papers

### 2. **Permanent Access**
- DOI provides a permanent link that won't break
- Even if journal website changes, DOI resolver still works

### 3. **Proper Citation**
- DOI is required for proper academic citations
- Makes it easy for others to find your references

### 4. **Quality Indicator**
- Papers with DOI have gone through peer review
- Published in recognized journals/conferences

## How to Verify a Paper Using DOI

### Method 1: DOI Resolver (Recommended)

Visit: **https://doi.org/** and append the DOI

**Example:**
```
https://doi.org/10.1109/JBHI.2023.3340956
```

This will redirect you to the official publisher page.

### Method 2: CrossRef

Visit: **https://www.crossref.org/**

Search for the DOI to get metadata including:
- Authors
- Publication date
- Journal/Conference
- Citation information

### Method 3: Google Scholar

Search: `DOI: 10.1109/JBHI.2023.3340956`

This will show:
- Citation count
- Related papers
- Versions available

## Checking Papers in Your Search Results

### Using the JSON File

```python
import json

# Load your search results
with open('papers_results.json', 'r') as f:
    papers = json.load(f)

# Check which papers have DOI
for paper in papers:
    doi = paper.get('doi')
    if doi:
        print(f"✅ {paper['title'][:60]}...")
        print(f"   DOI: {doi}")
        print(f"   Verify at: https://doi.org/{doi}\n")
    else:
        print(f"⚠️  {paper['title'][:60]}...")
        print(f"   No DOI available (might be preprint)\n")
```

### Using the CSV File

Open `papers_results.csv` in Excel/LibreOffice and:
1. Look at the **DOI** column
2. Papers with "N/A" don't have DOI (might be preprints)
3. Copy DOI and paste into: `https://doi.org/[DOI]`

## What If a Paper Has No DOI?

### Possible Reasons:

1. **Preprint (arXiv, bioRxiv)**
   - Not yet peer-reviewed
   - May be submitted but not accepted
   - ⚠️ Use with caution for your project

2. **Very Recent**
   - Just accepted, DOI not assigned yet
   - Check back in a few weeks

3. **Conference Paper**
   - Some conferences assign DOI later
   - Check conference proceedings

4. **Not Officially Published**
   - Might be a workshop paper
   - Might be a technical report
   - ⚠️ Verify quality before using

### How to Verify Papers Without DOI:

1. **Check arXiv ID**
   - arXiv papers have format: `arXiv:2301.12345`
   - Verify at: https://arxiv.org/abs/2301.12345

2. **Check PubMed ID (PMID)**
   - Medical papers have PMID
   - Verify at: https://pubmed.ncbi.nlm.nih.gov/[PMID]/

3. **Check Conference Proceedings**
   - Look up the conference
   - Verify paper is in official proceedings

4. **Contact Authors**
   - Email corresponding author
   - Ask about publication status

## Red Flags (Fake or Low-Quality Papers)

### 🚩 Warning Signs:

1. **No DOI and No arXiv ID**
   - Might not be a real publication
   - Could be unpublished work

2. **DOI Doesn't Resolve**
   - Try: https://doi.org/[DOI]
   - If it doesn't work, paper might be fake

3. **Published in Unknown Venue**
   - Check if journal/conference is recognized
   - Beware of predatory journals

4. **Too Good to Be True**
   - Claims "100% accuracy"
   - No limitations mentioned
   - No comparison to baselines

5. **Poor English/Formatting**
   - Real papers are professionally edited
   - Check for grammatical errors

## Recommended Verification Workflow

### Step 1: Check DOI Availability
```bash
# Run your search
python paper_search_tool.py --keywords "medical segmentation" --year-start 2023 --year-end 2024

# Check results
python analyze_papers.py papers_results.json --top 20
```

### Step 2: Verify Top Candidates

For each paper in your top 10:

1. **Copy the DOI** from the output
2. **Visit**: `https://doi.org/[DOI]`
3. **Verify**:
   - ✅ Redirects to official publisher
   - ✅ Authors match
   - ✅ Title matches
   - ✅ Year matches
   - ✅ Journal/Conference is recognized

### Step 3: Cross-Reference

1. **Google Scholar**: Search by title
   - Check citation count
   - Look for multiple versions
   - Verify authors

2. **Semantic Scholar**: Search by DOI
   - Check citation graph
   - See related papers
   - Check influence metrics

3. **Publisher Website**: Visit directly
   - Read full abstract
   - Check if open access
   - Download PDF

### Step 4: Quality Check

Before choosing a paper:
- [ ] DOI resolves correctly
- [ ] Published in recognized venue
- [ ] Authors are from real institutions
- [ ] Has reasonable citation count for its age
- [ ] Abstract makes sense
- [ ] Methodology is clearly described
- [ ] Results are realistic (not "perfect")

## DOI in Your APAI Project

### In Your Paper's References Section

Use DOI in citations:

```latex
@article{qayyum2024two,
  title={Two-Stage Self-Supervised Contrastive Learning Aided Transformer for Real-Time Medical Image Segmentation},
  author={Qayyum, Abdul and Razzak, Imran and Mazher, Moona and Khan, Tariq and Ding, Weiping and Niederer, Steven},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2024},
  doi={10.1109/JBHI.2023.3340956}
}
```

### In Your Paper's Abstract

Include your code repository URL:
```
Code available at: https://github.com/yourusername/yourproject
```

## Quick DOI Verification Script

Save this as `verify_dois.py`:

```python
#!/usr/bin/env python3
import json
import requests
import sys

def verify_doi(doi):
    """Verify if DOI is valid"""
    if not doi or doi == 'N/A':
        return False, "No DOI"
    
    url = f"https://doi.org/{doi}"
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        if response.status_code == 200:
            return True, f"Valid - redirects to {response.url}"
        else:
            return False, f"Invalid - status code {response.status_code}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    if len(sys.argv) < 2:
        print("Usage: python verify_dois.py papers_results.json")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        papers = json.load(f)
    
    print(f"Verifying DOIs for {len(papers)} papers...\n")
    
    valid_count = 0
    for i, paper in enumerate(papers, 1):
        doi = paper.get('doi')
        is_valid, message = verify_doi(doi)
        
        status = "✅" if is_valid else "❌"
        print(f"{i}. {status} {paper['title'][:60]}...")
        print(f"   DOI: {doi if doi else 'Not available'}")
        print(f"   Status: {message}\n")
        
        if is_valid:
            valid_count += 1
    
    print(f"\nSummary: {valid_count}/{len(papers)} papers have valid DOIs")

if __name__ == '__main__':
    main()
```

Run it:
```bash
python verify_dois.py papers_results.json
```

## Summary

### ✅ Papers WITH DOI:
- Officially published
- Peer-reviewed
- Safe to use for your project
- Easy to cite properly

### ⚠️ Papers WITHOUT DOI:
- Might be preprints (arXiv)
- Not yet peer-reviewed
- Use with caution
- Verify through other means

### 🚩 Papers with INVALID DOI:
- DOI doesn't resolve
- Might be fake
- Avoid for your project

## For Your APAI Project

### Before Choosing a Paper:

1. **Verify DOI** (if available)
2. **Check venue** (conference/journal reputation)
3. **Read full paper** (not just abstract)
4. **Verify code availability** (GitHub link)
5. **Check dataset access** (public or not)
6. **Discuss with teacher** (get approval)

### When Writing Your Paper:

1. **Include DOI** in all references
2. **Provide your code URL** in abstract
3. **Cite properly** using DOI
4. **Make your work reproducible**

## Resources

- **DOI Resolver**: https://doi.org/
- **CrossRef**: https://www.crossref.org/
- **Google Scholar**: https://scholar.google.com/
- **Semantic Scholar**: https://www.semanticscholar.org/
- **arXiv**: https://arxiv.org/
- **PubMed**: https://pubmed.ncbi.nlm.nih.gov/

---

**Remember**: DOI is your friend for verifying paper authenticity. Always check it before committing to a paper for your project!
