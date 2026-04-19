# Academic Paper Search Tool for APAI Project

This tool helps you find recent academic papers suitable for your APAI project by searching multiple academic databases (arXiv, Semantic Scholar, and PubMed) based on keywords and year range.

## Features

- 🔍 **Multi-source search**: Searches arXiv, Semantic Scholar, and PubMed
- 📅 **Year filtering**: Filter papers by publication year range
- 🔑 **Keyword matching**: Finds papers with keywords in title or abstract
- 📊 **Citation metrics**: Shows citation counts (where available)
- 💾 **Export results**: Saves results to JSON and CSV formats
- 🏥 **Health-focused**: Includes PubMed for medical/health papers

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python paper_search_tool.py --keywords "image segmentation" "medical" --year-start 2022 --year-end 2024
```

### Example Searches for APAI Project

#### 1. Medical Image Segmentation with Deep Learning
```bash
python paper_search_tool.py \
  --keywords "image segmentation" "medical" "deep learning" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 100
```

#### 2. Self-Supervised Learning for Medical Imaging
```bash
python paper_search_tool.py \
  --keywords "self-supervised learning" "medical imaging" \
  --year-start 2023 \
  --year-end 2024 \
  --max-results 100
```

#### 3. Continual Learning in Healthcare
```bash
python paper_search_tool.py \
  --keywords "continual learning" "healthcare" "computer vision" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 100
```

#### 4. Knowledge Distillation for Medical Diagnosis
```bash
python paper_search_tool.py \
  --keywords "knowledge distillation" "medical diagnosis" "neural networks" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 100
```

#### 5. Meta-Learning for Few-Shot Medical Image Analysis
```bash
python paper_search_tool.py \
  --keywords "meta learning" "few-shot" "medical image" \
  --year-start 2022 \
  --year-end 2024 \
  --max-results 100
```

#### 6. Vision-Language Models for Medical Applications
```bash
python paper_search_tool.py \
  --keywords "vision language model" "medical" "CLIP" \
  --year-start 2023 \
  --year-end 2024 \
  --max-results 100
```

## Command-Line Arguments

- `--keywords`: Keywords to search for (space-separated, required)
- `--year-start`: Start year for search range (required)
- `--year-end`: End year for search range (required)
- `--max-results`: Maximum results per source (default: 100)
- `--output-json`: Output JSON filename (default: papers_results.json)
- `--output-csv`: Output CSV filename (default: papers_results.csv)

## Output Files

The tool generates two output files:

1. **papers_results.json**: Complete paper data in JSON format
2. **papers_results.csv**: Spreadsheet-friendly format with key information

### Additional Tools

3. **verify_dois.py**: Verify DOI authenticity
   ```bash
   python verify_dois.py papers_results.json
   ```
   This checks if DOIs are valid and helps identify fake or low-quality papers.

## Output Format

Each paper includes:
- Title
- Authors
- Publication year
- Abstract
- URL
- Venue/Journal
- Citation count
- Source database
- Matched keywords
- **DOI (Digital Object Identifier)** - for verifying paper authenticity

## Tips for Finding Good Papers

### 1. Look for Recent Papers (2022-2024)
Recent papers often have:
- More detailed "Future Work" sections
- Open problems to address
- Available code repositories

### 2. Focus on Specific Domains
Combine methodology with domain:
- "self-supervised learning" + "medical imaging"
- "continual learning" + "radiology"
- "meta learning" + "pathology"

### 3. Check for Implementation Opportunities
Look for papers that mention:
- "Future work"
- "Limitations"
- "Can be extended to"
- "Further research needed"

### 4. Verify Methodology Complexity
According to APAI requirements, suitable methodologies include:
- Self-supervised learning
- Knowledge distillation
- Continual Learning
- Meta Learning
- VLMs/MLLMs (CLIP, BLIP, etc.)

## Recommended Search Strategy

1. **Start broad**: Search with general keywords
   ```bash
   python paper_search_tool.py --keywords "medical image segmentation" "deep learning" --year-start 2023 --year-end 2024
   ```

2. **Narrow down**: Add methodology keywords
   ```bash
   python paper_search_tool.py --keywords "medical image segmentation" "self-supervised" --year-start 2023 --year-end 2024
   ```

3. **Review results**: Check the CSV file for promising papers

4. **Read papers**: Focus on:
   - Abstract (problem and approach)
   - Introduction (motivation and gaps)
   - Conclusion (limitations and future work)

5. **Identify opportunities**: Look for:
   - Untested datasets
   - Proposed but unimplemented ideas
   - Limitations that can be addressed
   - Extensions mentioned in future work

## Example Workflow

```bash
# Step 1: Search for papers
python paper_search_tool.py \
  --keywords "image segmentation" "medical" "self-supervised" \
  --year-start 2023 \
  --year-end 2024

# Step 2: Open the CSV file
# Review papers_results.csv in Excel/LibreOffice

# Step 3: Read promising papers
# Focus on papers with:
# - High citations
# - Recent publication (2024)
# - Clear future work sections

# Step 4: Identify improvement opportunities
# Look for papers that mention:
# - "can be extended"
# - "future work"
# - "limitations"
```

## Troubleshooting

### No results found
- Try broader keywords
- Expand year range
- Check internet connection

### Too many results
- Add more specific keywords
- Narrow year range
- Reduce --max-results

### API rate limiting
- The tool includes delays between requests
- If you hit limits, wait a few minutes and try again

## Notes

- **arXiv**: Best for recent preprints and computer science papers
- **Semantic Scholar**: Good for citation counts and cross-disciplinary search
- **PubMed**: Essential for medical/health-related papers

## Next Steps After Finding Papers

1. **Read the full paper** (not just abstract)
2. **Check if code is available** (GitHub links)
3. **Review the future work section** carefully
4. **Verify dataset availability**
5. **Assess implementation complexity**
6. **Discuss with your teacher** before committing

## Contact

For APAI project questions, contact your teacher: Cigdem Beyan

Good luck with your project! 🚀
