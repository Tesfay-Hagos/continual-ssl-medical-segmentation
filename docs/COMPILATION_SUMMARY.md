# LaTeX Compilation Setup - Summary

## What Was Added

I've created a complete build system for compiling your LaTeX document to PDF.

### New Files Created

```
docs/
├── introduction.tex          # Your LaTeX document (already existed)
├── Makefile                  # ✨ NEW: Build automation
├── README.md                 # ✨ NEW: Documentation
├── MAKEFILE_GUIDE.md         # ✨ NEW: Quick guide
├── COMPILATION_SUMMARY.md    # ✨ NEW: This file
└── .gitignore                # ✨ NEW: Ignore auxiliary files
```

## Quick Start

### 1. Check Installation

```bash
cd continual_self_supervised_learning/docs/
make check
```

Expected output:
```
✓ pdflatex found
✓ bibtex found
```

### 2. Compile PDF

```bash
make
```

This will:
- Run pdflatex (pass 1)
- Run bibtex
- Run pdflatex (pass 2)
- Run pdflatex (pass 3)
- Generate `introduction.pdf`

### 3. View PDF

```bash
make view
```

This will compile and open the PDF in your default viewer.

## All Available Commands

| Command | Description |
|---------|-------------|
| `make` | Full compilation (4 passes with bibliography) |
| `make quick` | Quick compilation (1 pass, no bibliography) |
| `make view` | Compile and open PDF |
| `make clean` | Remove auxiliary files (keep PDF) |
| `make cleanall` | Remove all files including PDF |
| `make rebuild` | Clean and rebuild from scratch |
| `make check` | Check if LaTeX tools are installed |
| `make watch` | Auto-recompile on file changes |
| `make wordcount` | Count words in the PDF |
| `make help` | Show help message with all commands |

## Usage Examples

### First Time

```bash
cd docs/
make check    # Verify LaTeX is installed
make          # Compile PDF
make view     # Open PDF
```

### While Writing

```bash
# Option 1: Manual recompile
make quick

# Option 2: Auto-recompile (recommended)
make watch
# Edit introduction.tex
# PDF updates automatically
# Press Ctrl+C when done
```

### Before Submission

```bash
make rebuild    # Clean rebuild
make wordcount  # Check word count
make view       # View final PDF
```

### Clean Up

```bash
make clean      # Remove auxiliary files
make cleanall   # Remove everything including PDF
```

## What the Makefile Does

### Full Build (`make`)

1. **Pass 1**: Initial compilation
   ```bash
   pdflatex -interaction=nonstopmode -halt-on-error introduction.tex
   ```

2. **Pass 2**: Process bibliography
   ```bash
   bibtex introduction
   ```

3. **Pass 3**: Resolve references
   ```bash
   pdflatex -interaction=nonstopmode -halt-on-error introduction.tex
   ```

4. **Pass 4**: Final compilation
   ```bash
   pdflatex -interaction=nonstopmode -halt-on-error introduction.tex
   ```

### Quick Build (`make quick`)

Single pass compilation (faster, but references may be incorrect):
```bash
pdflatex -interaction=nonstopmode -halt-on-error introduction.tex
```

## Installation Requirements

### LaTeX Distribution

**Ubuntu/Debian**:
```bash
sudo apt-get install texlive-full
```

**macOS**:
```bash
brew install --cask mactex
```

**Windows**:
- Download MiKTeX: https://miktex.org/download
- Or use Overleaf: https://www.overleaf.com/

### Make Tool

**Linux**: Usually pre-installed

**macOS**:
```bash
xcode-select --install
```

**Windows**: Use Git Bash or WSL

## Alternative: Manual Compilation

If you don't have `make` or prefer manual compilation:

```bash
cd docs/

# Full compilation
pdflatex introduction.tex
bibtex introduction
pdflatex introduction.tex
pdflatex introduction.tex

# Quick compilation
pdflatex introduction.tex
```

## Alternative: Overleaf (Online)

No installation needed:

1. Go to https://www.overleaf.com/
2. Create free account
3. Click "New Project" → "Upload Project"
4. Upload `introduction.tex`
5. Click "Recompile"
6. Download PDF

## Troubleshooting

### "make: command not found"

**Solution 1**: Install make
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

**Solution 2**: Use manual compilation (see above)

### "pdflatex: command not found"

**Solution**: Install LaTeX distribution (see Installation Requirements)

### Compilation Errors

**Solution 1**: Check log file
```bash
cat introduction.log | grep -i error
```

**Solution 2**: Clean and rebuild
```bash
make cleanall
make
```

**Solution 3**: Use Overleaf (online compilation)

### PDF Viewer Not Opening

**Solution**: Edit `Makefile` line 23 to set your preferred viewer:
```makefile
VIEWER = xdg-open    # Linux (default)
VIEWER = evince      # Linux (alternative)
VIEWER = okular      # Linux (alternative)
VIEWER = open        # macOS
VIEWER = start       # Windows
```

## Files Generated

### Source Files (Keep These)
- `introduction.tex` - LaTeX source
- `Makefile` - Build script
- `README.md` - Documentation

### Generated Files (Can Delete)
- `introduction.pdf` - Output PDF ✓
- `introduction.aux` - Auxiliary file
- `introduction.log` - Compilation log
- `introduction.bbl` - Bibliography
- `introduction.blg` - Bibliography log
- `introduction.out` - Hyperref output

Use `make clean` to remove auxiliary files automatically.

## Git Integration

The `.gitignore` file is configured to ignore auxiliary files:

```gitignore
*.aux
*.log
*.out
*.bbl
*.blg
*.synctex.gz
# etc.
```

You can choose to track or ignore the PDF:
- **Track PDF**: Useful for sharing with team
- **Ignore PDF**: Keeps repository smaller

To ignore PDF, uncomment this line in `.gitignore`:
```gitignore
# *.pdf
```

## Workflow Recommendations

### Daily Writing

```bash
# Terminal 1: Auto-recompile
cd docs/
make watch

# Terminal 2: View PDF
make view

# Edit introduction.tex in your editor
# PDF updates automatically!
```

### Before Team Meeting

```bash
make rebuild    # Fresh build
make view       # Check result
# Share introduction.pdf with team
```

### Before Submission

```bash
make cleanall   # Remove all files
make            # Fresh compilation
make wordcount  # Check word count
make view       # Final review
# Submit introduction.pdf
```

## Tips

### Faster Iteration

Use `make quick` while writing:
```bash
make quick    # Fast compilation
```

Use full `make` before submission:
```bash
make          # Full compilation with bibliography
```

### Check Progress

```bash
make wordcount    # Count words
```

Target: 6-8 pages (excluding references)

### Multiple Edits

Use watch mode for continuous compilation:
```bash
make watch
```

Edit, save, and see changes immediately!

## Documentation

For more details, see:

- **`README.md`** - Comprehensive documentation
- **`MAKEFILE_GUIDE.md`** - Quick reference guide
- **`make help`** - Command-line help

## Summary

**What you have now**:
- ✅ Automated build system (Makefile)
- ✅ One-command compilation (`make`)
- ✅ Quick compilation for drafts (`make quick`)
- ✅ Auto-recompile on save (`make watch`)
- ✅ Easy cleanup (`make clean`)
- ✅ Comprehensive documentation

**How to use**:
1. First time: `make check` → `make` → `make view`
2. While writing: `make watch` or `make quick`
3. Before submission: `make rebuild` → `make view`

**Need help?**
- `make help` - Show all commands
- `README.md` - Full documentation
- `MAKEFILE_GUIDE.md` - Quick guide

Good luck with your documentation! 📄✨
