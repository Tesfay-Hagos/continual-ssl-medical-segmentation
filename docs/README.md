# Documentation Folder

This folder contains the LaTeX documentation for the project.

## Files

- **`introduction.tex`** - Main LaTeX document (12 pages)
- **`Makefile`** - Build automation for compiling LaTeX to PDF
- **`README.md`** - This file

## Quick Start

### Compile the PDF

```bash
# Full compilation (recommended for first build)
make

# Quick compilation (faster, for drafts)
make quick

# Compile and open the PDF
make view
```

### Clean Up

```bash
# Remove auxiliary files (keep PDF)
make clean

# Remove all files including PDF
make cleanall

# Rebuild from scratch
make rebuild
```

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make` | Full compilation with bibliography (4 passes) |
| `make quick` | Quick compilation (1 pass, no bibliography) |
| `make view` | Compile and open the PDF |
| `make clean` | Remove auxiliary files (keep PDF) |
| `make cleanall` | Remove all generated files including PDF |
| `make rebuild` | Clean and rebuild from scratch |
| `make check` | Check if LaTeX tools are installed |
| `make watch` | Watch for changes and auto-recompile |
| `make wordcount` | Count words in the PDF |
| `make help` | Show help message |

## Requirements

### Linux/macOS

```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS (using Homebrew)
brew install --cask mactex

# Fedora
sudo dnf install texlive-scheme-full
```

### Windows

Download and install:
- **MiKTeX**: https://miktex.org/download
- **TeX Live**: https://www.tug.org/texlive/

Or use **Overleaf** (online, no installation needed): https://www.overleaf.com/

## Usage Examples

### First Time Compilation

```bash
cd continual_self_supervised_learning/docs/

# Check if LaTeX is installed
make check

# Full compilation
make

# Open the PDF
make view
```

### During Writing

```bash
# Quick compile after making changes
make quick

# Or use watch mode (auto-recompile on save)
make watch
```

### Before Submission

```bash
# Clean rebuild to ensure everything is correct
make rebuild

# Check word count
make wordcount

# Open final PDF
make view
```

## Compilation Process

The full `make` command performs 4 passes:

1. **Pass 1**: Initial compilation
   - Generates `.aux` file with references
   
2. **Pass 2**: BibTeX
   - Processes bibliography
   - Generates `.bbl` file
   
3. **Pass 3**: Resolve references
   - Updates citations and references
   
4. **Pass 4**: Final compilation
   - Ensures all references are correct

## Troubleshooting

### "make: command not found"

**Linux/macOS**: `make` should be pre-installed. If not:
```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# macOS
xcode-select --install
```

**Windows**: Use Git Bash or WSL, or compile manually:
```bash
pdflatex introduction.tex
bibtex introduction
pdflatex introduction.tex
pdflatex introduction.tex
```

### "pdflatex: command not found"

Install LaTeX distribution (see Requirements section above).

### Compilation Errors

```bash
# Check the log file
cat introduction.log

# Clean and try again
make cleanall
make
```

### PDF Viewer Not Opening

Edit the `Makefile` and change the `VIEWER` variable:

```makefile
# Linux
VIEWER = xdg-open      # Default
VIEWER = evince        # Alternative
VIEWER = okular        # Alternative

# macOS
VIEWER = open

# Windows (Git Bash)
VIEWER = start
```

## Alternative: Using Overleaf

If you prefer not to install LaTeX locally:

1. Go to https://www.overleaf.com/
2. Create a free account
3. Click "New Project" → "Upload Project"
4. Upload `introduction.tex`
5. Click "Recompile" to generate PDF
6. Download PDF when done

## File Structure After Compilation

```
docs/
├── introduction.tex       # Source file
├── introduction.pdf       # Generated PDF ✓
├── introduction.aux       # Auxiliary file (can delete)
├── introduction.log       # Log file (can delete)
├── introduction.bbl       # Bibliography (can delete)
├── introduction.blg       # Bibliography log (can delete)
├── introduction.out       # Hyperref output (can delete)
├── Makefile              # Build script
└── README.md             # This file
```

Use `make clean` to remove auxiliary files.

## Tips

### Fast Iteration

When writing, use `make quick` for faster compilation:
```bash
# Edit introduction.tex
make quick
# Check PDF
# Repeat
```

Before final submission, use full `make` to ensure bibliography is correct.

### Auto-Recompile

Use watch mode to automatically recompile on save:
```bash
make watch
```

Press Ctrl+C to stop.

**Note**: Requires `inotify-tools` on Linux:
```bash
sudo apt-get install inotify-tools
```

### Word Count

Check if you're within the 6-8 page limit:
```bash
make wordcount
```

### Version Control

Add to `.gitignore`:
```
*.aux
*.log
*.out
*.bbl
*.blg
*.synctex.gz
*.fdb_latexmk
*.fls
```

Keep the PDF in version control or not, depending on preference.

## Common Workflows

### Daily Writing

```bash
# Start writing session
cd docs/
make watch    # Auto-recompile on save

# Edit introduction.tex in your editor
# PDF updates automatically

# When done, press Ctrl+C
```

### Before Team Meeting

```bash
# Generate fresh PDF
make rebuild

# Check word count
make wordcount

# Open PDF
make view
```

### Before Submission

```bash
# Clean rebuild
make cleanall
make

# Verify PDF
make view

# Check for errors
cat introduction.log | grep -i error

# If no errors, submit introduction.pdf
```

## Getting Help

### Makefile Help

```bash
make help
```

### LaTeX Help

- **LaTeX Documentation**: https://www.latex-project.org/help/documentation/
- **Overleaf Tutorials**: https://www.overleaf.com/learn
- **TeX Stack Exchange**: https://tex.stackexchange.com/

### Project Help

See the main project documentation:
- `../README.md` - Project overview
- `../QUICK_START.md` - Implementation guide
- `../NAVIGATION_GUIDE.md` - Where to find what

## Summary

**Quick Reference**:
```bash
make              # Full build
make quick        # Fast build
make view         # Build and view
make clean        # Clean up
make help         # Show help
```

**First time**: `make check` → `make` → `make view`

**While writing**: `make quick` or `make watch`

**Before submission**: `make rebuild` → `make view`

Good luck with your documentation! 📄✨
