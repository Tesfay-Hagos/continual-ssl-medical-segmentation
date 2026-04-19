# Makefile Quick Guide

## TL;DR

```bash
cd docs/
make              # Compile PDF
make view         # Compile and open PDF
make clean        # Clean up auxiliary files
```

## All Commands

| Command | What It Does | When to Use |
|---------|--------------|-------------|
| `make` | Full compilation (4 passes) | First build, before submission |
| `make quick` | Fast compilation (1 pass) | While writing, quick checks |
| `make view` | Compile and open PDF | To see the result |
| `make clean` | Remove auxiliary files | Clean up workspace |
| `make cleanall` | Remove all files including PDF | Start fresh |
| `make rebuild` | Clean and rebuild | Before final submission |
| `make check` | Check LaTeX installation | Troubleshooting |
| `make watch` | Auto-recompile on save | Active writing session |
| `make wordcount` | Count words in PDF | Check page limit |
| `make help` | Show help message | Learn more commands |

## Common Workflows

### First Time Setup

```bash
cd continual_self_supervised_learning/docs/

# 1. Check if LaTeX is installed
make check

# 2. Compile the PDF
make

# 3. Open the PDF
make view
```

### While Writing

```bash
# Option 1: Manual recompile after each change
make quick

# Option 2: Auto-recompile on save (recommended)
make watch
# Edit introduction.tex in your editor
# PDF updates automatically
# Press Ctrl+C when done
```

### Before Submission

```bash
# 1. Clean rebuild
make rebuild

# 2. Check word count
make wordcount

# 3. View final PDF
make view

# 4. Submit introduction.pdf
```

### Clean Up

```bash
# Remove auxiliary files (keep PDF)
make clean

# Remove everything including PDF
make cleanall
```

## What Each Pass Does

### Full Compilation (`make`)

1. **Pass 1**: `pdflatex introduction.tex`
   - Initial compilation
   - Generates `.aux` file with reference placeholders

2. **Pass 2**: `bibtex introduction`
   - Processes bibliography
   - Generates `.bbl` file with formatted citations

3. **Pass 3**: `pdflatex introduction.tex`
   - Resolves citations and references
   - Updates cross-references

4. **Pass 4**: `pdflatex introduction.tex`
   - Final compilation
   - Ensures all references are correct

### Quick Compilation (`make quick`)

1. **Pass 1**: `pdflatex introduction.tex`
   - Single pass
   - Fast but may have incorrect references
   - Good for drafts

## Installation Requirements

### Check Installation

```bash
make check
```

Should show:
```
✓ pdflatex found
✓ bibtex found
```

### Install LaTeX

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
- Or use Overleaf (online): https://www.overleaf.com/

### Install Make

**Linux**: Usually pre-installed

**macOS**:
```bash
xcode-select --install
```

**Windows**: Use Git Bash or WSL

## Troubleshooting

### "make: command not found"

**Solution**: Install build tools or use manual compilation:
```bash
pdflatex introduction.tex
bibtex introduction
pdflatex introduction.tex
pdflatex introduction.tex
```

### "pdflatex: command not found"

**Solution**: Install LaTeX (see Installation Requirements above)

### Compilation Errors

**Solution 1**: Check the log file
```bash
cat introduction.log | grep -i error
```

**Solution 2**: Clean and rebuild
```bash
make cleanall
make
```

**Solution 3**: Use Overleaf
- Upload to https://www.overleaf.com/
- Compile online

### PDF Viewer Not Opening

**Solution**: Edit `Makefile` line 23:
```makefile
# Change this line based on your system:
VIEWER = xdg-open    # Linux (default)
VIEWER = evince      # Linux (alternative)
VIEWER = okular      # Linux (alternative)
VIEWER = open        # macOS
VIEWER = start       # Windows
```

### Watch Mode Not Working

**Solution**: Install inotify-tools (Linux only)
```bash
sudo apt-get install inotify-tools
```

Or use manual recompilation:
```bash
make quick
```

## Tips and Tricks

### Faster Iteration

While writing, use `make quick` instead of `make`:
```bash
# Edit introduction.tex
make quick
# Check PDF
# Repeat
```

Before submission, use full `make` to ensure bibliography is correct.

### Auto-Recompile

Use watch mode for automatic recompilation:
```bash
make watch
```

Edit `introduction.tex` in your editor, and the PDF updates automatically!

### Check Progress

Count words to ensure you're within the 6-8 page limit:
```bash
make wordcount
```

### Multiple Terminals

**Terminal 1**: Watch mode
```bash
make watch
```

**Terminal 2**: View PDF
```bash
make view
```

Edit in your editor, and see changes in real-time!

## File Structure

### Before Compilation

```
docs/
├── introduction.tex    # Source file
├── Makefile           # Build script
└── README.md          # Documentation
```

### After Compilation

```
docs/
├── introduction.tex    # Source file
├── introduction.pdf    # Generated PDF ✓
├── introduction.aux    # Auxiliary (can delete)
├── introduction.log    # Log (can delete)
├── introduction.bbl    # Bibliography (can delete)
├── introduction.blg    # Bibliography log (can delete)
├── introduction.out    # Hyperref (can delete)
├── Makefile           # Build script
└── README.md          # Documentation
```

Use `make clean` to remove auxiliary files.

## Makefile Customization

### Change PDF Viewer

Edit `Makefile` line 23:
```makefile
VIEWER = your-preferred-viewer
```

### Change Compiler Flags

Edit `Makefile` line 20:
```makefile
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error
```

### Add Custom Targets

Add to `Makefile`:
```makefile
.PHONY: mycommand
mycommand:
	@echo "Running my custom command"
	# Your commands here
```

Use: `make mycommand`

## Comparison: Make vs Manual

| Aspect | `make` | Manual |
|--------|--------|--------|
| **Speed** | Fast (cached) | Slower |
| **Convenience** | One command | Multiple commands |
| **Error Handling** | Automatic | Manual |
| **Cleanup** | `make clean` | Manual deletion |
| **Consistency** | Always correct | Easy to forget steps |

**Recommendation**: Use `make` for convenience and consistency.

## Alternative: Overleaf

If you don't want to install LaTeX locally:

1. Go to https://www.overleaf.com/
2. Create free account
3. Upload `introduction.tex`
4. Click "Recompile"
5. Download PDF

**Pros**:
- No installation needed
- Works on any device
- Collaborative editing

**Cons**:
- Requires internet
- Free tier has limitations
- Less control

## Summary

**Quick Reference**:
```bash
make              # Full build (first time, before submission)
make quick        # Fast build (while writing)
make view         # Build and view
make watch        # Auto-recompile (active writing)
make clean        # Clean up
make help         # Show help
```

**Typical Workflow**:
1. First time: `make check` → `make` → `make view`
2. While writing: `make watch` (or `make quick`)
3. Before submission: `make rebuild` → `make view`

**Need Help?**
- `make help` - Show all commands
- `docs/README.md` - Detailed documentation
- `make check` - Check installation

Good luck! 📄✨
