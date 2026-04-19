# Makefile for LaTeX document compilation
# Usage:
#   make          - Compile the PDF (full build with bibliography)
#   make quick    - Quick compile (single pass, no bibliography)
#   make clean    - Remove all generated files
#   make view     - Compile and open the PDF
#   make help     - Show this help message

# Configuration
MAIN = introduction
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
BIB = $(MAIN).bib

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# PDF viewer (adjust based on your system)
# Linux: xdg-open, evince, okular
# macOS: open
# Windows: start
VIEWER = xdg-open

# Colors for output
COLOR_RESET = \033[0m
COLOR_GREEN = \033[32m
COLOR_YELLOW = \033[33m
COLOR_BLUE = \033[34m

# Default target
.PHONY: all
all: $(PDF)

# Full build with bibliography
$(PDF): $(TEX)
	@echo "$(COLOR_BLUE)Starting full LaTeX compilation...$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Pass 1: Initial compilation$(COLOR_RESET)"
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "$(COLOR_YELLOW)Pass 2: Running BibTeX$(COLOR_RESET)"
	-$(BIBTEX) $(MAIN)
	@echo "$(COLOR_YELLOW)Pass 3: Resolving references$(COLOR_RESET)"
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "$(COLOR_YELLOW)Pass 4: Final compilation$(COLOR_RESET)"
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "$(COLOR_GREEN)✓ PDF generated successfully: $(PDF)$(COLOR_RESET)"

# Quick compile (single pass, no bibliography)
.PHONY: quick
quick: $(TEX)
	@echo "$(COLOR_BLUE)Quick compilation (single pass)...$(COLOR_RESET)"
	$(LATEX) $(LATEX_FLAGS) $(TEX)
	@echo "$(COLOR_GREEN)✓ Quick compile complete: $(PDF)$(COLOR_RESET)"

# Compile and open the PDF
.PHONY: view
view: $(PDF)
	@echo "$(COLOR_BLUE)Opening PDF...$(COLOR_RESET)"
	$(VIEWER) $(PDF) &

# Clean all generated files
.PHONY: clean
clean:
	@echo "$(COLOR_YELLOW)Cleaning generated files...$(COLOR_RESET)"
	rm -f $(MAIN).aux
	rm -f $(MAIN).log
	rm -f $(MAIN).out
	rm -f $(MAIN).toc
	rm -f $(MAIN).lof
	rm -f $(MAIN).lot
	rm -f $(MAIN).bbl
	rm -f $(MAIN).blg
	rm -f $(MAIN).synctex.gz
	rm -f $(MAIN).fdb_latexmk
	rm -f $(MAIN).fls
	rm -f $(MAIN).nav
	rm -f $(MAIN).snm
	rm -f $(MAIN).vrb
	@echo "$(COLOR_GREEN)✓ Cleanup complete$(COLOR_RESET)"

# Clean everything including PDF
.PHONY: cleanall
cleanall: clean
	@echo "$(COLOR_YELLOW)Removing PDF...$(COLOR_RESET)"
	rm -f $(PDF)
	@echo "$(COLOR_GREEN)✓ All files removed$(COLOR_RESET)"

# Rebuild from scratch
.PHONY: rebuild
rebuild: cleanall all

# Check if LaTeX is installed
.PHONY: check
check:
	@echo "$(COLOR_BLUE)Checking LaTeX installation...$(COLOR_RESET)"
	@which $(LATEX) > /dev/null && echo "$(COLOR_GREEN)✓ pdflatex found$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗ pdflatex not found$(COLOR_RESET)"
	@which $(BIBTEX) > /dev/null && echo "$(COLOR_GREEN)✓ bibtex found$(COLOR_RESET)" || echo "$(COLOR_YELLOW)✗ bibtex not found$(COLOR_RESET)"

# Watch for changes and recompile (requires inotify-tools on Linux)
.PHONY: watch
watch:
	@echo "$(COLOR_BLUE)Watching for changes... (Press Ctrl+C to stop)$(COLOR_RESET)"
	@while true; do \
		inotifywait -e modify $(TEX) 2>/dev/null && make quick; \
	done

# Count words in the document
.PHONY: wordcount
wordcount: $(PDF)
	@echo "$(COLOR_BLUE)Counting words...$(COLOR_RESET)"
	@pdftotext $(PDF) - | wc -w
	@echo "$(COLOR_GREEN)words (approximate)$(COLOR_RESET)"

# Show help
.PHONY: help
help:
	@echo "$(COLOR_BLUE)LaTeX Makefile - Available targets:$(COLOR_RESET)"
	@echo ""
	@echo "  $(COLOR_GREEN)make$(COLOR_RESET)          - Full compilation with bibliography (4 passes)"
	@echo "  $(COLOR_GREEN)make quick$(COLOR_RESET)    - Quick compilation (1 pass, no bibliography)"
	@echo "  $(COLOR_GREEN)make view$(COLOR_RESET)     - Compile and open the PDF"
	@echo "  $(COLOR_GREEN)make clean$(COLOR_RESET)    - Remove auxiliary files (keep PDF)"
	@echo "  $(COLOR_GREEN)make cleanall$(COLOR_RESET) - Remove all generated files including PDF"
	@echo "  $(COLOR_GREEN)make rebuild$(COLOR_RESET)  - Clean and rebuild from scratch"
	@echo "  $(COLOR_GREEN)make check$(COLOR_RESET)    - Check if LaTeX tools are installed"
	@echo "  $(COLOR_GREEN)make watch$(COLOR_RESET)    - Watch for changes and auto-recompile"
	@echo "  $(COLOR_GREEN)make wordcount$(COLOR_RESET) - Count words in the PDF"
	@echo "  $(COLOR_GREEN)make help$(COLOR_RESET)     - Show this help message"
	@echo ""
	@echo "$(COLOR_YELLOW)Configuration:$(COLOR_RESET)"
	@echo "  Main file: $(TEX)"
	@echo "  Output:    $(PDF)"
	@echo "  Compiler:  $(LATEX)"
	@echo ""
	@echo "$(COLOR_YELLOW)Examples:$(COLOR_RESET)"
	@echo "  make              # Full build"
	@echo "  make quick        # Quick build for drafts"
	@echo "  make view         # Build and view"
	@echo "  make clean        # Clean up"
	@echo ""

# Phony targets (not actual files)
.PHONY: all quick view clean cleanall rebuild check watch wordcount help
