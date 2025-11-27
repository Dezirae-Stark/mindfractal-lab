# ============================================================================
# Makefile for Fractal Consciousness Book
# ============================================================================

# Main document
MAIN = fractal_consciousness_book
TEX = pdflatex
BIBTEX = bibtex

# Directories
CHAPTERS = chapters
IMAGES = images

# Output directory (optional)
OUTDIR = build

# ============================================================================
# Targets
# ============================================================================

.PHONY: all pdf clean distclean view chapters help

all: pdf

# Build PDF (run pdflatex twice for references)
pdf: $(MAIN).tex
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Full build with bibliography (if using bibtex)
full: $(MAIN).tex
	$(TEX) $(MAIN).tex
	$(BIBTEX) $(MAIN) || true
	$(TEX) $(MAIN).tex
	$(TEX) $(MAIN).tex

# Build with xelatex (for Unicode support)
xelatex: $(MAIN).tex
	xelatex $(MAIN).tex
	xelatex $(MAIN).tex

# Build with lualatex (alternative engine)
lualatex: $(MAIN).tex
	lualatex $(MAIN).tex
	lualatex $(MAIN).tex

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.out *.toc *.lof *.lot
	rm -f *.bbl *.blg *.bcf *.run.xml
	rm -f *.idx *.ind *.ilg
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	rm -f $(CHAPTERS)/*.aux

# Clean everything including PDF
distclean: clean
	rm -f $(MAIN).pdf

# View PDF (Linux)
view: pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "PDF viewer not found"

# List chapters
chapters:
	@echo "Chapters:"
	@ls -1 $(CHAPTERS)/*.tex 2>/dev/null || echo "No chapters found"

# Help
help:
	@echo "Available targets:"
	@echo "  make pdf      - Build PDF with pdflatex (default)"
	@echo "  make full     - Build PDF with bibliography"
	@echo "  make xelatex  - Build with xelatex engine"
	@echo "  make lualatex - Build with lualatex engine"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make distclean- Remove all generated files"
	@echo "  make view     - Build and open PDF"
	@echo "  make chapters - List chapter files"
	@echo "  make help     - Show this help"
