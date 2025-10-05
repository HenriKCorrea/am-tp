# --- Environment Configuration ---
# Prepend the project's texmf structure to the TEXINPUTS search path.
# This allows LaTeX to find custom packages and classes.
# The '//' recursively searches all subdirectories.
# The trailing colon (:) appends the default system paths.
$ENV{'TEXINPUTS'} = '../texmf/tex/latex//:' . ($ENV{'TEXINPUTS'} || '');
$ENV{'BIBINPUTS'} = '../texmf/bibtex/bib//:' . ($ENV{'BIBINPUTS'} || '');
$ENV{'BSTINPUTS'} = '../texmf/bibtex/bst//:' . ($ENV{'BSTINPUTS'} || '');

# --- Build Configuration ---
$aux_dir = 'build';        # Path where temporary compiler files are generated
# $out_dir = 'output';       # Path where desired artifacts are generated. Defaults to '' (same as source)
$pdf_mode = 1;             # Generate PDF (1: PDF, 0: DVI, 2: PS)

# --- Output configuration ---
# synctex=1: Enable synchronization with editors (e.g., VSCode <==> PDF viewer)
# file-line-error: Show errors with line numbers
# nonstopmode: Continue compiling even if errors are encountered
$pdflatex = 'pdflatex -synctex=1 -file-line-error -interaction=nonstopmode %O %S';

# --- Clean extensions ---
# Define which files to clean when running cleanup commands (e.g.: latexmk -c)
$clean_ext = 'auxlock figlist makefile synctex.gz synctex.gz(busy) run.xml tex.bak bbl bcf fls log fdb_latexmk run.xml';
