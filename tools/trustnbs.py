import nbformat.sign, sys
from pathlib import Path

# This script signs all notebooks in the doc_src directory as trusted
path = Path(sys.argv[1] or 'doc_src')
fnames = [file for file in path.glob("*.ipynb")]
for fname in fnames:
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
        nbformat.sign.NotebookNotary().sign(nb)
