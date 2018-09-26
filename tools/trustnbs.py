import glob, nbformat.sign

# This script signs all notebooks in the doc_src directory as trusted
fnames = [file for file in glob.glob("../doc_src/*.ipynb")]
for fname in fnames:
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
        nbformat.sign.NotebookNotary().sign(nb)
