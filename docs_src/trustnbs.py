import os, glob, nbformat.sign

# Search for all jupyter files in the directory
fnames = [file for file in glob.glob("*.ipynb")]

# Iterate over notebooks and sign each of them as trusted
for fname in fnames:
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
        nbformat.sign.NotebookNotary().sign(nb)
