import re, nbformat, jupyter_contrib_nbextensions
from nbconvert.preprocessors import Preprocessor
from nbconvert import HTMLExporter
from traitlets.config import Config
from pathlib import Path
import fire

class HandleLinksPreprocessor(Preprocessor):
    def preprocess_cell(self, cell, resources, index):
        if 'source' in cell and cell.cell_type == "markdown":
            cell.source = re.sub(r"\((.*)\.ipynb(.*)\)",r"(\1.html\2)",cell.source)

        return cell, resources

exporter = HTMLExporter(Config())
#Loads the template to deal with hidden cells.
exporter.template_file = 'nbextensions.tpl'
exporter.template_path.append(jupyter_contrib_nbextensions.__path__[0] + '\\templates')
#Preprocesser that converts the .ipynb links in .html
exporter.register_preprocessor(HandleLinksPreprocessor, enabled=True)

def read_nb(fname):
    with open(fname,'r') as f:
        return nbformat.reads(f.read(), as_version=4)

def convert_nb(fname):
    nb = read_nb(fname)
    new_name = re.sub(r"(.*)\.ipynb",r"\1.html",str(fname))
    with open(new_name,'w') as f:
        f.write(exporter.from_notebook_node(nb)[0])

PATH = Path('Results/')
convert_nb(PATH/'fastai_v1.nb_001b.ipynb')

def convert_all(path):
    nb_files = path.glob('*.ipynb')
    for file in nb_files: convert_nb(file)

if __name__ == 'main': fire.Fire(convert_all)
