"Script to generate notebooks and update html"
import argparse
from fastai.gen_doc.gen_notebooks import generate_all, update_all, update_module_page, import_mod
from fastai.gen_doc.convert2html import convert_all, convert_nb
from pathlib import Path
import fire

def resolve_path(path):
    "Creates absolute path if relative is provided"
    p = Path(path)
    if p.is_absolute(): return p
    return Path.cwd()/path

def get_module_name(source_path):
    "Finds module given a source path. Assumes it belongs to fastai directory"
    fpath = Path(source_path).resolve()
    fastai_idx = list(reversed(fpath.parts)).index('fastai')
    dirpath = fpath.parents[fastai_idx]
    relpath = fpath.relative_to(dirpath)
    return '.'.join(relpath.with_suffix('').parts)

def update_notebooks(source_path=None, dest_path=None, update_html=False, update_nb=True, html_path=None, overwrite=False):
    # Find default folders relative to this file
    fpath = Path(__file__).resolve()
    fastai_idx = list(reversed(fpath.parts)).index('fastai')
    dirpath = fpath.parents[fastai_idx] # should return 'fastai_pytorch'
    if source_path is None: source_path = dirpath/'fastai'
    else: source_path = resolve_path(source_path)
    if dest_path is None: dest_path = dirpath/'docs_src'
    else: dest_path = resolve_path(dest_path)
    if html_path is None: html_path = dirpath/'docs'
    else: html_path = resolve_path(html_path)

    if Path(source_path).is_file():
        mod = import_mod(get_module_name(source_path))
        if not mod: return print('Could not find module for path:', source_path)
        if update_nb: doc_path = update_module_page(mod, dest_path)
        print('Updating html:', doc_path, dest_path)
        if update_html: convert_nb(doc_path, dest_path)
    else:
        if overwrite and update_nb: generate_all(source_path, dest_path)
        elif update_nb: update_all(source_path, dest_path)
        if update_html: convert_all(dest_path, html_path)

if __name__ == '__main__': fire.Fire(update_notebooks)
