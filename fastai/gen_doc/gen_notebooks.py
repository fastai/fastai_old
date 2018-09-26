import pkgutil, inspect, sys,os, importlib,json,enum,warnings,nbformat,re
from IPython.core.display import display, Markdown
from nbconvert.preprocessors import ExecutePreprocessor
import nbformat.sign
from pathlib import Path
from .core import *
from .nbdoc import *

__all__ = ['create_module_page', 'generate_all', 'update_module_page', 'update_all']

def get_empty_notebook():
    "a default notbook with the minimum metadata"
    #TODO: check python version and nbformat
    return {'metadata': {'kernelspec': {'display_name': 'Python 3',
                                        'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'codemirror_mode': {'name': 'ipython', 'version': 3},
                         'file_extension': '.py',
                         'mimetype': 'text/x-python',
                         'name': 'python',
                         'nbconvert_exporter': 'python',
                         'pygments_lexer': 'ipython3',
                         'version': '3.6.6'}},
            'nbformat': 4,
            'nbformat_minor': 2}

def get_md_cell(source, metadata=None):
    "a markdown cell containing the source text"
    return {'cell_type': 'markdown',
            'metadata': {} if metadata is None else metadata,
            'source': source}

def get_empty_cell(ctype='markdown'):
    "an empty cell of type ctype"
    return {'cell_type': ctype, 'metadata': {}, 'source': []}

def get_code_cell(code, hidden=False):
    "a code cell containing the code"
    return {'cell_type' : 'code',
            'execution_count': 0,
            'metadata' : {'hide_input': hidden, 'trusted':True},
            'source' : code,
            'outputs': []}

def get_doc_cell(ft_name):
    "a code cell with the command to show the doc of a given function"
    code = f"show_doc({ft_name})"
    return get_code_cell(code, True)

def is_enum(cls):
    "True if cls is a enumerator class"
    return cls == enum.Enum or cls == enum.EnumMeta

def get_inner_fts(elt):
    "List the inner functions of a class"
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name[:2] == '__': continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.ismethod(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_global_vars(mod):
    "Returns globally assigned variables"
    # https://stackoverflow.com/questions/8820276/docstring-for-variable/31764368#31764368
    import ast,re
    with open(mod.__file__, 'r') as f: fstr = f.read()
    flines = fstr.splitlines()
    d = {}
    for node in ast.walk(ast.parse(fstr)):
        if isinstance(node,ast.Assign) and hasattr(node.targets[0], 'id'):
            key,lineno = node.targets[0].id,node.targets[0].lineno-1
            codestr = flines[lineno]
            if re.match(f"^{key}\s*=\s*.*", codestr): # only top level assignment
                d[key] = f'`{codestr}` {get_source_link(mod, lineno)}'
    return d

def get_source_link(mod, lineno) -> str:
    "Returns link to line number in source code"
    fpath = os.path.realpath(inspect.getfile(mod))
    relpath = os.path.relpath(fpath, os.getcwd())
    link = f"{relpath}#L{lineno}"
    return f'<div style="text-align: right"><a href="{link}">[source]</a></div>'

def execute_nb(fname):
    "Execute notebook `fname`"
    # Any module used in the notebook that isn't inside must be in the same directory as this script

    with open(fname) as f: nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})
    with open(fname, 'wt') as f: nbformat.write(nb, f)
    nbformat.sign.NotebookNotary().sign(nb)

def _symbol_skeleton(name): return [get_doc_cell(name), get_md_cell(f"`{name}`")]

def create_module_page(mod, dest_path, force=False):
    "Creates the documentation notebook for module `mod_name` in path `dest_path`"
    nb = get_empty_notebook()
    mod_name = mod.__name__
    strip_name = strip_fastai(mod_name)
    init_cell = [get_md_cell(f'# {strip_name}'), get_md_cell('Type an introduction of the package here.')]
    cells = [get_code_cell(f'from fastai.gen_doc.nbdoc import *\nfrom {mod_name} import * ', True)]

    gvar_map = get_global_vars(mod)
    if gvar_map: cells.append(get_md_cell('### Global Variable Definitions:'))
    for name in get_exports(mod):
        if name in gvar_map: cells.append(get_md_cell(gvar_map[name]))

    for ft_name in get_ft_names(mod):
        if not hasattr(mod, ft_name):
            warnings.warn(f"Module {strip_name} doesn't have a function named {ft_name}.")
            continue
        cells += _symbol_skeleton(ft_name)
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            in_ft_names.sort(key = str.lower)
            for name in in_ft_names:
                cells += _symbol_skeleton(name)
    nb['cells'] = init_cell + cells
    json.dump(nb, open(os.path.join(dest_path,f'{strip_name}.ipynb'),
                       'w' if force else 'x'))
    execute_nb(os.path.join(dest_path,f'{strip_name}.ipynb'))

_default_exclude = ['.ipynb_checkpoints', '__pycache__']

def get_module_names(path_dir, exclude=None):
    if exclude is None: exclude = _default_exclude
    "Searches a given directory and returns all the modules contained inside"
    files = path_dir.glob('*')
    res = []
    for f in files:
        if f.name[-3:] == '.py': res.append(f'{path_dir.name}.{f.name[:-3]}')
        elif f.is_dir() and not f.name in exclude:
            res += [f'{path_dir.name}.{name}' for name in get_module_names(f)]
    return res

def generate_all(pkg_name, dest_path, exclude=None):
    "Generate the documentation for all the modules in `pkg_name`"
    if exclude is None: exclude = _default_exclude
    mod_files = get_module_names(Path(pkg_name), exclude)
    for mod_name in mod_files:
        print(f'Generating module page of {mod_name}')
        create_module_page(mod_name, dest_path)

def read_nb(fname):
    "Read a notebook and returns its corresponding json"
    with open(fname,'r') as f: return nbformat.reads(f.read(), as_version=4)

def read_nb_content(cells, mod_name):
    "Builds a dictionary containing the position of the cells giving the document for functions in a notebook"
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'code':
            match = re.match(r"(.*)show_doc\(([\w\.]*)", cell['source'])
            if match is not None: doc_fns[match.groups()[1]] = i
    return doc_fns

def read_nb_types(cells):
    doc_fns = {}
    for i, cell in enumerate(cells):
        if cell['cell_type'] == 'markdown':
            match = re.match(r"^`?(\w*)\s*=\s*", cell['source'])
            if match is not None: doc_fns[match.group(1)] = i
    return doc_fns

def get_insert_idx(pos_dict, name):
    "Return the position to insert a given function doc in a notebook"
    keys,i = list(pos_dict.keys()),0
    while i < len(keys) and str.lower(keys[i]) < str.lower(name): i+=1
    if i == len(keys): return -1
    else:              return pos_dict[keys[i]]

def update_pos(pos_dict, start_key, nbr=2):
    "Updates the position dictionary by moving all positions after start_ket by nbr"
    for key,idx in pos_dict.items():
        if str.lower(key) >= str.lower(start_key): pos_dict[key] += nbr
    return pos_dict

def insert_cells(cells, pos_dict, ft_name):
    "Insert the function doc cells of a function in the list of cells at their correct postition and updates the position dictionary"
    idx = get_insert_idx(pos_dict, ft_name)
    if idx == -1: cells += [get_doc_cell(ft_name), get_empty_cell()]
    else:
        cells.insert(idx, get_doc_cell(ft_name))
        cells.insert(idx+1, get_empty_cell())
        pos_dict = update_pos(pos_dict, ft_name, 2)
    return cells, pos_dict

def update_module_page(mod, dest_path):
    "Updates the documentation notebook of a given module"
    strip_name = strip_fastai(mod.__name__)
    nb = read_nb(os.path.join(dest_path,f'{strip_name}.ipynb'))
    cells = nb['cells']

    type_dict = read_nb_types(cells)
    gvar_map = get_global_vars(mod)
    for name in get_exports(mod):
        if name not in gvar_map: continue
        code = gvar_map[name]
        if name in type_dict: cells[type_dict[name]] = get_md_cell(code)
        else: cells.append(get_md_cell(code))

    pos_dict = read_nb_content(cells, strip_name)
    for ft_name in get_ft_names(mod):
        if not hasattr(mod, ft_name):
            warnings.warn(f"Module {strip_name} doesn't have a function named {ft_name}.")
            continue

        if ft_name not in pos_dict.keys():
            cells, pos_dict = insert_cells(cells, pos_dict, ft_name)
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            in_ft_names.sort(key = str.lower)
            for name in in_ft_names:
                if name not in pos_dict.keys():
                    cells, pos_dict = insert_cells(cells, pos_dict, name)
    nb['cells'] = cells
    json.dump(nb, open(os.path.join(dest_path,f'{strip_name}.ipynb'),'w'))
    #execute_nb(os.path.join(dest_path,f'{mod_name}.ipynb'))

def update_all(pkg_name, dest_path, exclude=('.ipynb_checkpoints', '__pycache__')):
    "Updates all the notebooks in `pkg_name`"
    mod_files = get_module_names(Path(pkg_name), exclude)
    for f in mod_files:
        print(f'Updating module page of {f}')
        update_module_page(f, dest_path)
