import pkgutil, inspect, sys,os, importlib,json,enum,warnings,nbformat,re
from IPython.core.display import display, Markdown
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

def get_empty_notebook():
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
    return {'cell_type': 'markdown',
            'metadata': {} if metadata is None else metadata,
            'source': source}

def get_empty_cell(ctype='markdown'):
    return {'cell_type': ctype,
            'metadata': {},
            'source': []}

def get_code_cell(code, hidden=False):
    return {'cell_type' : 'code',
            'execution_count': 0,
            'metadata' : {'hide_input': hidden, 'trusted':True},
            'source' : code,
            'outputs': []}

def get_doc_cell(mod_name, ft_name):
    code = f"show_doc_from_name('{mod_name}','{ft_name}')"
    return get_code_cell(code, True)

def is_enum(cls):
    return cls == enum.Enum or cls == enum.EnumMeta

def get_inner_fts(elt):
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name[:2] == '__': continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_ft_names(mod):
    fn_names = []
    for elt_name in dir(mod):
        elt = getattr(mod,elt_name)
        #This removes the files imported from elsewhere
        try:    fname = inspect.getfile(elt)
        except: continue
        if fname != mod.__file__: continue
        if inspect.isclass(elt) or inspect.isfunction(elt): fn_names.append(elt_name)
    return fn_names

def execute_nb(fname):
    with open(fname) as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {})
    with open(fname, 'wt') as f:
        nbformat.write(nb, f)

def create_module_page(module, pkg):
    nb = get_empty_notebook()
    init_cell = [get_md_cell(f'# {pkg}.{module}'), get_md_cell('Type an introduction of the package here.')]
    mod = importlib.import_module(f'.{module}',pkg)
    ft_names = mod.__all__ if hasattr(mod,'__all__') else get_ft_names(mod)
    ft_names.sort(key = str.lower)
    cells = [get_code_cell('from nbdoc import * ', True), get_code_cell(f'get_module_toc("{pkg}.{module}")', True)]
    for ft_name in ft_names:
        if not hasattr(mod, ft_name): 
            warnings.warn(f"Module {pkg}.{module} doesn't have a function named {ft_name}.")
            continue
        cells += [get_doc_cell(f'{pkg}.{module}',ft_name), get_empty_cell()]
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            in_ft_names.sort(key = str.lower)
            for name in in_ft_names:
                cells += [get_doc_cell(f'{pkg}.{module}', name), get_empty_cell()]
    nb['cells'] = init_cell + cells
    json.dump(nb, open(PATH/f'{pkg}.{module}.ipynb','w'))
    execute_nb(PATH/f'{pkg}.{module}.ipynb')

def generate_all(pkg_name):
    path = Path(pkg_name)
    mod_files = path.glob('*.py')
    for file in mod_files:
        mod_name = file.name[:-3]
        print(f'Generating module page of {mod_name}')
        create_module_page(mod_name, pkg_name)

def read_nb(fname):
    with open(fname,'r') as f:
        return nbformat.reads(f.read(), as_version=4)

def read_nb_content(nb, mod_name):
    doc_fns = {}
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            match = re.match(r"(.*)show_doc_from_name\('([^']*)','([^']*)'\)", cell['source'])
            if match is not None and match.groups()[1] == mod_name:
                doc_fns[match.groups()[2]] = i
    return doc_fns

def get_insert_idx(pos_dict, name):
    keys,i = list(pos_dict.keys()),0
    while i < len(keys) and str.lower(keys[i]) < str.lower(name): i+=1
    if i == len(keys): return -1
    else:              return pos_dict[keys[i]]

def update_pos(pos_dict, start_key, nb):
    for key in pos_dict.keys():
        if str.lower(key) >= str.lower(start_key): pos_dict[key] += nb
    return pos_dict

def insert_cells(cells, pos_dict, mod_name, ft_name):
    idx = get_insert_idx(pos_dict, ft_name)
    if idx == -1: cells += [get_doc_cell(mod_name,ft_name), get_empty_cell()]
    else:
        cells.insert(idx, get_doc_cell(mod_name, ft_name))
        cells.insert(idx+1, get_empty_cell())
        pos_dict = update_pos(pos_dict, ft_name, 2)
    return cells, pos_dict

def update_module_page(module, pkg):
    nb = read_nb(PATH/f'{pkg}.{module}.ipynb')
    mod = importlib.import_module(f'.{module}',pkg)
    mod = importlib.reload(mod)
    ft_names = mod.__all__ if hasattr(mod,'__all__') else get_ft_names(mod)
    ft_names.sort(key = str.lower)
    cells = nb['cells']
    pos_dict = read_nb_content(nb, f'{pkg}.{module}')
    for ft_name in ft_names:
        if not hasattr(mod, ft_name): 
            warnings.warn(f"Module {pkg}.{module} doesn't have a function named {ft_name}.")
            continue
            
        if ft_name not in pos_dict.keys():
            cells, pos_dict = insert_cells(cells, pos_dict, f'{pkg}.{module}', ft_name)
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            in_ft_names.sort(key = str.lower)
            for name in in_ft_names:
                if name not in pos_dict.keys():
                    cells, pos_dict = insert_cells(cells, pos_dict, in_ft_name)
    nb['cells'] = cells
    json.dump(nb, open(PATH/f'{pkg}.{module}.ipynb','w'))
    execute_nb(PATH/f'{pkg}.{module}.ipynb')

def update_all(pkg_name):
    path = Path(pkg_name)
    mod_files = path.glob('*.py')
    for file in mod_files:
        mod_name = file.name[:-3]
        print(f'Updating module page of {mod_name}')
        update_module_page(mod_name, pkg_name)

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default='.')
parser.add_argument('--update', action='store_true')
arg = parser.parse_args()

PATH = arg.path
if arg.update: update_all()
else: generate_all()
