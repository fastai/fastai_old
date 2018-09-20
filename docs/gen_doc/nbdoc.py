import inspect,importlib,enum,os,re
from IPython.core.display import display, Markdown, HTML
from typing import Dict, Any, AnyStr, List, Sequence, TypeVar, Tuple, Optional, Union
from .docstrings import *

__all__ = ['create_anchor', 'get_class_toc', 'get_fn_link', 'get_module_toc', 'show_doc', 'show_doc_from_name',
           'show_video', 'show_video_from_youtube']

def is_enum(cls):
    return cls == enum.Enum or cls == enum.EnumMeta

def link_type(argtype, include_bt:bool=False):
    """creates link to documentation"""
    arg_name = wrap_class(argtype)
    if include_bt: arg_name = f'`{arg_name}`'
    if is_fastai_class(argtype): return f'[{arg_name}]({get_fn_link(argtype)})'
    return arg_name

def format_ft_def(elt, full_name:str, ignore_first:bool=False) -> str:
    """Formats and links function definition to show in documentation"""
    args, defaults, formatted_types = get_arg_spec(elt)
    if ignore_first: args = args[1:]

    parsedargs = ''
    diff = len(args) - len(defaults) if defaults is not None else len(args)
    for i,arg in enumerate(args):
        parsedargs += f'<em>{arg}</em>'
        if arg in formatted_types: parsedargs += f': {formatted_types[arg]}'
        if i-diff >= 0: parsedargs += f'={defaults[i-diff]}'
        if i+1 < len(args): parsedargs += ', '
    parsedreturn = f" -> {formatted_types['return']}" if 'return' in formatted_types else ''

    return f'**{full_name}**({parsedargs}){parsedreturn}'

def is_fastai_class(t):
    """checks if belongs to fastai module"""
    if not inspect.getmodule(t): return False
    base_module = inspect.getmodule(t).__name__.split('.')[0]
    return base_module in ['fastai_v1', 'gen_doc', 'dev_nb']

def wrap_class(t):
    if hasattr(t, '__name__'): return t.__name__
    else: return str(t)

def get_arg_spec(elt):
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(elt)
    formatted_types = {k:link_type(v) for k,v in annotations.items()}
    return (args, defaults, formatted_types)

def get_enum_doc(elt, full_name:str) -> str:
    """return formatted enum documentation"""
    vals = ', '.join(elt.__members__.keys())
    doc = f'**{full_name}**:Enum = [{vals}]'
    return doc

def get_cls_doc(elt, full_name:str) -> str:
    """return class definition"""
    parent_class = inspect.getclasstree([elt])[-1][0][1][0]
    doc = f'<em>class</em> ' + format_ft_def(elt, full_name, ignore_first=True)
    if parent_class != object: doc += f' :: Inherits from ({link_type(parent_class, include_bt=True)})'
    return doc

def show_doc(elt, doc_string:bool=True, full_name:str=None, arg_comments:dict={}, alt_doc_string:str=''):
    """show doc for element. Supported types: class, function, method, and enum"""
    if full_name is None and hasattr(elt, '__name__'): full_name = elt.__name__
    if inspect.isclass(elt):
        if is_enum(elt.__class__): doc = get_enum_doc(elt, full_name)
        else:                      doc = get_cls_doc(elt, full_name)
    elif inspect.isfunction(elt):  doc = format_ft_def(elt, full_name)
    elif inspect.ismethod(elt):    doc = format_ft_def(elt, full_name)
    else: doc = f'doc definition not supported for {full_name}'
    link = f'<a id={full_name}></a>'

    if is_fastai_class(elt): doc += get_source_link(elt)
    if doc_string and (inspect.getdoc(elt) or arg_comments):
        doc += '\n' + format_docstring(elt, arg_comments, alt_doc_string)
    display(Markdown(link + doc))

def format_docstring(elt, arg_comments:dict={}, alt_doc_string:str='') -> str:
    """merges and formats the docstring definition with arg_comments and alt_doc_string"""
    parsed = ""
    doc = parse_docstring(inspect.getdoc(elt))
    description = alt_doc_string or doc['long_description'] or doc['short_description']
    if description: parsed += f'\n\n{link_docstring(elt, description)}'

    resolved_comments = {**doc.get('comments', {}), **arg_comments} # arg_comments takes priority
    args = inspect.getfullargspec(elt).args if not is_enum(elt.__class__) else elt.__members__.keys()
    if resolved_comments: parsed += '\n'
    for a in args:
        if a in resolved_comments: parsed += f'\n- *{a}*: {resolved_comments[a]}'

    return_comment = arg_comments.get('return') or doc.get('return')
    if return_comment: parsed += f'\n\n*return*: {return_comment}'
    return parsed

BT_REGEX = re.compile("`([^`]*)`")
def link_docstring(elt, docstring:str) -> str:
    """searches `docstring` for backticks and attempts to link those functions to respective documentation"""
    for m in BT_REGEX.finditer(docstring):
        if m.group(1) in globals():
            link_elt = globals()[m.group(1)]
            if is_fastai_class(link_elt):
                link = f'[{m.group(0)}]({get_fn_link(link_elt)})'
                docstring = docstring.replace(m.group(0), link)
    return docstring


def import_mod(mod_name:str):
    """returns module from `mod_name`"""
    splits = str.split(mod_name, '.')
    try: 
        if len(splits) > 1 : mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
        else: mod = importlib.import_module(mod_name)
        return mod
    except: 
        print(f"Module {mod_name} doesn't exist.")

def show_doc_from_name(mod_name, ft_name:str, doc_string:bool=True, arg_comments:dict={}, alt_doc_string:str=''):
    """shows documentation for `ft_name`. see `show_doc`"""
    mod = import_mod(mod_name)
    splits = str.split(ft_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a function named {split}.")
        elt = getattr(elt, split)
    show_doc(elt, doc_string, ft_name, arg_comments, alt_doc_string)

def get_ft_names(mod) -> List[str]:
    """retrieves all the functions of `mod`"""
    fn_names = []
    for elt_name in dir(mod):
        elt = getattr(mod,elt_name)
        #This removes the files imported from elsewhere
        try:    fname = inspect.getfile(elt)
        except: continue
        if fname != mod.__file__: continue
        if inspect.isclass(elt) or inspect.isfunction(elt): fn_names.append(elt_name)
    return fn_names

def get_inner_fts(elt) -> List[str]:
    """return methods belonging to class"""
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name[:2] == '__': continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_module_toc(mod_name):
    """displays table of contents for given `mod_name`"""
    mod = import_mod(mod_name)
    ft_names = mod.__all__ if hasattr(mod,'__all__') else get_ft_names(mod)
    ft_names.sort(key = str.lower)
    tabmat = ''
    for ft_name in ft_names:
        tabmat += f'- [{ft_name}](#{ft_name})\n'
        elt = getattr(mod, ft_name)
        if inspect.isclass(elt) and not is_enum(elt.__class__):
            in_ft_names = get_inner_fts(elt)
            for name in in_ft_names:
                tabmat += f'  - [{name}](#{name})\n'
    display(Markdown(tabmat))

def get_class_toc(mod_name:str, cls_name:str):
    """displays table of contents for `cls_name`"""
    splits = str.split(mod_name, '.')
    try: mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
    except: 
        print(f"Module {mod_name} doesn't exist.")
        return
    splits = str.split(cls_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a subclass named {split}.")
        elt = getattr(elt, split)
    assert inspect.isclass(elt) and not is_enum(elt.__class__), "This is not a valid class."
    in_ft_names = get_inner_fts(elt)
    tabmat = ''
    for name in in_ft_names:
        tabmat += f'- [{name}](#{name})\n'
    display(Markdown(tabmat))

def show_video(url):
    data = f'<iframe width="560" height="315" src="{url}" frameborder="0" allowfullscreen></iframe>'
    return display(HTML(data))

def show_video_from_youtube(code, start=0):
    url = f'https://www.youtube.com/embed/{code}?start={start}&amp;rel=0&amp;controls=0&amp;showinfo=0'
    return show_video(url)

def get_fn_link(ft) -> str:
    """returns function link to notebook documentation"""
    if hasattr(ft, '__name__'):
        name = ft.__name__
    elif hasattr(ft,'__class__'):
        name = ft.__class__.__name__
    return f'{ft.__module__}.html#{name}'

def get_source_link(ft) -> str:
    """returns link to  line in source code"""
    lineno = inspect.getsourcelines(ft)[1]
    fpath = os.path.realpath(inspect.getfile(ft))
    relpath = os.path.relpath(fpath, os.getcwd())
    link = f"{relpath}#L{lineno}"
    return f'<div style="text-align: right"><a href="{link}">[source]</a></div>'

def create_anchor(name):
    display(Markdown(f'<a id={name}></a>'))

