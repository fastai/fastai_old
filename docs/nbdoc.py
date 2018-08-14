import inspect,importlib,enum
from IPython.core.display import display, Markdown, HTML

__all__ = ['create_anchor', 'get_class_toc', 'get_fn_link', 'get_module_toc', 'show_doc', 'show_doc_from_name', 
           'show_video', 'show_video_from_youtube']

def parse_args(elt, ignore_first=False, arg_comments={}):
    parsed = ""
    args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = get_arg_spec(elt)
    if ignore_first: args = args[1:]
    if 'return' in annotations: parsed+= f" (-> {annotations['return']})"
    if len(args) != 0:
        parsed += '\n\nArguments:\n'
        diff = len(args) - len(defaults) if defaults is not None else len(args)
        for i,arg in enumerate(args):
            parsed += f'- **{arg}**'
            if arg in annotations: parsed += f' ({annotations[arg]})'
            if arg in arg_comments: parsed += f': {arg_comments[arg]}'
            if i-diff >= 0: parsed += f', *default {defaults[i-diff]}*'
            parsed += '\n'
    return parsed

def is_enum(cls):
    return cls == enum.Enum or cls == enum.EnumMeta

def wrap_class(t):
    if hasattr(t, '__name__'): return t.__name__
    else: return str(t)

def get_arg_spec(elt):
    a = inspect.getfullargspec(elt)
    b= {k:wrap_class(v) for k,v in a[6].items()}
    return (*a[0:6], b)

def get_ft_doc(elt, full_name, arg_comments={}):
    doc = f'**{full_name}**' + parse_args(elt, False, arg_comments)
    return doc

def get_enum_doc(elt, full_name, arg_comments={}):
    doc = f'**{full_name}** (enumerator class)\n\nValues:\n'
    for val in elt.__members__.keys():
        doc += f'- **{val}**'
        if val in arg_comments: doc += f': arg_comments[val]'
        doc +='\n'
    return doc

def get_cls_doc(elt, full_name, arg_comments={}):
    parent_class = inspect.getclasstree([elt])[-1][0][1][0]
    doc = f'**{full_name}**'
    if parent_class != object: doc += f'(subclass of {parent_class})'
    doc += parse_args(elt, True, arg_comments)
    return doc

def show_doc(elt, doc_string=True, full_name=None, arg_comments={}, alt_doc_string=''):
    if full_name is None: full_name = elt.__name__
    if inspect.isclass(elt):
        if is_enum(elt.__class__): doc = get_enum_doc(elt, full_name, arg_comments)
        else:                      doc = get_cls_doc(elt, full_name, arg_comments) 
    elif inspect.isfunction(elt):  doc = get_ft_doc(elt, full_name, arg_comments)
    link = f'<a id={full_name}></a>'
    if doc_string and inspect.getdoc(elt) is not None: doc += '\n' + inspect.getdoc(elt)
    if len(alt_doc_string) != 0: doc += '\n\n' + alt_doc_string
    display(Markdown(link + doc))

def import_mod(mod_name):
    splits = str.split(mod_name, '.')
    try: 
        if len(splits) > 1 : mod = importlib.import_module('.' + '.'.join(splits[1:]), splits[0])
        else: mod = importlib.import_module(mod_name)
        return mod
    except: 
        print(f"Module {mod_name} doesn't exist.")

def show_doc_from_name(mod_name, ft_name, doc_string=True, arg_comments={}, alt_doc_string=''):
    mod = import_mod(mod_name)
    splits = str.split(ft_name, '.')
    assert hasattr(mod, splits[0]), print(f"Module {mod_name} doesn't have a function named {splits[0]}.")
    elt = getattr(mod, splits[0])
    for i,split in enumerate(splits[1:]):
        assert hasattr(elt, split), print(f"Class {'.'.join(splits[:i+1])} doesn't have a function named {split}.")
        elt = getattr(elt, split)
    show_doc(elt, doc_string, ft_name, arg_comments, alt_doc_string)

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

def get_inner_fts(elt):
    fts = []
    for ft_name in elt.__dict__.keys():
        if ft_name[:2] == '__': continue
        ft = getattr(elt, ft_name)
        if inspect.isfunction(ft): fts.append(f'{elt.__name__}.{ft_name}')
        if inspect.isclass(ft): fts += [f'{elt.__name__}.{n}' for n in get_inner_fts(ft)]
    return fts

def get_module_toc(mod_name):
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

def get_class_toc(mod_name, cls_name):
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

def get_fn_link(ft):
    if hasattr(ft, '__name__'):
        name = ft.__name__
    elif hasattr(ft,'__class__'):
        name = ft.__class__.__name__
    return f'{ft.__module__}.ipynb#{name}'

def create_anchor(name):
    display(Markdown(f'<a id={name}></a>'))

