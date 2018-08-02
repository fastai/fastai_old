import json, fire, re

def is_export(cell):
    if cell['cell_type'] != 'code': return False
    src = cell['source']
    if len(src) == 0 or len(src[0]) < 7: return False
    #import pdb; pdb.set_trace()
    return re.match('^\s*#export\s*$', src[0], re.IGNORECASE) is not None

def notebook2script(fname):
    main_dic = json.load(open(fname,'r'))
    cells = main_dic['cells']
    code_cells = [c for c in cells if is_export(c)]
    module = ''
    for cell in code_cells: module += ''.join(cell['source'][1:]) + '\n\n'
    number = file_name.split('_')[0]
    with open(f'nb_{number}.py','w') as f: f.write(module[:-2])

if __name__ == '__main__': fire.Fire(notebook2script)
