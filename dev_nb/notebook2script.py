import json, fire

def is_export(cell):
    if cell['cell_type'] != 'code': return False
    if len(cell['source']) == 0 or len(cell['source'][0]) < 7: return False
    return cell['source'][0][:7] == '#export' 

def notebook2script(fname, new_name):
    main_dic = json.load(open(fname,'r'))
    cells = main_dic['cells']
    code_cells = [c for c in cells if is_export(c)]
    module = ''
    for cell in code_cells:
        module += ''.join(cell['source'][1:]) + '\n\n'
    with open(new_name,'w') as f: f.write(module[:-2])
    print(f'Successfully converted {fname} to {new_name}.')

if __name__ == '__main__': fire.Fire(notebook2script)