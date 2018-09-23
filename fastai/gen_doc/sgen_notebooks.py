import argparse
from .gen_notebooks import generate_all, update_all
from pathlib import Path

if __name__=='main':
    parser = argparse.ArgumentParser()
    parser.add_argument("source_path", type=str, help='the directory where the modules to document are')
    parser.add_argument("dest_path", type=str, help='the destination folder')
    parser.add_argument('--update', action='store_true', help='decides if the script generates or updates the doc')
    arg = parser.parse_args()
    if arg.update: update_all(arg.source_path, Path(arg.dest_path))
    else: generate_all(arg.source_path, Path(arg.dest_path))