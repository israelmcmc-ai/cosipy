"""Convert a '# %%' script to a notebook and execute it: py2nb.py script.py out.ipynb"""
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

src, out = Path(sys.argv[1]), Path(sys.argv[2])
cells = []
for chunk in src.read_text().split('# %%')[1:]:
    header, _, body = chunk.partition('\n')
    if header.strip() == '[markdown]':
        text = '\n'.join(line[2:] if line.startswith('# ') else line.lstrip('#') for line in body.strip().splitlines())
        cells.append(nbformat.v4.new_markdown_cell(text))
    else:
        cells.append(nbformat.v4.new_code_cell(body.strip()))

nb = nbformat.v4.new_notebook(cells=cells)
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
NotebookClient(nb, timeout=None, resources={'metadata': {'path': str(out.parent)}}).execute()
nbformat.write(nb, out)
