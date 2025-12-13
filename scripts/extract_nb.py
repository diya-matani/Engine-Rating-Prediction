import json
import os

nb_path = r'C:\Users\dmata\OneDrive - vitbhopal.ac.in\Documents\GitHub\Engine-Rating-Prediction\engine-rating-prediction.ipynb'
out_path = r'C:\Users\dmata\OneDrive - vitbhopal.ac.in\Documents\GitHub\Engine-Rating-Prediction\scripts\extracted_nb.py'

print(f"Reading {nb_path}...")
try:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_cells.append(f"# Cell {i}\n{source}")
            
    print(f"Found {len(code_cells)} code cells.")
    
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(code_cells))
        
    print(f"Written to {out_path}")
    
except Exception as e:
    print(f"Error: {e}")
