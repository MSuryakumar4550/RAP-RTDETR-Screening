import json
import os

ipynb_path = 'training/RT_DETR_PPE_Retrain_V2.ipynb'
with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

filter_code = """# CELL 4: 🧹 FILTER DATASET (Keep only PPE & Person)
import os
import glob
import yaml

# The 5 classes we actually care about from the original 17
# 4: Hardhat, 6: NO-Hardhat, 8: NO-Safety Vest, 9: Person, 12: Safety Vest
KEEP_CLASSES = {
    4: 0,   # Hardhat -> new ID 0
    6: 1,   # NO-Hardhat -> new ID 1
    8: 2,   # NO-Safety Vest -> new ID 2
    9: 3,   # Person -> new ID 3
    12: 4   # Safety Vest -> new ID 4
}

print("Filtering labels to keep only PPE classes...")
label_files = glob.glob(os.path.join(dataset.location, "**", "labels", "*.txt"), recursive=True)

removed_count = 0
kept_count = 0

for file_path in label_files:
    with open(file_path, "r") as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts: continue
        class_id = int(parts[0])
        
        if class_id in KEEP_CLASSES:
            new_class_id = KEEP_CLASSES[class_id]
            parts[0] = str(new_class_id)
            new_lines.append(" ".join(parts) + "\\n")
            kept_count += 1
        else:
            removed_count += 1
            
    with open(file_path, "w") as f:
        f.writelines(new_lines)

print(f"Removed {removed_count} irrelevant bounding boxes (excavators, dumpsters, etc.)")
print(f"Kept {kept_count} PPE/Person bounding boxes.")

# Rewrite data.yaml
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

data_cfg["nc"] = 5
data_cfg["names"] = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Person", "Safety Vest"]

with open(DATA_YAML, "w") as f:
    yaml.dump(data_cfg, f)
print("✅ data.yaml updated with strict 5-class schema!")
"""

new_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [line + '\n' for line in filter_code.split('\n')]
}

# Insert after Cell 3 (which is index 3)
nb['cells'].insert(4, new_cell)

# Update the next cell's comment (was Cell 4, now Cell 5)
if '# CELL 4' in nb['cells'][5]['source'][0]:
    nb['cells'][5]['source'][0] = nb['cells'][5]['source'][0].replace('CELL 4', 'CELL 5')
if '# CELL 5' in nb['cells'][6]['source'][0]:
    nb['cells'][6]['source'][0] = nb['cells'][6]['source'][0].replace('CELL 5', 'CELL 6')
if '# CELL 6' in nb['cells'][7]['source'][0]:
    nb['cells'][7]['source'][0] = nb['cells'][7]['source'][0].replace('CELL 6', 'CELL 7')
if '# CELL 7' in nb['cells'][8]['source'][0]:
    nb['cells'][8]['source'][0] = nb['cells'][8]['source'][0].replace('CELL 7', 'CELL 8')

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully inserted dataset filtering cell into notebook.')
