import json
import os

ipynb_path = 'training/RT_DETR_PPE_Retrain_V2.ipynb'
with open(ipynb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update CELL 3: The Download step
cell_3_source = """# CELL 3: Download APD-Juru Selamat TA Dataset from Roboflow
!pip install roboflow -q
from roboflow import Roboflow

API_KEY = input('Enter your Roboflow API key: ')
rf = Roboflow(api_key=API_KEY)

project = rf.workspace("futureroom").project("apd-juru-selamat-ta")
# We will download the latest version, usually version 1 or 2. Let's just grab the latest.
version = project.version(1) 
dataset = version.download("yolov8")

DATA_YAML = dataset.location + '/data.yaml'
print(f'\\n✅ Dataset downloaded to: {dataset.location}')
print(f'data.yaml path: {DATA_YAML}')
"""
nb['cells'][3]['source'] = [line + '\n' for line in cell_3_source.split('\n')]

# Update CELL 4: The Filter step
cell_4_source = """# CELL 4: 🧹 DYNAMIC FILTER DATASET (Keep only PPE & Person)
import os
import glob
import yaml

# 1. Read the downloaded data.yaml to find the exact class IDs
with open(DATA_YAML, "r") as f:
    data_cfg = yaml.safe_load(f)

original_names = data_cfg['names']
print("Original classes found in dataset:", original_names)

# We want to map their specific class names to our strict 5 classes
# Futureroom Dataset uses: Helmet, No-Helmet, Vest, No-Vest, Person
TARGET_MAPPING = {
    'Helmet': 'Hardhat',
    'No-Helmet': 'NO-Hardhat',
    'Vest': 'Safety Vest',
    'No-Vest': 'NO-Safety Vest',
    'Person': 'Person'
}

# Find the original integer IDs for these target classes
KEEP_CLASSES = {}
new_id = 0
for orig_id, orig_name in enumerate(original_names):
    # Try to match case-insensitively just in case
    for target_key, standardized_name in TARGET_MAPPING.items():
        if target_key.lower() == orig_name.lower():
            KEEP_CLASSES[orig_id] = new_id
            new_id += 1
            break

print(f"Mapping dictionary mapping original IDs to new IDs: {KEEP_CLASSES}")

print("\\nFiltering labels to keep only these classes...")
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

print(f"Removed {removed_count} irrelevant bounding boxes (gloves, shoes, glasses, etc.)")
print(f"Kept {kept_count} PPE/Person bounding boxes.")

# 2. Rewrite data.yaml with our strict 5 classes
data_cfg["nc"] = 5
# Ensure the names are written in the exact order of the new IDs (0 to 4)
# We will just write the standard names we use in the spatial engine
data_cfg["names"] = ["Hardhat", "NO-Hardhat", "NO-Safety Vest", "Person", "Safety Vest"]

with open(DATA_YAML, "w") as f:
    yaml.dump(data_cfg, f)
print("✅ data.yaml updated with strict 5-class schema!")
"""
nb['cells'][4]['source'] = [line + '\n' for line in cell_4_source.split('\n')]

with open(ipynb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print('Successfully updated notebook for APD-Juru Selamat dataset.')
