import numpy as np
import os

# 定义文件路径 (根据你之前的操作，假设在 data/08-data-01 下)
# 如果你放在了其他文件夹，请修改这里的 dataset_name
dataset_name = "08-data-01"
structure_path = os.path.join("data", dataset_name, "structures.npy")
property_path = os.path.join("data", dataset_name, "properties.npy")

print(f"Checking dataset: {dataset_name}...\n")

# 1. 检查结构文件 (Structures)
if os.path.exists(structure_path):
    try:
        structures = np.load(structure_path)
        print(f"=== Structure File: {structure_path} ===")
        print(f"Total Shape: {structures.shape}")
        print(f"Data Type:   {structures.dtype}")
        print(f"First Sample (Shape: {structures[1].shape}):\n{structures[1]}")
    except Exception as e:
        print(f"Error loading structures: {e}")
else:
    print(f"[Error] File not found: {structure_path}")

print("-" * 30)

# 2. 检查属性文件 (Properties)
if os.path.exists(property_path):
    try:
        properties = np.load(property_path)
        print(f"=== Property File: {property_path} ===")
        print(f"Total Shape: {properties.shape}")
        print(f"Data Type:   {properties.dtype}")
        print(f"First Sample (Shape: {properties[1].shape}):\n{properties[1]}")
        
        # 如果是复数形式（实部+虚部），尝试解读一下
        if properties.shape[1] == 2:
            print(f"Interpreted as Complex: {properties[1][0]} + {properties[1][1]}i")
            
    except Exception as e:
        print(f"Error loading properties: {e}")
else:
    print(f"[Error] File not found: {property_path}")

print("\nDone.")