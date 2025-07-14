#!/usr/bin/env python3
"""
查看专家数据的实际结构
"""

import pickle
import numpy as np

# 加载专家数据
with open('rob831/expert_data/expert_data_Ant-v2.pkl', 'rb') as f:
    expert_data = pickle.load(f)

print("=== 专家数据结构 ===")
print(f"类型: {type(expert_data)}")
print(f"长度: {len(expert_data)}")

if len(expert_data) > 0:
    first_traj = expert_data[0]
    print(f"\n第一条轨迹的keys: {first_traj.keys()}")
    
    for key in first_traj.keys():
        data = first_traj[key]
        if hasattr(data, 'shape'):
            print(f"{key}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"{key}: type={type(data)}")

print("\n=== 这就是您需要返回的 paths 格式！ ===")
