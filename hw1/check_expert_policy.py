#!/usr/bin/env python3
"""
检查专家策略文件的内容结构
"""

import pickle
import numpy as np

# 检查专家策略文件
policy_file = 'rob831/policies/experts/Ant.pkl'
with open(policy_file, 'rb') as f:
    policy_data = pickle.load(f)

print("=== 专家策略文件内容 ===")
print(f"顶层keys: {list(policy_data.keys())}")

if 'GaussianPolicy' in policy_data:
    gp = policy_data['GaussianPolicy']
    print(f"\nGaussianPolicy keys: {list(gp.keys())}")
    
    if 'hidden' in gp and 'FeedforwardNet' in gp['hidden']:
        ffn = gp['hidden']['FeedforwardNet']
        print(f"隐藏层数量: {len(ffn)}")
        
        for layer_name in sorted(ffn.keys()):
            layer = ffn[layer_name]['AffineLayer']
            W_shape = layer['W'].shape
            b_shape = layer['b'].shape
            print(f"  {layer_name}: W{W_shape}, b{b_shape}")

print(f"\n这是一个完整的神经网络！不只是数据轨迹！")
