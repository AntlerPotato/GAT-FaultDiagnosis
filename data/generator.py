"""
Input:
  - 标准库: random
  - 第三方: numpy
  - 本地: topologies.hypercube.Hypercube
Output:
  - generate_data: 生成并划分训练/验证/测试数据集
Position: 根据网络拓扑生成故障诊断数据，支持自定义故障数量和样本数

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import numpy as np
import random
from topologies.hypercube import Hypercube


def generate_data(topo: Hypercube, max_faults: int, n_samples: int,
                  split: tuple = (0.8, 0.1, 0.1), seed: int = 42):
    """
    生成并划分数据集
    
    Args:
        topo: 网络拓扑
        max_faults: 最大故障数
        n_samples: 总样本数
        split: (train, val, test) 比例，默认 80/10/10
        seed: 随机种子，确保可重复性
        
    Returns:
        (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)
    """
    # 设置随机种子以确保可重复性
    np.random.seed(seed)
    random.seed(seed)
    
    X, Y = [], []
    for _ in range(n_samples):
        n_faults = random.randint(1, max_faults)
        faulty = set(random.sample(range(topo.n_nodes), n_faults))
        X.append(topo.generate_PMC_syndrome(faulty))
        Y.append(np.array([1.0 if i in faulty else 0.0 for i in range(topo.n_nodes)], dtype=np.float32))
    
    X, Y = np.array(X), np.array(Y)
    
    # 打乱并划分
    idx = np.random.permutation(n_samples)
    train_end = int(n_samples * split[0])
    val_end = train_end + int(n_samples * split[1])
    
    return (
        (X[idx[:train_end]], Y[idx[:train_end]]),
        (X[idx[train_end:val_end]], Y[idx[train_end:val_end]]),
        (X[idx[val_end:]], Y[idx[val_end:]])
    )
