"""
Input:
  - 标准库: os, json, datetime
  - 第三方: numpy
Output:
  - save_dataset: 保存数据集到磁盘
  - load_dataset: 从磁盘加载数据集
Position: 提供数据集的持久化功能，支持保存和加载训练/验证/测试数据

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import os
import json
import numpy as np
from datetime import datetime

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")


def save_dataset(train_data: tuple, val_data: tuple, test_data: tuple,
                 name: str, metadata: dict = None) -> str:
    """
    保存数据集，每个 syndrome 单独保存
    
    目录结构:
        datasets/{name}/{timestamp}/
            metadata.json
            train/1.npz, 2.npz, ...
            val/1.npz, 2.npz, ...
            test/1.npz, 2.npz, ...
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = os.path.join(DATASETS_DIR, name, timestamp)
    
    # 保存各个子集
    for split_name, (X, Y) in [("train", train_data), ("val", val_data), ("test", test_data)]:
        split_dir = os.path.join(dataset_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i, (x, y) in enumerate(zip(X, Y), 1):
            np.savez(os.path.join(split_dir, f"{i}.npz"), syndrome=x, label=y)
    
    # 保存元数据
    meta = metadata or {}
    meta.update({
        "timestamp": timestamp,
        "train_size": len(train_data[0]),
        "val_size": len(val_data[0]),
        "test_size": len(test_data[0])
    })
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    
    return dataset_dir


def load_dataset(name: str, timestamp: str = None) -> tuple:
    """
    加载数据集

    Args:
        name: 数据集名称
        timestamp: 时间戳，None 表示加载最新的

    Returns:
        (train_data, val_data, test_data, metadata)

    Raises:
        FileNotFoundError: 数据集不存在
        ValueError: 数据集目录为空
    """
    name_dir = os.path.join(DATASETS_DIR, name)

    # 检查数据集目录是否存在
    if not os.path.exists(name_dir):
        raise FileNotFoundError(f"Dataset '{name}' not found at {name_dir}")

    # 找最新的时间戳
    if timestamp is None:
        timestamps = [d for d in os.listdir(name_dir)
                     if os.path.isdir(os.path.join(name_dir, d))]
        if not timestamps:
            raise ValueError(f"No valid dataset found in {name_dir}")
        timestamp = sorted(timestamps)[-1]

    dataset_dir = os.path.join(name_dir, timestamp)

    # 检查时间戳目录是否存在
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset timestamp '{timestamp}' not found at {dataset_dir}")

    # 加载元数据
    with open(os.path.join(dataset_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)

    # 加载各个子集
    def load_split(split_name):
        split_dir = os.path.join(dataset_dir, split_name)
        # 只处理 .npz 文件，过滤其他文件
        files = [f for f in os.listdir(split_dir) if f.endswith('.npz')]
        # 按数字排序
        files = sorted(files, key=lambda x: int(x.split(".")[0]))
        X, Y = [], []
        for fname in files:
            data = np.load(os.path.join(split_dir, fname))
            X.append(data["syndrome"])
            Y.append(data["label"])
        return np.array(X), np.array(Y)

    train_data = load_split("train")
    val_data = load_split("val")
    test_data = load_split("test")

    return train_data, val_data, test_data, metadata
