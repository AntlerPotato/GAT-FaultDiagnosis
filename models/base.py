"""
Input:
  - 标准库: abc, os
  - 第三方: numpy, torch
Output:
  - BaseModel: 诊断模型的抽象基类
Position: 定义诊断模型的接口规范，所有具体模型必须继承此类并实现 train 和 predict 方法

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import os
from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseModel(ABC):
    """
    诊断模型的基类

    所有具体的模型（如 BPNN, GAT 等）都需要继承这个类，
    并实现 train 和 predict 方法。
    """

    @abstractmethod
    def train(self, train_data: tuple, val_data: tuple, epochs: int = 100, **kwargs):
        """
        训练模型

        Args:
            train_data: (X_train, Y_train) 训练集
                X_train: 输入数据（综合征），shape = (样本数, 综合征大小)
                Y_train: 标签（节点状态），shape = (样本数, 节点数)
                         每个元素是 0（无故障）或 1（有故障）
            val_data: (X_val, Y_val) 验证集，格式同 train_data
            epochs: 训练轮数，默认 100
            **kwargs: 其他训练参数（如 batch_size, learning_rate 等）
        """
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        预测节点故障概率

        Args:
            x: 单个综合征，shape = (综合征大小,)

        Returns:
            每个节点是故障的概率，shape = (节点数,)
        """
        pass

    def save_model(self, path: str) -> str:
        """
        保存模型权重到磁盘

        Args:
            path: 保存路径（不含扩展名，自动添加 .pth）

        Returns:
            实际保存的文件路径
        """
        if not path.endswith('.pth'):
            path = path + '.pth'
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        # 子类需要有 network 属性（nn.Module）
        torch.save(self.network.state_dict(), path)
        return path

    def load_model(self, path: str) -> None:
        """
        从磁盘加载模型权重

        Args:
            path: 模型文件路径

        Raises:
            FileNotFoundError: 文件不存在
        """
        if not path.endswith('.pth'):
            path = path + '.pth'
        state_dict = torch.load(path, weights_only=True)
        self.network.load_state_dict(state_dict)
