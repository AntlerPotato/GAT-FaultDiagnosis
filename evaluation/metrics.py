"""
Input:
  - 标准库: 无
  - 第三方: numpy
  - 本地: models.base.BaseModel
Output:
  - evaluate: 在测试集上评估模型性能
Position: 提供模型评估功能，计算准确率、精确率、召回率和 F1-Score

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import numpy as np
from models.base import BaseModel


def evaluate(model: BaseModel, test_data: tuple, threshold: float = 0.5) -> dict:
    """
    在测试集上评估模型

    Args:
        model: 训练好的模型
        test_data: (X_test, Y_test)
        threshold: 判定故障的阈值，默认 0.5

    Returns:
        包含 accuracy, precision, recall, f1 的字典
    """
    X_test, Y_test = test_data
    correct, total_prec, total_rec = 0, 0, 0

    for x, y in zip(X_test, Y_test):
        actual = set(np.where(y == 1)[0])
        pred_probs = model.predict(x)
        pred = set(np.where(pred_probs > threshold)[0])

        if pred == actual:
            correct += 1

        total_prec += len(pred & actual) / len(pred) if pred else 0
        total_rec += len(pred & actual) / len(actual) if actual else (1.0 if not pred else 0.0)

    n = len(X_test)
    precision = total_prec / n
    recall = total_rec / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
