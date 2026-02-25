"""
本包职责：提供模型评估功能

包结构：
- metrics.py: 模型评估指标计算

对外接口：
从本包可导入：
- evaluate: 在测试集上评估模型性能

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['evaluate']

from .metrics import evaluate

