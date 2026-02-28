"""
本包职责：提供工具函数，包括日志配置、可视化和注意力分析功能

包结构：
- logger.py: 日志配置和获取
- visualizer.py: Syndrome 可视化
- attention_viz.py: GAT 注意力权重分布可视化

对外接口：
从本包可导入：
- setup_logger: 配置并返回 logger
- get_logger: 获取 logger 实例
- visualize_syndrome: 可视化单个 syndrome 文件
- plot_attention_boxplot: 按边类型绘制注意力权重分布箱线图

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['setup_logger', 'get_logger', 'visualize_syndrome', 'plot_attention_boxplot']

from .logger import setup_logger, get_logger
from .visualizer import visualize_syndrome
from .attention_viz import plot_attention_boxplot
