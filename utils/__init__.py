"""
本包职责：提供工具函数，包括日志配置和可视化功能

包结构：
- logger.py: 日志配置和获取
- visualizer.py: Syndrome 可视化

对外接口：
从本包可导入：
- setup_logger: 配置并返回 logger
- get_logger: 获取 logger 实例
- visualize_syndrome: 可视化单个 syndrome 文件

⚠️ 提醒：包内文件变化时必须更新本文件
"""

__all__ = ['setup_logger', 'get_logger', 'visualize_syndrome']

from .logger import setup_logger, get_logger
from .visualizer import visualize_syndrome
