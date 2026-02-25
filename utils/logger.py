"""
Input:
  - 标准库: logging, sys
  - 第三方: 无
Output:
  - setup_logger: 配置并返回 logger
  - get_logger: 获取 logger 实例
Position: 提供统一的日志配置和获取功能

⚠️ 提醒：修改本文件后务必：
   1. 更新上方 Input/Output/Position 说明
   2. 更新所属文件夹的 README.md（如果影响模块结构）
   3. 更新根目录 README.md（如果影响核心架构）
"""

import logging
import sys

_logger = None


def setup_logger(name: str = "AI4FaultDiagnosis", level: int = logging.INFO) -> logging.Logger:
    """
    配置并返回 logger
    
    Args:
        name: logger 名称
        level: 日志级别
    """
    global _logger
    
    _logger = logging.getLogger(name)
    _logger.setLevel(level)
    
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        ))
        _logger.addHandler(handler)
    
    return _logger


def get_logger() -> logging.Logger:
    """获取 logger，如果未初始化则自动创建"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger

