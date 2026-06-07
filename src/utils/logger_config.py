"""
日志配置模块 — 基于标准库 logging 的统一日志输出和性能监控

所有模块通过 from src.utils.logger_config import logger 获取日志实例。

功能:
  - 统一的日志格式（时间、级别、模块名、消息）
  - 控制台输出（UTF-8 编码）
  - @monitor_performance 装饰器自动记录函数耗时和异常
"""
import sys
import time
import logging
import functools
from logging import StreamHandler, Formatter


def setup_logger():
    """
    配置根日志记录器（防止重复添加 handler）

    Returns:
        命名空间为 "MedicalAI" 的 Logger 实例
    """
    formatter = Formatter(
        fmt='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not root_logger.handlers:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    return logging.getLogger("MedicalAI")


logger = setup_logger()


def monitor_performance(func):
    """
    函数性能监控装饰器

    自动记录:
      - 开始执行
      - 完成耗时
      - 异常退出 + 耗时

    被 cli_router.run_cli() 使用。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"🚀 开始执行: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"🏁 完成执行: {func.__name__} | 耗时: {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 异常退出: {func.__name__} | 耗时: {duration:.2f}s | 错误: {e}")
            raise

    return wrapper
