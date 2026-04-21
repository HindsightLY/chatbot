import sys
import time
import logging
import functools
from logging import StreamHandler, Formatter


def setup_logger():
    # 创建一个格式器
    formatter = Formatter(
        fmt='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # 设置全局日志级别

    # 避免重复添加 Handler (防止日志重复打印)
    if not root_logger.handlers:
        # 创建控制台处理器
        handler = StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # 返回一个命名空间日志记录器，方便区分来源
    return logging.getLogger("MedicalAI")


# 全局实例
logger = setup_logger()

def monitor_performance(func):
    """一个简单的性能监控装饰器"""
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