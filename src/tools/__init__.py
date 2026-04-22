"""
工具模块
包含外部API集成和业务相关工具
"""
from src.tools.news_tool import get_daily_news, NewsResponse, NewsRequest
from src.tools.weather_tool import get_weather_info

# 导出常用工具
__all__ = [
    'get_weather_info',
    'get_daily_news',
    'NewsResponse',
    'NewsRequest'
]
