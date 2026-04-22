"""
聚合新闻客户端模块
提供新闻获取功能
"""
import json
import os
from urllib import parse, request
from .logger_config import logger
from pydantic import BaseModel

# https://www.juhe.cn/docs/api/id/235
class JuHeNewsClient:
    """
    聚合新闻客户端
    用于获取各类新闻信息
    """

    def __init__(self, api_key=None):
        """
        初始化聚合新闻客户端

        Args:
            api_key: 聚合数据API密钥，如果不提供则从环境变量获取
        """
        self.url = 'http://v.juhe.cn/toutiao/index'
        self.api_key = api_key or os.getenv('JUHE_NEWS_API_KEY', 'eedeb472d6177bfecb950f01febf4884')

    def get_daily_news(self, news_type="top"):
        """
        获取每日新闻

        Args:
            news_type: 新闻类型，默认为"top"(头条)
                      可选值：top(头条),shehui(社会),guonei(国内),guoji(国际),
                            yule(娱乐),tiyu(体育),junshi(军事),keji(科技),
                            caijing(财经),shishang(时尚)

        Returns:
            dict: 新闻数据或错误信息
        """
        params = {
            "type": news_type,
            "key": self.api_key,
        }

        # 编码参数
        querys = parse.urlencode(params).encode('utf-8')

        # 创建请求
        req = request.Request(self.url, data=querys)

        try:
            response = request.urlopen(req)
            content = response.read().decode('utf-8')

            if content:
                result = json.loads(content)
                error_code = result.get('error_code', -1)

                if error_code == 0:
                    data = result.get('result', {}).get('data', [])

                    # 格式化新闻数据
                    formatted_news = []
                    for item in data:
                        formatted_item = {
                            "title": item.get('title', ''),
                            "date": item.get('date', ''),
                            "url": item.get('url', ''),
                            "author_name": item.get('author_name', ''),
                            "thumbnail_pic_s": item.get('thumbnail_pic_s', '')
                        }
                        formatted_news.append(formatted_item)

                    return {
                        "success": True,
                        "news": formatted_news,
                        "total": len(formatted_news)
                    }
                else:
                    return {
                        "success": False,
                        "error_code": error_code,
                        "reason": result.get('reason', 'Unknown error')
                    }
        except json.JSONDecodeError as e:
            logger.error(f"解析JSON异常：{e}")
            return {
                "success": False,
                "error": f"解析JSON异常：{e}"
            }
        except Exception as e:
            logger.error(f"请求/解析异常：{e}")
            return {
                "success": False,
                "error": f"请求/解析异常：{e}"
            }


# 便捷函数
def get_daily_news(news_type="top"):
    """
    便捷获取每日新闻的函数

    Args:
        news_type: 新闻类型

    Returns:
        dict: 新闻数据
    """
    client = JuHeNewsClient()
    return client.get_daily_news(news_type)


# 新增：新闻请求的数据模型
class NewsRequest(BaseModel):
    """新闻请求数据模型"""
    news_type: str = "top"  # 默认获取头条新闻


class NewsResponse(BaseModel):
    """新闻响应数据模型"""
    success: bool
    news: list = []
    total: int = 0
    error: str = ""