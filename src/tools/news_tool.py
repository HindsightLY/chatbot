"""
新闻获取工具 — 聚合数据新闻 API 封装

API: https://www.juhe.cn/docs/api/id/235

被 chat_router.py 的 /api/chat/daily_news 接口调用:
  get_daily_news() → JuHeNewsClient.get_daily_news()
"""
import json
import os
from urllib import parse, request
from src.utils.logger_config import logger
from pydantic import BaseModel


class JuHeNewsClient:
    """
    聚合数据新闻 API 客户端。

    API Key 优先级: 构造参数 > 环境变量 JUHE_NEWS_API_KEY > 默认值
    """

    def __init__(self, api_key=None):
        self.url = 'http://v.juhe.cn/toutiao/index'
        self.api_key = api_key or os.getenv('JUHE_NEWS_API_KEY', 'eedeb472d6177bfecb950f01febf4884')

    def get_daily_news(self, news_type="top"):
        """
        调用聚合数据新闻 API 获取分类新闻。

        Args:
            news_type: 新闻分类（top/shehui/guonei/guoji/yule/tiyu/junshi/keji/caijing/shishang）

        Returns:
            成功: {"success": True, "news": [{"title", "date", "url", "author_name", "thumbnail_pic_s"}, ...], "total": int}
            失败: {"success": False, "error": str, "reason": str}
        """
        params = {
            "type": news_type,
            "key": self.api_key,
        }

        querys = parse.urlencode(params).encode('utf-8')
        req = request.Request(self.url, data=querys)

        try:
            response = request.urlopen(req)
            content = response.read().decode('utf-8')

            if content:
                result = json.loads(content)
                error_code = result.get('error_code', -1)

                if error_code == 0:
                    data = result.get('result', {}).get('data', [])

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
            return {"success": False, "error": f"解析JSON异常：{e}"}
        except Exception as e:
            logger.error(f"请求/解析异常：{e}")
            return {"success": False, "error": f"请求/解析异常：{e}"}


def get_daily_news(news_type="top"):
    """便捷函数: 创建客户端并获取新闻"""
    client = JuHeNewsClient()
    return client.get_daily_news(news_type)


class NewsRequest(BaseModel):
    """新闻请求体"""
    news_type: str = "top"


class NewsResponse(BaseModel):
    """新闻响应体"""
    success: bool
    news: list = []
    total: int = 0
    error: str = ""
