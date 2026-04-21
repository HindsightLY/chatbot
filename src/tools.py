from config.settings import AMAP_WEATHER_URL, AMAP_API_KEY
import requests
from datetime import datetime
from src.logger_config import logger


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取指定地点的天气情况",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市名称，例如 北京, 上海",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
    }
]


def get_current_weather(location, unit="celsius"):
    """这是一个模拟函数，实际项目中这里会调用真实的天气API"""
    # return f"{location} 当前的天气是晴天，气温 25 摄氏度。"
    return search_weather(city=location)


def get_current_time():
    """获取当前时间"""
    now = datetime.now()
    return f"当前时间是 {now.strftime('%Y年%m月%d日 %H:%M:%S')}，星期{now.strftime('%A')[:2]}"


def search_weather(city: str):
    """
    调用高德天气API获取天气信息
    :param city: 城市名称
    :return: 天气信息字符串或错误信息
    """
    params = {
        'key': AMAP_API_KEY,
        'city': city,
        'extensions': 'base' # 获取基本天气信息
    }
    try:
        response = requests.get(AMAP_WEATHER_URL, params=params)
        response.raise_for_status() # 检查HTTP错误
        data = response.json()

        if data.get('status') == '1':
            weather_info_list = data.get('lives', [])
            if weather_info_list:
                weather = weather_info_list[0]
                info = (
                    f"{weather['city']}的天气情况：\n"
                    f"天气: {weather['weather']}\n"
                    f"温度: {weather['temperature']}°C\n"
                    f"湿度: {weather['humidity']}%\n"
                    f"风向: {weather['winddirection']}\n"
                    f"风力: {weather['windpower']}级"
                )
                return info
            else:
                return f"未能获取到 {city} 的天气信息。"
        else:
            return f"高德API返回错误: {data.get('info', '未知错误')}"

    except requests.exceptions.RequestException as e:
        logger.error(f"调用高德天气API时发生错误: {e}")
        return "获取天气信息时出现网络错误。"
    except Exception as e:
        logger.error(f"解析高德天气API响应时发生错误: {e}")
        return "获取天气信息时出现解析错误。"


def get_weather_info(city: str):
    """
    获取天气信息的函数，供main.py调用
    :param city: 城市名称
    :return: 天气信息字符串
    """
    return search_weather(city)