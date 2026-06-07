"""
天气查询工具 — 高德地图天气 API 封装

调用链:
  tool_manager.get_weather_response()
    → extract_city_from_text() 或 extract_city_by_llm() 提取城市
    → search_weather() 调用高德 API
    → LLM 将结构化数据润色为自然语言

API: 高德地图 Web 服务 — 天气查询（实时）
"""
import requests
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


def search_weather(city: str):
    """
    调用高德地图天气 API 获取指定城市的实时天气。

    Args:
        city: 城市名（如 "北京"）

    Returns:
        格式化后的天气描述字符串:
          "{city}的天气情况：\n天气: ...\n温度: ...°C\n湿度: ...%\n风向: ...\n风力: ...级"
        异常时返回中文错误描述。
    """
    params = {
        'key': APP_CONFIG.amap_api_key,
        'city': city,
        'extensions': 'base'
    }
    try:
        response = requests.get(APP_CONFIG.amap_weather_url, params=params)
        response.raise_for_status()
        data = response.json()

        if data.get('status') == '1':
            weather_info_list = data.get('lives', [])
            if weather_info_list:
                weather = weather_info_list[0]
                return (
                    f"{weather['city']}的天气情况：\n"
                    f"天气: {weather['weather']}\n"
                    f"温度: {weather['temperature']}°C\n"
                    f"湿度: {weather['humidity']}%\n"
                    f"风向: {weather['winddirection']}\n"
                    f"风力: {weather['windpower']}级"
                )
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
