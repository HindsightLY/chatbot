"""
文本工具模块 — 通用的文本处理函数

引用关系:
  - extract_city_from_text() → tool_manager.py（天气查询城市提取，快速路径）
  - is_weather_query()       → chat_router.py, cli_router.py（天气意图判别）
  - clean_text() / truncate_text() / format_response() → 通用文本预处理/后处理
"""
import re
from typing import Optional
from config.app_config import APP_CONFIG


def extract_city_from_text(text: str) -> Optional[str]:
    """
    从文本中提取城市名（两阶段: 正则匹配 → 列表全量匹配）。

    正则模式:
      1. 匹配 "X天气"、"X今天气温" 等常见表述
      2. 匹配 "X 天气"（空格分隔）

    列表匹配: 逐 city 检查是否出现在 text 中。

    注意: 这是快速路径（正则+城市列表匹配），失败后由 LLM 兜底提取。

    Args:
        text: 用户输入文本

    Returns:
        城市名 | None
    """
    patterns = [
        r'(?:在|去|查|问问|了解)?([A-Za-z\u4e00-\u9fa5]{2,6}?)(?:今天|明天|后天|当前|现在的)?(?:的)?(?:天气|气温|温度|湿度|风|雨|晴|阴|雪|雾霾|空气质量)',
        r'([A-Za-z\u4e00-\u9fa5]{2,6}?)\s+(?:天气|气温|温度)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            city = match.group(1).strip()
            if city in APP_CONFIG.common_cities:
                return city

    for city in APP_CONFIG.common_cities:
        if city in text:
            return city

    return None


WEATHER_KEYWORDS = ['天气', 'weather', '气温', '温度']


def is_weather_query(text: str) -> bool:
    """
    检查用户输入是否为天气查询（大小写不敏感）。

    通过匹配预定义关键词列表做快速判断。
    """
    return any(keyword in text.lower() for keyword in WEATHER_KEYWORDS)


def clean_text(text: str) -> str:
    """
    去除多余空白和不可见字符，保留中英文及常见标点。

    Args:
        text: 原始文本

    Returns:
        清洗后的文本
    """
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\u4e00-\u9fa5，。、；：‘’“”【】《》？！……（）]', '', text)
    return text


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    在句子边界截断文本，避免截断在词中间。

    按句号/感叹号/问号/分号拆分句子，
    尽可能包含完整句子，超出 max_length 时截断并追加 "...".

    Args:
        text:       原始文本
        max_length: 最大字符数

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    sentences = re.split(r'(?<=[。！？；])', text)
    result = ''
    for sentence in sentences:
        if len(result + sentence) > max_length:
            break
        result += sentence

    if not result:
        result = text[:max_length]

    return result + '...'


def format_response(response: str, max_line_length: int = 80) -> str:
    """
    按标点符号换行，保证每行不超过 max_line_length。

    Args:
        response:       原始回复文本
        max_line_length: 每行最大字符数

    Returns:
        换行后的文本
    """
    lines = []
    current_line = ''

    for char in response:
        if char in '。！？；\n':
            current_line += char
            lines.append(current_line.strip())
            current_line = ''
        elif len(current_line) >= max_line_length - 1:
            lines.append(current_line.strip())
            current_line = char
        else:
            current_line += char

    if current_line:
        lines.append(current_line.strip())

    return '\n'.join(lines)
