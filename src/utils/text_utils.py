"""
文本工具模块
提供通用的文本处理函数
"""
import re
from typing import Optional

from config.app_config import APP_CONFIG


def extract_city_from_text(text: str) -> Optional[str]:
    """
    从文本中提取城市名

    Args:
        text: 输入文本

    Returns:
        Optional[str]: 提取的城市名，如果未找到则返回None
    """
    # 尝试正则匹配
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

    # 尝试从常见城市列表中匹配
    for city in APP_CONFIG.common_cities:
        if city in text:
            return city

    return None


def clean_text(text: str) -> str:
    """
    清理文本，去除多余空白和特殊字符

    Args:
        text: 输入文本

    Returns:
        str: 清理后的文本
    """
    # 去除首尾空白
    text = text.strip()
    # 替换多个空白为单个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除不可见字符
    text = re.sub(r'[^\x20-\x7E\u4e00-\u9fa5，。、；：‘’“”【】《》？！……（）]', '', text)
    return text


def truncate_text(text: str, max_length: int = 500) -> str:
    """
    截断文本

    Args:
        text: 输入文本
        max_length: 最大长度

    Returns:
        str: 截断后的文本
    """
    if len(text) <= max_length:
        return text

    # 在句子边界截断
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
    格式化响应文本

    Args:
        response: 响应文本
        max_line_length: 每行最大长度

    Returns:
        str: 格式化后的文本
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
