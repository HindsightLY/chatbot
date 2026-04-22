"""
工具管理服务
统一管理各种外部工具和服务
"""
import re
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from config.app_config import APP_CONFIG
from src.tools import get_weather_info
from src.logger_config import logger


class ToolManager:
    """
    工具管理器
    负责管理各种外部工具和辅助功能
    """

    def __init__(self):
        """初始化工具管理器"""
        self.general_llm = OllamaLLM(
            model=APP_CONFIG.llm_model_name,
            temperature=APP_CONFIG.llm_temperature,
            base_url=APP_CONFIG.llm_base_url
        )

        self.general_prompt_template = PromptTemplate.from_template(
            "你是一个友好的AI助手。用户向你提问：{query}\n"
            "根据你掌握的知识或提供的额外信息，回答用户的问题：\n"
            "{additional_context}\n"
            "请直接回答用户的问题，语言亲切自然。"
        )

    def extract_city_by_regex(self, query: str) -> Optional[str]:
        """
        使用正则表达式从查询中提取城市名

        Args:
            query: 用户查询

        Returns:
            Optional[str]: 提取的城市名，如果未找到则返回None
        """
        patterns = [
            r'(?:在|去|查|问问|了解)?([A-Za-z\u4e00-\u9fa5]{2,6}?)(?:今天|明天|后天|当前|现在的)?(?:的)?(?:天气|气温|温度|湿度|风|雨|晴|阴|雪|雾霾|空气质量)',
        ]

        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                city = match.group(1).strip()
                if city in APP_CONFIG.common_cities:
                    return city

        return None

    def extract_city_by_llm(self, query: str) -> Optional[str]:
        """
        使用大模型从用户查询中提取城市名

        Args:
            query: 用户查询

        Returns:
            Optional[str]: 提取的城市名，如果未找到则返回None
        """
        extraction_prompt = f"""
        请从以下句子中提取出城市名称。只返回城市名称，不要有任何其他文字。
        如果句子中没有提及具体城市，则返回"未找到"。

        句子: {query}

        城市名称:
        """
        try:
            city_name = self.general_llm.invoke(extraction_prompt).strip()
            # 过滤无效结果
            invalid_results = ["未找到", "没有", "", "无法", "不知道", "不清楚"]
            if city_name in invalid_results or len(city_name) < 2 or len(city_name) > 10:
                return None
            return city_name
        except Exception as e:
            logger.error(f"LLM提取城市失败: {e}")
            return None

    def get_weather_response(self, query: str) -> str:
        """
        获取天气相关的响应

        Args:
            query: 用户查询

        Returns:
            str: 天气响应
        """
        # 尝试正则提取城市
        city = self.extract_city_by_regex(query)

        # 如果正则未提取到，再用LLM提取
        if not city:
            city = self.extract_city_by_llm(query)

        if city:
            logger.info(f"Detected city: {city}")
            weather_data = get_weather_info(city)

            # 使用通用 LLM 生成回复
            formatted_prompt = self.general_prompt_template.format(
                query=query,
                additional_context=weather_data
            )
            response = self.general_llm.invoke(formatted_prompt)
            return response
        else:
            # 没有提取到城市，返回提示
            return "请告诉我具体的城市名称，例如'北京天气'或'上海今天气温'"

    def handle_general_query(self, query: str, additional_context: str = "") -> str:
        """
        处理一般性查询

        Args:
            query: 用户查询
            additional_context: 额外上下文信息

        Returns:
            str: 生成的响应
        """
        formatted_prompt = self.general_prompt_template.format(
            query=query,
            additional_context=additional_context
        )
        return self.general_llm.invoke(formatted_prompt)


# 全局工具管理器实例
tool_manager = ToolManager()