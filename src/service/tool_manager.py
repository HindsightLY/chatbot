"""
工具管理服务 — 为非医疗意图（闲聊、天气等）提供 LLM 响应

调用链:
  chat_router / cli_router
    → is_weather_query()           [text_utils.py]
        → True  → get_weather_response() → search_weather()  [weather_tool.py]
        → False → handle_general_query()
"""
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from config.app_config import APP_CONFIG
from src.tools.weather_tool import search_weather
from src.utils.logger_config import logger
from src.utils.text_utils import extract_city_from_text


class ToolManager:
    """
    非 RAG 工具管理器

    职责:
      - 天气查询: 正则 → LLM 兜底提取城市 → 高德 API → LLM 润色
      - 通用闲聊 / 系统查询: 直接走 LLM 生成回复
    """

    def __init__(self):
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

    def extract_city_by_llm(self, query: str) -> Optional[str]:
        """
        两阶段城市提取中的 LLM 兜底。

        当 text_utils.extract_city_from_text 的正则+列表匹配失败后，
        用 LLM 做语义理解提取城市名。输出后过滤常见无效词防止幻觉。

        Returns:
            城市名，或 None（未识别）
        """
        extraction_prompt = f"""
        请从以下句子中提取出城市名称。只返回城市名称，不要有任何其他文字。
        如果句子中没有提及具体城市，则返回"未找到"。

        句子: {query}

        城市名称:
        """
        try:
            city_name = self.general_llm.invoke(extraction_prompt).strip()
            invalid_results = ["未找到", "没有", "", "无法", "不知道", "不清楚"]
            if city_name in invalid_results or len(city_name) < 2 or len(city_name) > 10:
                return None
            return city_name
        except Exception as e:
            logger.error(f"LLM提取城市失败: {e}")
            return None

    def get_weather_response(self, query: str) -> str:
        """
        天气查询处理流水线。

        步骤:
          1. 正则快速匹配城市（extract_city_from_text）
          2. LLM 语义兜底（extract_city_by_llm）
          3. 高德天气 API 查询（search_weather）
          4. LLM 将结构化天气数据润色为自然语言回复

        Args:
            query: 用户输入，如 "北京天气"

        Returns:
            自然语言天气回答，或提示用户提供城市名称
        """
        city = extract_city_from_text(query)

        if not city:
            city = self.extract_city_by_llm(query)

        if city:
            logger.info(f"Detected city: {city}")
            weather_data = search_weather(city)

            formatted_prompt = self.general_prompt_template.format(
                query=query,
                additional_context=weather_data
            )
            response = self.general_llm.invoke(formatted_prompt)
            return response
        else:
            return "请告诉我具体的城市名称，例如'北京天气'或'上海今天气温'"

    def handle_general_query(self, query: str, additional_context: str = "") -> str:
        """
        通用闲聊 / 系统查询。

        将用户输入和可选上下文拼入 prompt 交给 LLM 生成回答。

        Args:
            query:              用户输入
            additional_context: 可选额外上下文（如系统功能说明）

        Returns:
            LLM 生成的回答文本
        """
        formatted_prompt = self.general_prompt_template.format(
            query=query,
            additional_context=additional_context
        )
        return self.general_llm.invoke(formatted_prompt)


tool_manager = ToolManager()
