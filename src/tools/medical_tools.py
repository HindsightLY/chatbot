"""
LangChain 工具集
使用 @tool 装饰器定义可复用工具，供 Agent 的 bind_tools() + ToolNode 调用

工具列表:
  search_medical_knowledge  — 医学知识库检索
  get_weather               — 天气查询
  chat_general              — 通用闲聊兜底
"""
from langchain_core.tools import tool
from config.app_config import APP_CONFIG
from src.utils.text_utils import extract_city_from_text
from src.utils.logger_config import logger


def _get_vector_store():
    """延迟获取向量存储实例，避免循环导入"""
    from src.service.system_initializer import system_initializer
    return system_initializer.vector_store


def _get_llm():
    """延迟获取 LLM 实例（与 agent.py 保持一致使用 ChatOllama）"""
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=APP_CONFIG.llm_model_name,
        temperature=APP_CONFIG.llm_temperature,
        base_url=APP_CONFIG.llm_base_url
    )


@tool
def search_medical_knowledge(query: str) -> str:
    """
    搜索医学知识库，获取与疾病、症状、治疗方法相关的医学资料。
    当用户询问疾病、症状、用药、治疗方法等医疗相关问题时，调用此工具。

    Args:
        query: 用户的医疗问题原文
    """
    logger.info(f"🔧 调用工具: search_medical_knowledge(query='{query}')")
    vs = _get_vector_store()
    if not vs or not vs.vector_store:
        return "医学知识库暂未就绪，无法检索。"

    try:
        search_query = query
        if APP_CONFIG.use_hyde:
            from src.service.hyde_transformer import hyde_transformer
            hyde_query = hyde_transformer.transform(query)
            search_query = hyde_query
        if APP_CONFIG.use_hybrid_search:
            results = vs.hybrid_search(search_query, k=APP_CONFIG.retrieval_k)
        else:
            results = vs.similarity_search(
                query, k=APP_CONFIG.retrieval_k,
                score_threshold=APP_CONFIG.retrieval_score_threshold
            )
        docs = [doc.page_content for doc in results]
        if not docs:
            return "未找到相关医学资料。"
        logger.info(f"📚 工具检索到 {len(docs)} 篇文档")
        return "\n\n".join(docs)
    except Exception as e:
        logger.exception(f"❌ 医学检索工具失败")
        return "检索医学知识时发生错误。"


@tool
def get_weather(location: str) -> str:
    """
    查询指定城市的实时天气信息。
    当用户询问天气时调用此工具。

    Args:
        location: 城市名称，如"北京"、"上海"
    """
    logger.info(f"🔧 调用工具: get_weather(location='{location}')")
    city = extract_city_from_text(location)
    if not city:
        city = location.strip()
    if not city or len(city) < 2:
        return "请提供具体的城市名称。"

    from src.tools.weather_tool import search_weather
    result = search_weather(city)
    return result


@tool
def chat_general(query: str) -> str:
    """
    回答用户的一般性问题、闲聊、问候等非医疗非天气问题。
    仅当问题不涉及疾病症状、不涉及天气查询时使用。

    Args:
        query: 用户输入
    """
    logger.info(f"🔧 调用工具: chat_general(query='{query}')")
    llm = _get_llm()
    result = llm.invoke(
        f"你是一个友好的AI助手。请直接回答用户的问题，语言亲切自然。\n用户: {query}\n回答:"
    )
    return result.content if hasattr(result, "content") else str(result)


tools = [search_medical_knowledge, get_weather, chat_general]
