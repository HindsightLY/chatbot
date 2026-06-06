"""
LangChain 工具集 — 供 Agent 的 bind_tools() + ToolNode 调用

工具列表:
  search_medical_knowledge  — 医学知识库检索（支持 HyDE 查询转换 + 混合检索）
  get_weather               — 天气查询（高德 API）
  chat_general              — 通用闲聊兜底（Ollama LLM）

延迟导入: 通过 _get_vector_store / _get_llm 避免循环导入
"""
from langchain_core.tools import tool
from config.app_config import APP_CONFIG
from src.utils.text_utils import extract_city_from_text
from src.utils.logger_config import logger


def _get_vector_store():
    """延迟获取向量存储实例，避免与 system_initializer 循环导入"""
    from src.service.system_initializer import system_initializer
    return system_initializer.vector_store


def _get_llm():
    """延迟获取 ChatOllama 实例"""
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

    检索流程:
      1. 若 use_hyde 启用，先用 HyDE 将 query 转换为假设文档再检索
      2. 若 use_hybrid_search 启用，执行向量 + BM25 混合检索 + RRF 融合 + 重排序
      3. 否则执行纯向量相似度搜索

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

    先用 extract_city_from_text 做正则提取，
    失败后用 location 参数直接调用高德 API。

    Args:
        location: 城市名称，如 "北京"、"上海"
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

    仅当问题不涉及疾病症状、不涉及天气查询时由 Agent 工具路由触发。

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
