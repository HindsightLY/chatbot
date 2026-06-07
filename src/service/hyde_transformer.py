"""
HyDE（假设文档嵌入）查询转换模块

原理:
  先用 LLM 生成一段假设性的"理想回答文档"，再用该文档的嵌入向量替代原始查询进行稠密检索。
  假设文档与知识库中的真实文档在语义空间上更接近，从而提升召回质量。

适用场景:
  - medical_inquiry 意图的稠密检索（ChromaDB 余弦相似度）
  - BM25 稀疏检索仍使用原始查询（基于关键词匹配，不受益于 HyDE）

配置项:
  - use_hyde: 是否启用 HyDE（config/app_config.py）
  - hyde_temperature: 生成假设文档时的温度参数
"""
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


HYDE_PROMPT_CN = """你是一位资深医学专家。请根据用户的问题，撰写一段假设性的医学回答。

这段回答将用于检索医学知识库中的真实资料，请确保内容详实、专业、全面，
涵盖可能的疾病、症状、病因、治疗建议等方面，篇幅控制在 300 字以内。

【用户问题】
{query}

【假设性医学回答】"""


class HyDEQueryTransformer:
    """
    HyDE 查询转换器

    职责:
      - 接收用户原始查询
      - 调用 LLM 生成假设性的医学回答
      - 返回假设回答文本（用于嵌入检索）

    用法:
      transformer = HyDEQueryTransformer()
      hyde_query = transformer.transform("头痛怎么办")
      results = vector_store.hybrid_search(hyde_query, ...)
    """

    def __init__(self, llm: ChatOllama = None):
        self.llm = llm or ChatOllama(
            model=APP_CONFIG.llm_model_name,
            temperature=0.1,
            base_url=APP_CONFIG.llm_base_url
        )

    def transform(self, query: str) -> str:
        """
        将用户原始查询转换为假设性医学文档

        Args:
            query: 用户原始查询

        Returns:
            假设性医学回答文本（用于替换原始查询进行稠密检索）
            若 LLM 调用失败，回退返回原始查询
        """
        if not query or not query.strip():
            return query

        try:
            prompt = HYDE_PROMPT_CN.format(query=query)
            result = self.llm.invoke([HumanMessage(content=prompt)])
            hyde_text = result.content if hasattr(result, "content") else str(result)
            logger.info(f"✅ HyDE 生成完成，长度: {len(hyde_text)} 字")
            return hyde_text
        except Exception as e:
            logger.warning(f"⚠️ HyDE 生成失败，回退原始查询: {e}")
            return query


hyde_transformer = HyDEQueryTransformer()
