"""
分层记忆模块 — 长对话历史摘要与语义检索

核心机制:
  1. 当近期对话超过 memory_summary_turns 条时，将最旧的一批摘要后移出 Redis 列表
  2. 摘要使用 LLM 生成（医疗专用 prompt），保留症状、用药等关键信息
  3. 新轮次查询时用 OllamaEmbeddings 对历史摘要做余弦相似度检索
  4. 将相关摘要拼入近期对话历史，形成完整上下文输入给 Agent

Redis 存储结构:
  - chat:{session_id}:messages      近期对话（List，原有）
  - chat:{session_id}:summaries     历史摘要（List，新增）
      每个元素: {"summary": str, "start_time": str, "end_time": str, "turn_count": int}

配置项:
  - use_hierarchical_memory: 是否启用分层记忆
  - memory_summary_turns:    触发摘要的对话轮数阈值（默认 20）
  - memory_summary_max_age:  保留的最大摘要数（默认 30）
  - memory_retrieval_k:      检索返回的相关摘要数（默认 3）
"""
import json
from datetime import datetime
from typing import List, Optional, Dict
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


SUMMARY_PROMPT_CN = """请对以下对话进行医学摘要，保留所有关键医疗信息：

【对话记录】
{conversation}

【摘要要求】
- 保留症状描述、诊断意见、用药建议、检查结果等关键医疗信息
- 保留用户的主诉和病史
- 语言简洁，每条摘要控制在 100 字以内
- 如果对话中不包含医疗信息，仅输出"无医疗相关信息"

【医学摘要】"""


class MemorySummarizer:
    """
    分层记忆管理器

    职责:
      - 对话摘要生成与存储（check_and_summarize 定期触发）
      - 历史摘要语义检索（retrieve_relevant_summaries，余弦相似度 Top-K）
      - 拼接增强上下文（get_enhanced_history = 近期对话 + 相关摘要）
    """

    def __init__(self, memory_store, llm: ChatOllama = None):
        self.memory_store = memory_store
        self.llm = llm or ChatOllama(
            model=APP_CONFIG.llm_model_name,
            temperature=0.1,
            base_url=APP_CONFIG.llm_base_url
        )
        self._embedding = None

    # ────────── Embedding 懒加载（首次检索时初始化） ──────────

    @property
    def embedding(self):
        """OllamaEmbeddings 懒加载，仅在需要语义检索时创建"""
        if self._embedding is None:
            try:
                self._embedding = OllamaEmbeddings(
                    model=APP_CONFIG.embedding_model_name,
                    base_url=APP_CONFIG.llm_base_url
                )
            except Exception as e:
                logger.warning(f"⚠️ MemorySummarizer 嵌入模型加载失败: {e}")
        return self._embedding

    # ────────── Redis Key ──────────

    @staticmethod
    def _summary_key(session_id: str) -> str:
        return f"chat:{session_id}:summaries"

    # ────────── 摘要生成 ──────────

    def summarize_conversation(self, messages: List[Dict]) -> str:
        """
        调用 LLM 对一段对话进行医学摘要。

        Args:
            messages: 对话消息列表，每项含 role/content/timestamp

        Returns:
            摘要文本，失败时返回空字符串
        """
        if not messages:
            return ""

        conversation_text = ""
        for msg in messages:
            role = "用户" if msg.get("role") == "user" else "助手"
            content = msg.get("content", "")
            conversation_text += f"{role}: {content}\n"

        try:
            prompt = SUMMARY_PROMPT_CN.format(conversation=conversation_text)
            result = self.llm.invoke([HumanMessage(content=prompt)])
            summary = result.content if hasattr(result, "content") else str(result)
            logger.info(f"📝 对话摘要生成完成，长度: {len(summary)} 字")
            return summary
        except Exception as e:
            logger.warning(f"⚠️ 对话摘要生成失败: {e}")
            return ""

    # ────────── 摘要存储与检索 ──────────

    def store_summary(self, session_id: str, messages: List[Dict]):
        """
        对消息列表做摘要并存入 Redis List（rpush）。

        Args:
            session_id: 会话 ID
            messages:   待摘要的消息列表（摘要后从 messages 中清除）
        """
        if not messages:
            return

        summary_text = self.summarize_conversation(messages)
        if not summary_text:
            return

        summary_entry = {
            "summary": summary_text,
            "start_time": messages[0].get("timestamp", datetime.now().isoformat()) if messages else datetime.now().isoformat(),
            "end_time": messages[-1].get("timestamp", datetime.now().isoformat()) if messages else datetime.now().isoformat(),
            "turn_count": len(messages)
        }

        rc = self.memory_store.client
        if not rc:
            logger.warning("⚠️ Redis 不可用，摘要未持久化")
            return
        try:
            key = self._summary_key(session_id)
            rc.rpush(key, json.dumps(summary_entry, ensure_ascii=False))
            rc.expire(key, APP_CONFIG.redis_ttl)
            logger.info(f"✅ 摘要已存储，session_id={session_id}")
        except Exception as e:
            logger.warning(f"⚠️ 摘要存储失败: {e}")

    def get_all_summaries(self, session_id: str) -> List[Dict]:
        """
        从 Redis 获取全部历史摘要列表。

        Returns:
            [{"summary": str, "start_time": str, "end_time": str, "turn_count": int}, ...]
        """
        rc = self.memory_store.client
        if not rc:
            return []
        try:
            key = self._summary_key(session_id)
            raw_list = rc.lrange(key, 0, -1)
            return [json.loads(item) for item in raw_list]
        except Exception as e:
            logger.warning(f"⚠️ 读取摘要失败: {e}")
            return []

    # ────────── 语义检索 ──────────

    def retrieve_relevant_summaries(self, session_id: str, query: str, k: int = None) -> List[str]:
        """
        根据当前查询，从历史摘要中检索最相关的 Top-K 条。

        检索流程:
          1. 获取全部历史摘要（Redis lrange）
          2. query 编码为向量，各摘要编码为向量（OllamaEmbeddings）
          3. 余弦相似度排序 → 取 Top-K
          4. 若 embedding 不可用，直接返回最近 K 条作为回退

        Args:
            session_id: 会话 ID
            query:      当前用户问题
            k:          返回条数（默认 APP_CONFIG.memory_retrieval_k）

        Returns:
            相关摘要文本列表，按相似度降序
        """
        if k is None:
            k = APP_CONFIG.memory_retrieval_k

        summaries = self.get_all_summaries(session_id)
        if not summaries:
            return []

        embed_model = self.embedding
        if embed_model:
            try:
                query_vec = embed_model.embed_query(query)
                summary_texts = [s["summary"] for s in summaries]
                summary_vecs = embed_model.embed_documents(summary_texts)

                import numpy as np
                query_np = np.array(query_vec)
                scores = []
                for sv in summary_vecs:
                    sv_np = np.array(sv)
                    sim = np.dot(query_np, sv_np) / (np.linalg.norm(query_np) * np.linalg.norm(sv_np) + 1e-10)
                    scores.append(sim)

                ranked = sorted(zip(summary_texts, scores), key=lambda x: x[1], reverse=True)
                top_k = ranked[:k]
                logger.info(f"📚 语义检索到 {len(top_k)} 条相关摘要")
                return [item[0] for item in top_k]
            except Exception as e:
                logger.warning(f"⚠️ 摘要语义检索失败，回退返回全部: {e}")

        return [s["summary"] for s in summaries[:k]]

    # ────────── 便捷方法 ──────────

    def get_enhanced_history(self, session_id: str, query: str,
                             recent_n: int = None) -> str:
        """
        获取增强后的对话历史文本（近期对话 + 相关历史摘要）。

        当 use_hierarchical_memory 关闭时仅返回近期对话。
        检索到的摘要拼在近期对话末尾，构造格式:
          【历史对话摘要（与当前问题相关）】
          摘要 1: ...
          摘要 2: ...

        Args:
            session_id: 会话 ID
            query:      当前用户问题（用于检索相关摘要）
            recent_n:   返回的近期对话轮数

        Returns:
            合并后的对话历史文本
        """
        recent_text = self.memory_store.get_history_text(session_id, recent_n)

        if not APP_CONFIG.use_hierarchical_memory:
            return recent_text

        relevant_summaries = self.retrieve_relevant_summaries(session_id, query)
        if not relevant_summaries:
            return recent_text

        summary_section = "\n\n【历史对话摘要（与当前问题相关）】\n"
        for i, summary_text in enumerate(relevant_summaries, 1):
            summary_section += f"摘要 {i}: {summary_text}\n"

        return recent_text + summary_section

    # ────────── 管理接口 ──────────

    def check_and_summarize(self, session_id: str):
        """
        检查并触发摘要。

        触发条件:
          - use_hierarchical_memory 已启用
          - memory_summary_turns > 0
          - 当前消息数 > memory_summary_turns

        动作:
          1. 取最旧的 memory_summary_turns 条消息
          2. 送 LLM 生成摘要并存入 Redis summaries 列表
          3. 从 messages 列表中 ltrim 已摘要的旧消息
        """
        threshold = APP_CONFIG.memory_summary_turns
        if threshold <= 0:
            return

        messages = self.memory_store.get_recent_messages(session_id, n=threshold + 1)
        if len(messages) < threshold:
            return

        oldest_messages = messages[:threshold]
        self.store_summary(session_id, oldest_messages)

        rc = self.memory_store.client
        if not rc:
            return
        try:
            key = f"chat:{session_id}:messages"
            rc.ltrim(key, len(oldest_messages), -1)
            logger.info(f"🗑️ 已清理 {len(oldest_messages)} 条旧消息，移至摘要")
        except Exception as e:
            logger.warning(f"⚠️ 清理旧消息失败: {e}")


memory_summarizer = None
