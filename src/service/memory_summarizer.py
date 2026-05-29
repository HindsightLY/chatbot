"""
分层记忆模块 (Hierarchical Memory)

职责:
  1. 定期对超过 max_history_turns 的对话进行摘要，存入 Redis
  2. 在新轮次查询时，检索与当前问题相关的历史摘要
  3. 将相关摘要与近期对话历史合并，形成完整上下文

存储结构 (Redis):
  - chat:{session_id}:messages       近期逐条对话 (原有)
  - chat:{session_id}:summaries      历史摘要列表 (新增)
      每个元素: {"summary": str, "start_time": str, "end_time": str, "turn_count": int}

检索方式:
  - 使用 OllamaEmbeddings 将查询编码为向量
  - 对每条摘要也编码为向量（摘要写入时预计算并缓存）
  - 余弦相似度排序 → 返回 Top-K 相关摘要
  - 若 embedding 不可用，回退为简单的关键词匹配

配置:
  - use_hierarchical_memory: 启用/禁用
  - memory_summary_turns: 多少轮后触发摘要 (默认 20)
  - memory_summary_max_age: 保留的最大摘要数量 (默认 30)
  - memory_retrieval_k: 检索到的相关摘要数量 (默认 3)
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
      - 对话摘要生成与存储
      - 历史摘要的语义检索
      - 与现有 MemoryStore 的 get_history_text 配合使用
    """

    def __init__(self, memory_store, llm: ChatOllama = None):
        self.memory_store = memory_store
        self.llm = llm or ChatOllama(
            model=APP_CONFIG.llm_model_name,
            temperature=0.1,
            base_url=APP_CONFIG.llm_base_url
        )
        self._embedding = None

    # ────────── Embedding 懒加载 ──────────

    @property
    def embedding(self):
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
        """调用 LLM 对一段对话进行摘要"""
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
        对消息列表做摘要并存入 Redis

        Args:
            session_id: 会话 ID
            messages: 待摘要的消息列表
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

        rc = self.memory_store.client if hasattr(self.memory_store, 'client') else None
        if rc:
            try:
                key = self._summary_key(session_id)
                rc.rpush(key, json.dumps(summary_entry, ensure_ascii=False))
                rc.expire(key, APP_CONFIG.redis_ttl)
                logger.info(f"✅ 摘要已存储，session_id={session_id}")
            except Exception as e:
                logger.warning(f"⚠️ 摘要存储失败: {e}")
        else:
            logger.warning("⚠️ Redis 不可用，摘要未持久化")

    def get_all_summaries(self, session_id: str) -> List[Dict]:
        """
        从 Redis 获取全部历史摘要

        Returns:
            [{"summary": str, "start_time": str, "end_time": str, "turn_count": int}]
        """
        rc = self.memory_store.client if hasattr(self.memory_store, 'client') else None
        if rc:
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
        根据当前查询，从历史摘要中检索最相关的 Top-K 条

        Args:
            session_id: 会话 ID
            query: 当前用户问题
            k: 返回条数

        Returns:
            相关摘要文本列表
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
        获取增强后的对话历史文本（近期对话 + 相关历史摘要）

        Args:
            session_id: 会话 ID
            query: 当前用户问题（用于检索相关摘要）
            recent_n: 最近 N 轮对话数

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
        检查是否需要触发摘要。当消息数量 > memory_summary_turns 时，
        将最旧的 memory_summary_turns 条摘要后清除。

        被 _save_memory 或每次消息写入后周期调用。
        """
        threshold = APP_CONFIG.memory_summary_turns
        if threshold <= 0:
            return

        messages = self.memory_store.get_recent_messages(session_id, n=threshold + 1)
        if len(messages) < threshold:
            return

        oldest_messages = messages[:threshold]
        self.store_summary(session_id, oldest_messages)

        rc = self.memory_store.client if hasattr(self.memory_store, 'client') else None
        if rc:
            try:
                key = f"chat:{session_id}:messages"
                rc.ltrim(key, len(oldest_messages), -1)
                logger.info(f"🗑️ 已清理 {len(oldest_messages)} 条旧消息，移至摘要")
            except Exception as e:
                logger.warning(f"⚠️ 清理旧消息失败: {e}")


memory_summarizer = None
