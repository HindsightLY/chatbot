"""
聊天机器人模块
提供基于 RAG 的医疗咨询功能

调用链:
  intent_classifier.classify() → 判定意图
    → medical_inquiry → MedicalChatbot.get_answer()  [本模块]
    → chat_general   → ToolManager                   [tool_manager.py]
"""
from operator import itemgetter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger
from datetime import datetime
import os


class HybridChatMemory:
    """
    混合聊天记忆存储：内存缓存 + FAISS 向量存储

    两级存储策略:
      内存层 — 缓存最近 N 轮对话，读写零延迟
      FAISS 层 — 缓存溢出时写入磁盘，支持语义检索
    """

    def __init__(self, persist_dir="../data/chat_memory"):
        self.memory_cache = {}
        self.persist_dir = persist_dir
        self.max_cache_turns = 10
        self.embedding = OllamaEmbeddings(
            model=APP_CONFIG.embedding_model_name,
            base_url=APP_CONFIG.llm_base_url
        )
        self._init_faiss_store()

    def _init_faiss_store(self):
        """初始化或加载已有的 FAISS 向量库"""
        os.makedirs(self.persist_dir, exist_ok=True)
        index_path = os.path.join(self.persist_dir, "index")

        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embedding,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"✅ 加载历史对话向量库，包含 {self.vector_store.index.ntotal} 个对话")
            except Exception as e:
                logger.info(f"⚠️ 加载历史对话失败: {e}，创建新的向量库")
                self.vector_store = self._create_empty_vector_store()
        else:
            self.vector_store = self._create_empty_vector_store()

    def _create_empty_vector_store(self):
        """创建空的 FAISS 向量库（FAISS 要求至少有一条记录）"""
        try:
            empty_doc = Document(page_content="empty", metadata={"session_id": "system"})
            return FAISS.from_documents([empty_doc], self.embedding)
        except Exception as e:
            logger.error(f"❌ 创建空向量库失败: {e}")
            raise

    def _format_dialogue(self, human_msg, ai_msg, session_id):
        """构造可向量化的对话记录"""
        return {
            "session_id": session_id,
            "human": human_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat(),
            "combined": f"用户: {human_msg}\n助手: {ai_msg}"
        }

    def add_dialogue(self, session_id, human_msg, ai_msg):
        """
        添加一轮对话到混合存储

        策略: 优先写入内存；内存超过上限时将最早记录迁移到 FAISS
        """
        if session_id not in self.memory_cache:
            self.memory_cache[session_id] = []

        self.memory_cache[session_id].append({
            "human": human_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat()
        })

        if len(self.memory_cache[session_id]) > self.max_cache_turns:
            old_dialogue = self.memory_cache[session_id].pop(0)
            self._save_to_faiss(session_id, old_dialogue["human"], old_dialogue["ai"])

    def _save_to_faiss(self, session_id, human_msg, ai_msg):
        """将对话持久化到 FAISS 并同步到磁盘（失败不影响内存缓存）"""
        try:
            formatted = self._format_dialogue(human_msg, ai_msg, session_id)
            doc = Document(
                page_content=formatted["combined"],
                metadata={
                    "session_id": session_id,
                    "timestamp": formatted["timestamp"],
                    "human": human_msg,
                    "ai": ai_msg
                }
            )

            self.vector_store.add_documents([doc])

            index_path = os.path.join(self.persist_dir, "index")
            self.vector_store.save_local(index_path)
            logger.info(f"💾 保存对话历史到FAISS，当前总数: {self.vector_store.index.ntotal}")
        except Exception as e:
            logger.warning(f"⚠️ 对话历史持久化失败（对话仍保留在内存）: {e}")

    def get_relevant_history(self, session_id, current_query, k=3):
        """
        合并内存 + FAISS 两路历史记录，作为 prompt 上下文

        召回优先级: 最近 > 语义相关
        """
        recent_history = self._get_recent_memory(session_id)
        relevant_history = self._search_faiss_history(current_query, session_id, k)
        combined_history = recent_history + relevant_history
        return self._format_history_for_prompt(combined_history)

    def _get_recent_memory(self, session_id):
        """从内存取出最近 N 轮对话"""
        if session_id not in self.memory_cache:
            return []
        return self.memory_cache[session_id][-self.max_cache_turns:]

    def _search_faiss_history(self, query, session_id, k=3):
        """
        从 FAISS 中按语义相似度召回历史对话

        用 filter 限定当前 session，避免跨会话干扰
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"session_id": session_id}
            )

            relevant_dialogues = []
            for doc, score in results:
                if score > 0.5:
                    relevant_dialogues.append({
                        "human": doc.metadata["human"],
                        "ai": doc.metadata["ai"],
                        "timestamp": doc.metadata["timestamp"],
                        "relevance": score
                    })

            return relevant_dialogues
        except Exception as e:
            logger.warning(f"🔍 搜索历史对话失败（不影响主流程）: {e}")
            return []

    def _format_history_for_prompt(self, history_list):
        """将对话记录拼接为 prompt 可读的文本块（最多 5 轮）"""
        if not history_list:
            return "无历史对话记录"

        formatted = []
        for i, dialogue in enumerate(history_list[-5:]):
            formatted.append(f"对话 {i + 1}:")
            formatted.append(f"用户: {dialogue['human']}")
            formatted.append(f"助手: {dialogue['ai']}")
            formatted.append("-" * 30)

        return "\n".join(formatted)

    def clear_session(self, session_id):
        """清除会话内存缓存（FAISS 端需重建索引才能彻底删除）"""
        if session_id in self.memory_cache:
            del self.memory_cache[session_id]

        logger.info(f"🧹 清除会话 {session_id} 的记忆")


class MedicalChatbot:
    """
    医疗聊天机器人

    基于 LangChain 表达式语言构建 RAG 链:
      retriever(检索) → prompt(组装) → llm(生成)

    依赖:
      - VectorStoreManager 提供向量检索能力   [vector_store.py]
      - HybridChatMemory 提供对话记忆         [本模块]
    """

    def __init__(self, vector_store):
        """
        Args:
            vector_store: 由 VectorStoreManager 创建或加载的 FAISS 实例
        """
        self.vector_store = vector_store

        self.llm = OllamaLLM(
            model=APP_CONFIG.llm_model_name,
            temperature=APP_CONFIG.llm_temperature,
            base_url=APP_CONFIG.llm_base_url
        )

        self.hybrid_memory = HybridChatMemory()

        self.prompt = PromptTemplate.from_template(
            """你是一位专业医疗顾问。请根据以下医学资料和对话历史回答用户问题。
            若资料中无直接匹配，请基于医学常识谨慎推断，但需注明"可能"、"常见原因包括"等措辞。

            【相关对话历史】
            {relevant_history}

            【医学资料】
            {context}

            【当前问题】
            {input}

            请直接给出清晰、专业的回答，分点说明可能疾病、症状关联与建议。
            """
        )

        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 6, "score_threshold": 0.3}
        )

        # LCEL 管道: dict → prompt → llm
        self.qa_chain = {
                            "context": itemgetter("input") | self.retriever,
                            "input": itemgetter("input"),
                            "relevant_history": lambda x: self.hybrid_memory.get_relevant_history(
                                x["session_id"],
                                x["input"]
                            )
                        } | self.prompt | self.llm

    def get_answer(self, question, session_id="default"):
        """
        执行完整 RAG 流程: 检索 → 组装 → 生成

        同时将问答对写回记忆存储，供后续对话参考。
        """
        try:
            inputs = {"input": question, "session_id": session_id}
            result = self.qa_chain.invoke(inputs)

            if isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            elif hasattr(result, 'content'):
                answer = result.content
            else:
                answer = str(result)

            self.hybrid_memory.add_dialogue(session_id, question, answer)
            return {"answer": answer}

        except Exception as e:
            logger.exception(f"获取答案失败")
            return {"answer": "抱歉，我暂时无法回答这个问题。"}

    def ask_stream(self, question, session_id="default"):
        """简化版流式接口（当前为全量返回后逐字符 yield）"""
        result = self.get_answer(question, session_id)
        yield result["answer"]
