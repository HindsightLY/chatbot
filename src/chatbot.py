"""
聊天机器人模块
提供基于RAG的医疗咨询功能
"""
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from config.app_config import APP_CONFIG
from langchain_ollama import OllamaLLM
from .logger_config import logger
from datetime import datetime
import os

# 全局存储历史
store = {}


class HybridChatMemory:
    """
    混合聊天记忆存储：内存缓存 + FAISS向量存储
    """

    def __init__(self, persist_dir="data/chat_memory"):
        self.memory_cache = {}  # 内存缓存：存储最近N轮对话
        self.persist_dir = persist_dir
        self.max_cache_turns = 10  # 每个会话在内存中缓存的最大轮数
        self.embedding = OllamaEmbeddings(
            model=APP_CONFIG.llm_model_name,
            base_url=APP_CONFIG.llm_base_url
        )
        self._init_faiss_store()

    def _init_faiss_store(self):
        """初始化FAISS向量存储"""
        os.makedirs(self.persist_dir, exist_ok=True)
        index_path = os.path.join(self.persist_dir, "index")

        if os.path.exists(index_path):
            try:
                self.vector_store = FAISS.load_local(
                    index_path,
                    self.embedding,
                    allow_dangerous_deserialization=True
                )
                print(f"✅ 加载历史对话向量库，包含 {self.vector_store.index.ntotal} 个对话")
            except Exception as e:
                print(f"⚠️ 加载历史对话失败: {e}，创建新的向量库")
                self.vector_store = self._create_empty_vector_store()
        else:
            self.vector_store = self._create_empty_vector_store()

    def _create_empty_vector_store(self):
        """创建空的FAISS向量库"""
        from langchain_core.documents import Document
        empty_doc = Document(page_content="empty", metadata={"session_id": "system"})
        return FAISS.from_documents([empty_doc], self.embedding)

    def _format_dialogue(self, human_msg, ai_msg, session_id):
        """格式化对话内容用于向量化"""
        return {
            "session_id": session_id,
            "human": human_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat(),
            "combined": f"用户: {human_msg}\n助手: {ai_msg}"
        }

    def add_dialogue(self, session_id, human_msg, ai_msg):
        """添加对话到混合存储"""
        # 1. 添加到内存缓存
        if session_id not in self.memory_cache:
            self.memory_cache[session_id] = []

        self.memory_cache[session_id].append({
            "human": human_msg,
            "ai": ai_msg,
            "timestamp": datetime.now().isoformat()
        })

        # 保持内存缓存大小
        if len(self.memory_cache[session_id]) > self.max_cache_turns:
            # 将最早的一轮对话保存到FAISS
            old_dialogue = self.memory_cache[session_id].pop(0)
            self._save_to_faiss(session_id, old_dialogue["human"], old_dialogue["ai"])

    def _save_to_faiss(self, session_id, human_msg, ai_msg):
        """保存对话到FAISS向量库"""
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

        # 添加到向量库
        self.vector_store.add_documents([doc])

        # 保存到磁盘
        index_path = os.path.join(self.persist_dir, "index")
        self.vector_store.save_local(index_path)
        print(f"💾 保存对话历史到FAISS，当前总数: {self.vector_store.index.ntotal}")

    def get_relevant_history(self, session_id, current_query, k=3):
        """获取相关的对话历史"""
        # 1. 先获取内存中的最近对话
        recent_history = self._get_recent_memory(session_id)

        # 2. 从FAISS中搜索相关历史对话
        relevant_history = self._search_faiss_history(current_query, session_id, k)

        # 3. 合并结果（最近的优先）
        combined_history = recent_history + relevant_history

        # 4. 格式化为字符串
        return self._format_history_for_prompt(combined_history)

    def _get_recent_memory(self, session_id):
        """获取内存中的最近对话"""
        if session_id not in self.memory_cache:
            return []

        return self.memory_cache[session_id][-self.max_cache_turns:]

    def _search_faiss_history(self, query, session_id, k=3):
        """从FAISS中搜索相关历史对话"""
        try:
            results = self.vector_store.similarity_search_with_score(
                query,
                k=k,
                filter={"session_id": session_id}  # 只搜索当前会话的历史
            )

            relevant_dialogues = []
            for doc, score in results:
                if score > 0.5:  # 相似度阈值
                    relevant_dialogues.append({
                        "human": doc.metadata["human"],
                        "ai": doc.metadata["ai"],
                        "timestamp": doc.metadata["timestamp"],
                        "relevance": score
                    })

            return relevant_dialogues
        except Exception as e:
            print(f"🔍 搜索历史对话失败: {e}")
            return []

    def _format_history_for_prompt(self, history_list):
        """格式化历史对话为prompt字符串"""
        if not history_list:
            return "无历史对话记录"

        formatted = []
        for i, dialogue in enumerate(history_list[-5:]):  # 只取最近5轮
            formatted.append(f"对话 {i + 1}:")
            formatted.append(f"用户: {dialogue['human']}")
            formatted.append(f"助手: {dialogue['ai']}")
            formatted.append("-" * 30)

        return "\n".join(formatted)

    def clear_session(self, session_id):
        """清除指定会话的记忆"""
        if session_id in self.memory_cache:
            del self.memory_cache[session_id]

        # 从FAISS中删除该会话的所有记录（简化版，实际需要重建索引）
        print(f"🧹 清除会话 {session_id} 的记忆")


class MedicalChatbot:
    """
    医疗聊天机器人
    基于RAG技术提供医疗咨询服务
    """

    def __init__(self, vector_store):
        """
        初始化医疗聊天机器人

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store

        # 使用Ollama LLM
        self.llm = OllamaLLM(
            model=APP_CONFIG.llm_model_name,
            temperature=APP_CONFIG.llm_temperature,
            base_url=APP_CONFIG.llm_base_url
        )

        # 初始化混合记忆存储
        self.hybrid_memory = HybridChatMemory()

        # 定义prompt（支持对话历史）
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

        # 创建文档链
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # 创建retriever
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 6, "score_threshold": 0.3}
        )

        # 创建最终检索链
        self.qa_chain = {
                            "context": self.retriever,
                            "input": lambda x: x["input"],
                            "relevant_history": lambda x: self.hybrid_memory.get_relevant_history(
                                x["session_id"],
                                x["input"]
                            )
                        } | self.prompt | self.llm

    def invoke_answer(self, question, session_id="default"):
        inputs = {
            "input": question,
            "session_id": session_id
        }
        result = self.qa_chain.invoke(inputs)

        # 保存对话到混合记忆
        if isinstance(result, dict) and "answer" in result:
            answer = result["answer"]
        elif hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)

        self.hybrid_memory.add_dialogue(session_id, question, answer)
        return answer

    def ask_stream(self, question, session_id="default"):
        """流式回答，逐字符输出"""
        logger.info("💬 医疗顾问回复：")
        try:
            answer = self.invoke_answer(question, session_id)
            # 流式输出
            for char in answer:
                yield char

        except Exception as e:
            logger.error(f"流式输出失败: {e}")
            yield "抱歉，我暂时无法回答这个问题。"

    def get_answer(self, question, session_id="default"):
        """
        非流式获取答案的方法

        Args:
            question: 问题
            session_id: 会话ID

        Returns:
            dict: 包含答案的字典
        """
        """非流式获取答案的方法"""
        try:
            answer = self.invoke_answer(question, session_id)
            return {"answer": answer}

        except Exception as e:
            logger.error(f"获取答案失败: {e}")
            return {"answer": "抱歉，我暂时无法回答这个问题。"}
