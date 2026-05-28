"""
LangGraph 智能体模块
实现基于 LangGraph StateGraph 的多节点 Agent，替代原有 LCEL 链

节点流转:
  classify_intent → conditional
    ├─ medical_inquiry → retrieve_docs → generate_answer → save_memory
    ├─ chat_general+天气 → weather_query → save_memory
    ├─ chat_general/other → general_chat → save_memory
    └─ unknown → retrieve_docs (兜底 RAG)

关键设计: 所有分支（RAG / 闲聊 / 天气）在构造 prompt 时均注入对话历史
  self.prompt          → {history} + {context} + {input}   (RAG)
  self.general_prompt  → {history} + {input}               (闲聊 / 天气)
  确保多轮对话中用户提到的信息（如姓名、症状等）可被后续轮次引用
"""
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from config.app_config import APP_CONFIG
from src.utils.text_utils import is_weather_query
from src.utils.logger_config import logger


class AgentState(TypedDict):
    """LangGraph 状态定义"""
    messages: List[Dict[str, str]]
    session_id: str
    intent: str
    context_docs: List[str]
    answer: str


class MedicalAgent:
    """
    医疗咨询智能体

    管理 LangGraph 状态图的生命周期:
      - 构建图结构
      - 注入外部依赖（LLM / 向量库 / 记忆存储 / 意图分类器 / 工具）
      - 执行单轮推理
    """

    def __init__(self, vector_store, memory_store, intent_classifier, tool_manager):
        """
        Args:
            vector_store: ChromaDB 实例（来自 VectorStoreManager）
            memory_store: MemoryStore 实例（Redis）
            intent_classifier: IntentClassifier 实例
            tool_manager: ToolManager 实例
        """
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.intent_classifier = intent_classifier
        self.tool_manager = tool_manager

        self.llm = OllamaLLM(
            model=APP_CONFIG.llm_model_name,
            temperature=APP_CONFIG.llm_temperature,
            base_url=APP_CONFIG.llm_base_url
        )

        self.prompt = PromptTemplate.from_template(
            """你是一位专业医疗顾问。请根据以下医学资料和对话历史回答用户问题。
若资料中无直接匹配，请基于医学常识谨慎推断，但需注明"可能"、"常见原因包括"等措辞。

【相关对话历史】
{history}

【医学资料】
{context}

【当前问题】
{input}

请直接给出清晰、专业的回答，分点说明可能疾病、症状关联与建议。"""
        )

        self.general_prompt = PromptTemplate.from_template(
            """你是一个友好的AI助手。请根据以下对话历史和当前问题，回答用户的问题。

【对话历史】
{history}

【当前问题】
{input}

请直接回答用户的问题，语言亲切自然。"""
        )

        self.graph = self._build_graph()

    # ────────── 节点函数 ──────────

    def _classify_intent(self, state: AgentState) -> AgentState:
        """节点: 意图分类"""
        messages = state["messages"]
        last_content = messages[-1]["content"] if messages else ""
        intent = self.intent_classifier.classify(last_content)
        logger.info(f"🔍 Agent 识别意图: {intent}")
        return {**state, "intent": intent}

    def _retrieve_docs(self, state: AgentState) -> AgentState:
        """节点: 向量库检索"""
        messages = state["messages"]
        query = messages[-1]["content"] if messages else ""

        if not self.vector_store:
            logger.warning("⚠️ 向量库未就绪，跳过检索")
            return {**state, "context_docs": []}

        try:
            results = self.vector_store.similarity_search(
                query, k=APP_CONFIG.retrieval_k,
                score_threshold=APP_CONFIG.retrieval_score_threshold
            )
            docs = [doc.page_content for doc in results]
            logger.info(f"📚 检索到 {len(docs)} 篇相关文档")
            return {**state, "context_docs": docs}
        except Exception as e:
            logger.exception(f"❌ 检索失败")
            return {**state, "context_docs": []}

    def _generate_answer(self, state: AgentState) -> AgentState:
        """节点: RAG 生成回答"""
        messages = state["messages"]
        last_content = messages[-1]["content"] if messages else ""

        context_text = "\n\n".join(state.get("context_docs", [])) or "未找到相关医学资料"
        history_text = self.memory_store.get_history_text(state["session_id"])

        formatted_prompt = self.prompt.format(
            history=history_text,
            context=context_text,
            input=last_content
        )

        try:
            answer = self.llm.invoke(formatted_prompt)
            logger.info(f"💡 RAG 生成完成，长度: {len(answer)}")
            return {**state, "answer": answer}
        except Exception as e:
            logger.exception(f"❌ LLM 生成失败")
            return {**state, "answer": "抱歉，我暂时无法回答这个问题。"}

    def _weather_query(self, state: AgentState) -> AgentState:
        """节点: 天气查询 — 带对话历史的润色输出"""
        messages = state["messages"]
        query = messages[-1]["content"] if messages else ""
        history = self.memory_store.get_history_text(state["session_id"])

        try:
            weather_data = self.tool_manager.get_weather_response(query)
            formatted_prompt = self.general_prompt.format(
                history=history,
                input=f"用户问天气：{query}\n天气数据：{weather_data}"
            )
            answer = self.llm.invoke(formatted_prompt)
            return {**state, "answer": answer}
        except Exception as e:
            logger.exception(f"❌ 天气查询失败")
            return {**state, "answer": "获取天气信息失败，请稍后再试。"}

    def _general_chat(self, state: AgentState) -> AgentState:
        """节点: 通用闲聊/系统查询 — 带对话历史的完整上下文"""
        messages = state["messages"]
        query = messages[-1]["content"] if messages else ""
        history = self.memory_store.get_history_text(state["session_id"])

        try:
            formatted_prompt = self.general_prompt.format(
                history=history,
                input=query
            )
            answer = self.llm.invoke(formatted_prompt)
            return {**state, "answer": answer}
        except Exception as e:
            logger.exception(f"❌ 通用对话失败")
            return {**state, "answer": "抱歉，我暂时无法回答这个问题。"}

    def _save_memory(self, state: AgentState) -> AgentState:
        """节点: 记忆持久化到 Redis"""
        messages = state["messages"]
        last_content = messages[-1]["content"] if messages else ""

        self.memory_store.add_message(state["session_id"], "user", last_content)
        self.memory_store.add_message(state["session_id"], "assistant", state["answer"])
        return state

    # ────────── 路由 ──────────

    def _route_by_intent(self, state: AgentState) -> str:
        """根据意图条件路由到下一节点"""
        intent = state.get("intent", "unknown")

        if intent == "medical_inquiry" or intent == "unknown":
            return "retrieve_docs"

        if intent == "chat_general":
            messages = state["messages"]
            last_content = messages[-1]["content"] if messages else ""
            if is_weather_query(last_content):
                return "weather_query"
            return "general_chat"

        # system_query 或其他
        return "general_chat"

    # ────────── 图构建 ──────────

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图"""
        builder = StateGraph(AgentState)

        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("retrieve_docs", self._retrieve_docs)
        builder.add_node("generate_answer", self._generate_answer)
        builder.add_node("weather_query", self._weather_query)
        builder.add_node("general_chat", self._general_chat)
        builder.add_node("save_memory", self._save_memory)

        builder.set_entry_point("classify_intent")

        builder.add_conditional_edges(
            "classify_intent",
            self._route_by_intent,
            {
                "retrieve_docs": "retrieve_docs",
                "weather_query": "weather_query",
                "general_chat": "general_chat",
            }
        )

        builder.add_edge("retrieve_docs", "generate_answer")
        builder.add_edge("generate_answer", "save_memory")
        builder.add_edge("weather_query", "save_memory")
        builder.add_edge("general_chat", "save_memory")
        builder.add_edge("save_memory", END)

        return builder.compile(checkpointer=MemorySaver())

    # ────────── 推理入口 ──────────

    def run(self, user_input: str, session_id: str = "default") -> str:
        """
        执行单轮 Agent 推理（同步阻塞，返回完整回答）

        Args:
            user_input: 用户输入
            session_id: 会话 ID

        Returns:
            回答文本
        """
        initial_state: AgentState = {
            "messages": [{"role": "user", "content": user_input}],
            "session_id": session_id,
            "intent": "",
            "context_docs": [],
            "answer": ""
        }

        thread_config = {"configurable": {"thread_id": session_id}}

        try:
            final_state = self.graph.invoke(initial_state, config=thread_config)
            return final_state.get("answer", "抱歉，我暂时无法回答这个问题。")
        except Exception as e:
            logger.exception(f"❌ Agent 推理失败")
            return "抱歉，系统处理出现异常，请稍后再试。"

    def run_stream(self, user_input: str, session_id: str = "default"):
        """
        流式推理入口，逐 token 产出，无需等待 LLM 完全生成

        所有分支在生成 prompt 前均从 Redis 拉取对话历史，
        确保多轮记忆（姓名、既往症状等）被带入当前上下文。

        Yields:
            {"type": "intent", "content": str}  — 意图事件（第一个发出）
            {"type": "token",  "content": str}  — LLM 输出片段
            {"type": "done"}                     — 结束信号
        """
        try:
            # === 1. 意图分类 ===
            intent = self.intent_classifier.classify(user_input)
            logger.info(f"🔍 Agent 识别意图: {intent}")
            yield {"type": "intent", "content": intent}

            full_answer = ""

            # === 2. 按意图路由 ===
            if intent == "medical_inquiry" or intent == "unknown":
                # ---- 2a. 检索文档 ----
                docs = []
                if self.vector_store:
                    try:
                        results = self.vector_store.similarity_search(
                            user_input, k=APP_CONFIG.retrieval_k,
                            score_threshold=APP_CONFIG.retrieval_score_threshold
                        )
                        docs = [doc.page_content for doc in results]
                        logger.info(f"📚 检索到 {len(docs)} 篇相关文档")
                    except Exception as e:
                        logger.exception(f"❌ 检索失败")

                context_text = "\n\n".join(docs) or "未找到相关医学资料"
                history_text = self.memory_store.get_history_text(session_id)

                formatted_prompt = self.prompt.format(
                    history=history_text,
                    context=context_text,
                    input=user_input
                )

                # ---- 2b. 流式 LLM 生成 ----
                for token in self.llm.stream(formatted_prompt):
                    full_answer += token
                    yield {"type": "token", "content": token}

            elif intent == "chat_general" and is_weather_query(user_input):
                history_text = self.memory_store.get_history_text(session_id)
                weather_data = self.tool_manager.get_weather_response(user_input)
                weather_prompt = self.general_prompt.format(
                    history=history_text,
                    input=f"用户问天气：{user_input}\n天气数据：{weather_data}"
                )
                for token in self.llm.stream(weather_prompt):
                    full_answer += token
                    yield {"type": "token", "content": token}

            elif intent == "chat_general" or intent == "system_query":
                history_text = self.memory_store.get_history_text(session_id)
                general_prompt = self.general_prompt.format(
                    history=history_text,
                    input=user_input
                )
                for token in self.llm.stream(general_prompt):
                    full_answer += token
                    yield {"type": "token", "content": token}

            else:
                # 兜底走 RAG
                for token in self.llm.stream(
                    self.prompt.format(
                        history=self.memory_store.get_history_text(session_id),
                        context="未找到相关医学资料",
                        input=user_input
                    )
                ):
                    full_answer += token
                    yield {"type": "token", "content": token}

            # === 3. 记忆持久化 ===
            self.memory_store.add_message(session_id, "user", user_input)
            self.memory_store.add_message(session_id, "assistant", full_answer)

            logger.info(f"💡 流式生成完成，长度: {len(full_answer)}")
            yield {"type": "done"}

        except Exception as e:
            logger.exception(f"❌ Agent 流式推理失败")
            yield {"type": "token", "content": "抱歉，系统处理出现异常，请稍后再试。"}
            yield {"type": "done"}
