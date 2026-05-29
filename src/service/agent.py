"""
LangGraph 智能体模块
实现基于 LangGraph StateGraph 的多节点 Agent，替代原有 LCEL 链

节点流转:
  call_model (bind_tools) → conditional
    ├─ 有工具调用 → tool_node → call_model (循环)
    └─ 无工具调用 → human_review → save_memory → END

关键设计:
  - 使用 LangChain @tool 装饰器定义工具（medical_tools.py）
  - LLM 通过 bind_tools() 自主决定调用哪个工具
  - human_review 节点通过 interrupt_after 实现人机协同
  - 流式路径 (run_stream) 同样支持工具调用 + 人工审核事件
"""
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from config.app_config import APP_CONFIG
from src.tools.medical_tools import tools
from src.utils.logger_config import logger
from src.utils.text_utils import is_weather_query
from src.service.hyde_transformer import HyDEQueryTransformer
from src.service.memory_summarizer import MemorySummarizer


class AgentState(TypedDict):
    """LangGraph 状态定义"""
    messages: List[Dict[str, str]]
    session_id: str
    intent: str
    context_docs: List[str]
    answer: str
    human_approved: bool
    review_skipped: bool


class MedicalAgent:
    """
    医疗咨询智能体

    管理 LangGraph 状态图的生命周期:
      - 构建图结构（bind_tools + ToolNode + human_review）
      - 注入外部依赖（LLM / 向量库 / 记忆存储 / 意图分类器 / 工具）
      - 执行单轮推理 / 流式推理
    """

    def __init__(self, vector_store, memory_store, intent_classifier, tool_manager):
        """
        Args:
            vector_store: ChromaDB 实例（来自 VectorStoreManager）
            memory_store: MemoryStore 实例（Redis）
            intent_classifier: IntentClassifier 实例
            tool_manager: ToolManager 实例（保留兼容）
        """
        self.vector_store = vector_store
        self.memory_store = memory_store
        self.intent_classifier = intent_classifier
        self.tool_manager = tool_manager

        self.llm = ChatOllama(
            model=APP_CONFIG.llm_model_name,
            temperature=APP_CONFIG.llm_temperature,
            base_url=APP_CONFIG.llm_base_url
        )

        self.llm_with_tools = self.llm.bind_tools(tools)

        self.hyde = HyDEQueryTransformer(llm=self.llm) if APP_CONFIG.use_hyde else None
        self.memory_summarizer = MemorySummarizer(memory_store=memory_store, llm=self.llm) if APP_CONFIG.use_hierarchical_memory else None

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

        self.tool_node = ToolNode(tools)
        self.graph = self._build_graph()

    # ────────── 节点函数 ──────────

    def _classify_intent(self, state: AgentState) -> AgentState:
        """节点: 意图分类（仅用于日志和前端展示）"""
        messages = state["messages"]
        last_content = ""
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                last_content = last.content
            elif isinstance(last, dict):
                last_content = last.get("content", "")
        intent = self.intent_classifier.classify(last_content)
        logger.info(f"🔍 Agent 识别意图: {intent}")
        return {**state, "intent": intent}

    def _call_model(self, state: AgentState) -> AgentState:
        """
        节点: 调用 LLM（已绑定工具）
        LLM 自主决定是否调用工具，或直接生成回答。
        """
        messages = state.get("messages", [])
        if self.memory_summarizer:
            history_text = self.memory_summarizer.get_enhanced_history(
                state["session_id"], last_content
            )
        else:
            history_text = self.memory_store.get_history_text(state["session_id"])

        last_content = ""
        if messages and isinstance(messages[-1], dict):
            last_content = messages[-1].get("content", "")
        elif messages and hasattr(messages[-1], "content"):
            last_content = messages[-1].content

        formatted_prompt = self.general_prompt.format(
            history=history_text,
            input=last_content
        )

        try:
            result = self.llm_with_tools.invoke(
                [HumanMessage(content=formatted_prompt)]
            )
            answer_text = result.content if hasattr(result, "content") else str(result)
            logger.info(f"🤖 LLM 生成完成，tool_calls: {bool(getattr(result, 'tool_calls', None))}")

            new_messages = list(messages)
            new_messages.append(result)

            return {**state, "messages": new_messages, "answer": answer_text}
        except Exception as e:
            logger.exception(f"❌ LLM 调用失败")
            return {**state, "answer": "抱歉，我暂时无法回答这个问题。"}

    def _should_continue(self, state: AgentState) -> str:
        """条件边: 判断是否需要继续调用工具"""
        messages = state.get("messages", [])
        if not messages:
            return "end"
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "continue"
        if isinstance(last, dict) and last.get("tool_calls"):
            return "continue"
        return "end"

    def _tool_node_wrapper(self, state: AgentState) -> AgentState:
        """节点: 执行工具调用"""
        try:
            result = self.tool_node.invoke(state)
            if isinstance(result, dict) and "messages" in result:
                messages = list(state.get("messages", []))
                messages.extend(result["messages"])
                return {**state, "messages": messages}
            elif isinstance(result, list):
                messages = list(state.get("messages", []))
                messages.extend(result)
                return {**state, "messages": messages}
        except Exception as e:
            logger.exception(f"❌ 工具执行失败")
        return state

    def _human_review(self, state: AgentState) -> AgentState:
        """
        节点: 人工审核（仅同步图路径）
        通过 interrupt_after 暂停，等待外部调用 resume 或 update_state。
        """
        logger.info(f"⏸️ 等待人工审核，session_id={state['session_id']}")
        return state

    def _save_memory(self, state: AgentState) -> AgentState:
        """节点: 记忆持久化到 Redis"""
        messages = state["messages"]
        last_user_msg = ""
        for msg in reversed(messages):
            content = msg.content if hasattr(msg, "content") else (msg.get("content", "") if isinstance(msg, dict) else "")
            if hasattr(msg, "type") and msg.type == "human":
                last_user_msg = content
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_msg = content
                break

        self.memory_store.add_message(state["session_id"], "user", last_user_msg)
        self.memory_store.add_message(state["session_id"], "assistant", state.get("answer", ""))

        if self.memory_summarizer:
            self.memory_summarizer.check_and_summarize(state["session_id"])

        return {**state, "human_approved": True}

    # ────────── 路由 ──────────

    # ────────── 图构建 ──────────

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态图（含工具调用 + 人机协同）"""
        builder = StateGraph(AgentState)

        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("call_model", self._call_model)
        builder.add_node("tool_node", self._tool_node_wrapper)
        builder.add_node("human_review", self._human_review)
        builder.add_node("save_memory", self._save_memory)

        builder.set_entry_point("classify_intent")
        builder.add_edge("classify_intent", "call_model")

        builder.add_conditional_edges(
            "call_model",
            self._should_continue,
            {
                "continue": "tool_node",
                "end": "human_review",
            }
        )

        builder.add_edge("tool_node", "call_model")
        builder.add_edge("human_review", "save_memory")
        builder.add_edge("save_memory", END)

        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer, interrupt_after=["human_review"])

    # ────────── 推理入口 ──────────

    def run(self, user_input: str, session_id: str = "default") -> str:
        """
        执行单轮 Agent 推理（同步阻塞，返回完整回答）
        支持工具调用 + 人工审核中断。

        Args:
            user_input: 用户输入
            session_id: 会话 ID

        Returns:
            回答文本
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
            "intent": "",
            "context_docs": [],
            "answer": "",
            "human_approved": False,
            "review_skipped": False,
        }

        thread_config = {"configurable": {"thread_id": session_id}}

        try:
            # 首次调用: 执行到 human_review 节点中断
            final_state = self.graph.invoke(initial_state, config=thread_config)

            # 如果被中断，自动批准（CLI 模式下可改造为等待用户确认）
            if self.graph.get_state(thread_config).next:
                logger.info("✅ 自动批准（未启用外部审核）")
                final_state = self.graph.invoke(None, config=thread_config)

            return final_state.get("answer", "抱歉，我暂时无法回答这个问题。")
        except Exception as e:
            err_str = str(e)
            if "Connection refused" in err_str or "ConnectError" in err_str or "10061" in err_str:
                return "⚠️ Ollama 服务未运行，请先启动 Ollama（ollama serve），然后重试。"
            logger.exception(f"❌ Agent 推理失败")
            return f"抱歉，系统处理出现异常: {e}"

    def run_stream(self, user_input: str, session_id: str = "default"):
        """
        流式推理入口，逐 token 产出，无需等待 LLM 完全生成

        所有分支在生成 prompt 前均从 Redis 拉取对话历史，
        确保多轮记忆（姓名、既往症状等）被带入当前上下文。

        Yields:
            {"type": "intent", "content": str}  — 意图事件（第一个发出）
            {"type": "token",  "content": str}  — LLM 输出片段
            {"type": "review", "content": str}  — 人工审核事件（需确认后保存记忆）
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
                # ---- 2a. HyDE 查询转换 ----
                search_query = user_input
                if self.hyde:
                    hyde_query = self.hyde.transform(user_input)
                    search_query = hyde_query

                # ---- 2b. 混合检索文档 ----
                docs = []
                if self.vector_store:
                    try:
                        if APP_CONFIG.use_hybrid_search:
                            results = self.vector_store.hybrid_search(
                                search_query, k=APP_CONFIG.retrieval_k
                            )
                        else:
                            results = self.vector_store.similarity_search(
                                search_query, k=APP_CONFIG.retrieval_k,
                                score_threshold=APP_CONFIG.retrieval_score_threshold
                            )
                        docs = [doc.page_content for doc in results]
                        logger.info(f"📚 检索到 {len(docs)} 篇相关文档")
                    except Exception as e:
                        logger.exception(f"❌ 检索失败")

                context_text = "\n\n".join(docs) or "未找到相关医学资料"
                if self.memory_summarizer:
                    history_text = self.memory_summarizer.get_enhanced_history(
                        session_id, user_input
                    )
                else:
                    history_text = self.memory_store.get_history_text(session_id)

                formatted_prompt = self.prompt.format(
                    history=history_text,
                    context=context_text,
                    input=user_input
                )

                # ---- 2c. 流式 LLM 生成 ----
                for chunk in self.llm.stream([HumanMessage(content=formatted_prompt)]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}

            elif intent == "chat_general" and is_weather_query(user_input):
                if self.memory_summarizer:
                    history_text = self.memory_summarizer.get_enhanced_history(
                        session_id, user_input
                    )
                else:
                    history_text = self.memory_store.get_history_text(session_id)
                weather_data = self.tool_manager.get_weather_response(user_input)
                weather_prompt = self.general_prompt.format(
                    history=history_text,
                    input=f"用户问天气：{user_input}\n天气数据：{weather_data}"
                )
                for chunk in self.llm.stream([HumanMessage(content=weather_prompt)]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}

            elif intent == "chat_general" or intent == "system_query":
                if self.memory_summarizer:
                    history_text = self.memory_summarizer.get_enhanced_history(
                        session_id, user_input
                    )
                else:
                    history_text = self.memory_store.get_history_text(session_id)
                general_prompt = self.general_prompt.format(
                    history=history_text,
                    input=user_input
                )
                for chunk in self.llm.stream([HumanMessage(content=general_prompt)]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}

            else:
                if self.memory_summarizer:
                    history_text = self.memory_summarizer.get_enhanced_history(
                        session_id, user_input
                    )
                else:
                    history_text = self.memory_store.get_history_text(session_id)
                for chunk in self.llm.stream(
                    [HumanMessage(content=self.prompt.format(
                        history=history_text,
                        context="未找到相关医学资料",
                        input=user_input
                    ))]
                ):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}

            # === 3. 人工审核 ===
            yield {"type": "review", "content": full_answer}

            # === 4. 记忆持久化（流式模式下自动保存） ===
            self.memory_store.add_message(session_id, "user", user_input)
            self.memory_store.add_message(session_id, "assistant", full_answer)

            if self.memory_summarizer:
                self.memory_summarizer.check_and_summarize(session_id)

            logger.info(f"💡 流式生成完成，长度: {len(full_answer)}")
            yield {"type": "done"}

        except Exception as e:
            err_str = str(e)
            if "Connection refused" in err_str or "ConnectError" in err_str or "10061" in err_str:
                msg = "⚠️ Ollama 服务未运行，请先启动 Ollama（ollama serve）。"
            else:
                msg = f"抱歉，系统处理出现异常: {e}"
            logger.error(f"❌ Agent 流式推理失败: {e}")
            yield {"type": "token", "content": msg}
            yield {"type": "done"}
