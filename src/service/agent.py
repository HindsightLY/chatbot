"""
LangGraph 智能体模块 — 多节点状态图 Agent

节点流转（同步图路径 run）:
  classify_intent → call_model (bind_tools) → _route_after_model
    ├─ 有 tool_calls → tool_node → call_model（循环，LLM 用工具结果生成最终回答）
    └─ 无 tool_calls → human_review（interrupt_after 暂停）→ save_memory → END

流式路径 run_stream:
  不经过 LangGraph 图，直接走条件分支 + ChatOllama.stream()
  意图分类 → 按意图路由（medical_inquiry 走 HyDE+检索，chat_general 走天气/闲聊）
  → 记忆持久化 → done

关键设计:
  - 工具使用 LangChain @tool 装饰器定义（medical_tools.py）
  - LLM 通过 bind_tools() 自主决定调用哪个工具
  - human_review 节点通过 interrupt_after 实现人机协同（同步路径）
  - 流式路径自动保存记忆，不支持人工审核中断
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
    """LangGraph 状态定义，所有图节点共享的数据结构"""
    messages: List[Dict[str, str]]   # 消息列表（HumanMessage / AIMessage）
    session_id: str                  # 会话隔离标识
    intent: str                      # 意图分类结果
    context_docs: List[str]          # ChromaDB 检索到的文档正文
    answer: str                      # LLM 最终回答文本
    human_approved: bool             # 人工审核是否通过
    review_skipped: bool             # 是否跳过审核
    tool_used: bool                  # 本轮是否调用了工具


class MedicalAgent:
    """
    医疗咨询智能体

    管理 LangGraph 状态图的生命周期:
      - 构建图结构（5 个节点 + 条件边 + interrupt_after）
      - 注入外部依赖（LLM / 向量库 / 记忆存储 / 意图分类器 / 工具）
      - 提供两种推理入口:
          run()        — 同步阻塞，返回完整回答，支持工具调用 + 人工审核中断
          run_stream() — 流式逐 token 产出，不支持工具调用 + 人工审核
    """

    def __init__(self, vector_store, memory_store, intent_classifier, tool_manager):
        """
        Args:
            vector_store: VectorStoreManager 实例（管理 ChromaDB + BM25）
            memory_store: MemoryStore 实例（Redis 对话记忆）
            intent_classifier: IntentClassifier 实例（BERT/LLM/关键词三引擎）
            tool_manager: ToolManager 实例（天气/闲聊处理器）
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

        # 医疗 RAG 模板：用于 medical_inquiry 意图，包含对话历史 + 检索文档 + 用户问题
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

        # 通用对话模板：用于 chat_general / system_query / 天气润色
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

    # ══════════════════════════════════════════
    #  图节点函数 — 每个函数对应 StateGraph 的一个节点
    # ══════════════════════════════════════════

    def _classify_intent(self, state: AgentState) -> AgentState:
        """
        节点: 意图分类

        从 state.messages 的末尾提取用户输入，调用 IntentClassifier 分类。
        分类结果存入 state.intent，仅用于日志和可能的 UI 展示。
        """
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

        1. 先从 messages 末尾提取用户输入（last_content）
        2. 从 Redis 拉取对话历史（enhanced_history 或裸历史）
        3. 组装 general_prompt → 调用 llm_with_tools.invoke()
        4. LLM 自主决定是否调用工具（tool_calls），或直接生成回答
        5. 将 LLM 返回追加到 messages 列表，answer 字段存回答文本
        """
        messages = state.get("messages", [])
        # 注意: last_content 必须先提取，用于后续的 memory_summarizer 和 prompt 组装
        last_content = ""
        if messages and isinstance(messages[-1], dict):
            last_content = messages[-1].get("content", "")
        elif messages and hasattr(messages[-1], "content"):
            last_content = messages[-1].content

        if self.memory_summarizer:
            history_text = self.memory_summarizer.get_enhanced_history(
                state["session_id"], last_content
            )
        else:
            history_text = self.memory_store.get_history_text(state["session_id"])

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

    def _route_after_model(self, state: AgentState) -> str:
        """
        条件边路由函数: 根据 LLM 输出判断下一步

        Returns:
            "continue" — LLM 生成了 tool_calls，需要进入 tool_node 执行工具
            "end"      — LLM 直接生成回答，无需调用工具，进入 human_review
        """
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
        """
        节点: 执行工具调用

        调用 ToolNode.invoke(state) 执行 LLM 请求的工具。
        工具执行结果（ToolMessage）会追加到 messages 列表。
        设置 tool_used=True 标记本轮回合经过工具调用。
        """
        try:
            result = self.tool_node.invoke(state)
            if isinstance(result, dict) and "messages" in result:
                messages = list(state.get("messages", []))
                messages.extend(result["messages"])
                return {**state, "messages": messages, "tool_used": True}
            elif isinstance(result, list):
                messages = list(state.get("messages", []))
                messages.extend(result)
                return {**state, "messages": messages, "tool_used": True}
        except Exception as e:
            logger.exception(f"❌ 工具执行失败")
        return state

    def _human_review(self, state: AgentState) -> AgentState:
        """
        节点: 人工审核（仅同步图路径）

        通过 compile 时的 interrupt_after=["human_review"] 暂停图执行。
        外部代码可以调用 graph.invoke(None, config) 恢复（自动批准），
        或者调用 graph.update_state() 修改状态后恢复。
        """
        logger.info(f"等待人工审核，session_id={state['session_id']}")
        return state

    def _save_memory(self, state: AgentState) -> AgentState:
        """
        节点: 记忆持久化到 Redis

        从 messages 列表中提取最后一条用户消息和 AI 回答，
        分别调用 memory_store.add_message() 写入 Redis List。
        如果启用了分层记忆，触发 check_and_summarize() 检查是否需要摘要。
        """
        messages = state["messages"]
        last_user_msg = ""
        # 从后往前遍历 messages，找到最后一条用户消息
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

    # ══════════════════════════════════════════
    #  图构建
    # ══════════════════════════════════════════

    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图

        节点:
          classify_intent → call_model → _route_after_model
            ├─ "continue" → tool_node → call_model（循环）
            └─ "end"      → human_review（interrupt_after）→ save_memory → END

        编译参数:
          - checkpointer=MemorySaver(): 内存检查点，保存执行状态
          - interrupt_after=["human_review"]: 在 human_review 节点后暂停
        """
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
            self._route_after_model,
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

    # ══════════════════════════════════════════
    #  推理入口
    # ══════════════════════════════════════════

    def run(self, user_input: str, session_id: str = "default") -> str:
        """
        同步推理入口 — 执行单轮 Agent 推理，返回完整回答文本

        流程:
          1. 构建初始状态 initial_state（含 user_input）
          2. graph.invoke() 执行图 → 执行到 human_review 节点时中断
          3. 如果被中断，自动批准（调用 graph.invoke(None) 恢复）
          4. 返回 final_state["answer"]

        Args:
            user_input: 用户输入文本
            session_id: 会话 ID，用于隔离不同用户的记忆

        Returns:
            回答文本，失败时返回错误提示
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "session_id": session_id,
            "intent": "",
            "context_docs": [],
            "answer": "",
            "human_approved": False,
            "review_skipped": False,
            "tool_used": False,
        }

        thread_config = {"configurable": {"thread_id": session_id}}

        try:
            # 首次调用: 执行到 human_review 节点中断（若 LLM 未触发工具调用）
            final_state = self.graph.invoke(initial_state, config=thread_config)

            # 如果在 human_review 被中断，自动批准恢复执行
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
        流式推理入口 — 逐 token 产出，无需等待 LLM 完全生成

        与 run() 不同，此方法不走 LangGraph 图，直接走条件分支。
        所有分支在生成 prompt 前均从 Redis 拉取对话历史。

        意图路由:
          medical_inquiry / unknown:
            → HyDE 查询转换 → ChromaDB 混合检索 → self.prompt + 历史 + 文档 → LLM.stream()
          chat_general + 天气:
            → tool_manager.get_weather_response() → self.general_prompt → LLM.stream()
          chat_general / system_query:
            → self.general_prompt + 历史 → LLM.stream()
          其他（兜底）:
            → self.prompt + 历史 + "未找到相关医学资料" → LLM.stream()

        Yields:
            {"type": "intent", "content": str}  — 意图事件（第一个发出）
            {"type": "token",  "content": str}  — LLM 输出片段
            {"type": "done"}                    — 结束信号
        """
        try:
            # === 步骤 1: 意图分类 ===
            intent = self.intent_classifier.classify(user_input)
            logger.info(f"Agent 识别意图: {intent}")
            yield {"type": "intent", "content": intent}

            full_answer = ""

            # === 步骤 2: 按意图路由 ===
            if intent == "medical_inquiry" or intent == "unknown":
                # ── 2a. HyDE 查询转换（可选项） ──
                search_query = user_input
                if self.hyde:
                    hyde_query = self.hyde.transform(user_input)
                    search_query = hyde_query

                # ── 2b. 混合检索文档 ──
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

                # ── 2c. 流式 LLM 生成 ──
                for chunk in self.llm.stream([HumanMessage(content=formatted_prompt)]):
                    token = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if token:
                        full_answer += token
                        yield {"type": "token", "content": token}

            elif intent == "chat_general" and is_weather_query(user_input):
                # 天气查询分支: 调用高德 API 获取天气数据 → LLM 润色输出
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
                # 通用闲聊/系统查询分支: 直接走 general_prompt
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
                # 兜底分支: 走 medical prompt 但 context 标记为未找到
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

            # === 步骤 3: 记忆持久化（流式模式下自动保存，无需人工审核） ===
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
