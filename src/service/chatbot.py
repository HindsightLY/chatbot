"""
聊天机器人模块
提供基于 RAG + LangGraph Agent 的医疗咨询功能

调用链:
  intent_classifier.classify() → 判定意图
    → medical_inquiry   → MedicalAgent (RAG, 带对话历史)
    → chat_general      → MedicalAgent (闲聊/天气, 均带对话历史)

多轮记忆: 所有意图分支在生成回答前均会从 Redis 拉取历史，
  确保用户之前提到的信息（姓名、症状等）可被后续轮次引用。
"""
from src.service.agent import MedicalAgent
from src.utils.logger_config import logger


class MedicalChatbot:
    """
    医疗聊天机器人

    基于 LangGraph Agent 构建:
      Agent 内部管理 classify → retrieve → generate → save 完整流程

    依赖:
      - MedicalAgent 提供 LangGraph 状态机  [agent.py]
      - MemoryStore 提供 Redis 对话记忆    [memory_store.py]
    """

    def __init__(self, agent: MedicalAgent):
        """
        Args:
            agent: 由 SystemInitializer 创建并注入的 MedicalAgent 实例
        """
        self.agent = agent

    def get_answer(self, question: str, session_id: str = "default") -> dict:
        """
        执行单轮 Agent 推理

        Args:
            question: 用户输入
            session_id: 会话 ID

        Returns:
            {"answer": str}
        """
        try:
            answer = self.agent.run(user_input=question, session_id=session_id)
            return {"answer": answer}

        except Exception as e:
            err_str = str(e)
            if "Connection refused" in err_str or "ConnectError" in err_str or "10061" in err_str:
                msg = "⚠️ Ollama 服务未运行，请先启动 Ollama（ollama serve）后重试。"
            else:
                msg = f"抱歉，处理出错: {e}"
            logger.exception(f"获取答案失败")
            return {"answer": msg}

    def ask_stream(self, question: str, session_id: str = "default"):
        """
        流式推理，逐事件 yield（意图 + token + done）

        内部调用 agent.run_stream()，将事件原样透传给调用方。

        Yields:
            dict: {"type": "intent", "content": str} | {"type": "token", "content": str} | {"type": "done"}
        """
        try:
            yield from self.agent.run_stream(user_input=question, session_id=session_id)
        except Exception as e:
            err_str = str(e)
            if "Connection refused" in err_str or "ConnectError" in err_str or "10061" in err_str:
                msg = "⚠️ Ollama 服务未运行，请先启动 Ollama（ollama serve）后重试。"
            else:
                msg = f"抱歉，处理出错: {e}"
            logger.exception(f"流式获取答案失败")
            yield {"type": "token", "content": msg}
            yield {"type": "done"}
