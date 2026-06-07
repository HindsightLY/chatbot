"""
聊天机器人模块 — MedicalAgent 的薄包装层

提供统一的外部调用接口，将 agent.run() 和 agent.run_stream() 包装为 get_answer() 和 ask_stream()。
除了包装功能外，还提供了 get_chatbot_agent() 工厂函数，可以直接获取 MedicalAgent 实例。

调用链:
  chat_router / cli_router → MedicalChatbot (或直接使用 system_initializer.agent)
    → MedicalAgent.run() / run_stream()
      → IntentClassifier → VectorStore → LLM → MemoryStore
"""
from src.service.agent import MedicalAgent
from src.utils.logger_config import logger


class MedicalChatbot:
    """
    医疗聊天机器人包装类

    将 MedicalAgent 的 run() 和 run_stream() 包装为更易用的接口。
    主要用于向后兼容；新代码建议直接使用 system_initializer.agent。
    """

    def __init__(self, agent: MedicalAgent):
        """
        Args:
            agent: 由 SystemInitializer 创建并注入的 MedicalAgent 实例
        """
        self.agent = agent

    def get_answer(self, question: str, session_id: str = "default") -> dict:
        """
        执行单轮 Agent 推理（同步阻塞）

        内部调用 self.agent.run()，将返回的字符串包装为 {"answer": str} 格式。

        Args:
            question: 用户输入问题
            session_id: 会话 ID

        Returns:
            {"answer": str} — 回答文本
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
        流式推理，逐事件 yield

        内部调用 self.agent.run_stream()，将生成的事件原样透传。

        Yields:
            {"type": "intent", "content": str} — 意图事件
            {"type": "token",  "content": str} — LLM 输出片段
            {"type": "done"}                   — 结束信号
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


def get_chatbot_agent(agent: MedicalAgent) -> MedicalAgent:
    """
    获取聊天机器人 Agent 实例的工厂函数（已废弃）

    Args:
        agent: MedicalAgent 实例

    Returns:
        同一个 MedicalAgent 实例

    注意:
      此函数实质为恒等映射，保留仅用于向后兼容。
      新代码应直接使用 system_initializer.agent。
    """
    return agent
