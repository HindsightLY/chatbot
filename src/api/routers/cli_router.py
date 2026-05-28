"""
CLI 路由器
提供命令行交互界面，与 chat_router.py 共享同一套意图路由逻辑

被 main.py 调用:
  python main.py              → CLI 模式（默认）
  python main.py --api        → API 模式
"""
from src.utils.logger_config import monitor_performance, logger
from src.utils.text_utils import is_weather_query
from src.service.system_initializer import system_initializer


@monitor_performance
def run_cli():
    """
    CLI 主循环

    流程与 API 完全一致:
      intent_classifier.classify()
        ├─ medical_inquiry → chatbot.get_answer()
        ├─ chat_general    → tool_manager (天气/闲聊)
        └─ else            → 兜底 RAG
    """
    vector_store = system_initializer.vector_store
    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    if vector_store is None or intent_classifier is None or chatbot is None:
        system_initializer.initialize_system()
        vector_store = system_initializer.vector_store
        intent_classifier = system_initializer.intent_classifier
        chatbot = system_initializer.chatbot

    logger.info("\n" + "=" * 60)
    logger.info("🤖 医疗疾病咨询AI已启动 (CLI模式)")
    logger.info("💡 输入 'quit' 或 'exit' 退出")
    logger.info("=" * 60)

    current_session_id = "user_session_123"
    while True:
        logger.info("\n📝 您: ")
        user_input = input().strip()
        if user_input.lower() in ['quit', 'exit']:
            logger.info("👋 再见！")
            break
        if not user_input:
            continue

        intent = intent_classifier.classify(user_input)
        logger.info(f"\n🔍 识别意图: {intent}")

        try:
            if intent == "medical_inquiry" or intent == "unknown":
                for event in chatbot.ask_stream(user_input, session_id=current_session_id):
                    if event["type"] == "token":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "done":
                        break
                print()
            elif intent == "chat_general":
                from src.service.tool_manager import tool_manager
                if is_weather_query(user_input):
                    response = tool_manager.get_weather_response(user_input)
                else:
                    response = tool_manager.handle_general_query(user_input)
                logger.info(f"AI: {response}")
            else:
                for event in chatbot.ask_stream(user_input, session_id=current_session_id):
                    if event["type"] == "token":
                        print(event["content"], end="", flush=True)
                    elif event["type"] == "done":
                        break
                print()

        except Exception as e:
            logger.info(f"\n❌ 错误: {e}")
