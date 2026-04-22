from src.logger_config import monitor_performance, logger
from src.service.system_initializer import system_initializer


@monitor_performance
def run_cli():
    """
    命令行交互逻辑
    """
    # 获取系统组件
    vector_store = system_initializer.vector_store
    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    # 如果组件未初始化，则初始化
    if vector_store is None or intent_classifier is None or chatbot is None:
        system_initializer.initialize_system()
        # 重新获取初始化后的组件
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

        # 意图识别
        intent = intent_classifier.classify(user_input)
        logger.info(f"\n🔍 识别意图: {intent}")

        try:
            # 根据意图分流处理
            if intent == "medical_inquiry":
                # 医疗意图：走RAG检索流程
                chatbot.ask_stream(user_input, session_id=current_session_id)
            elif intent == "chat_general":
                # 闲聊意图：使用工具管理器处理
                from src.service.tool_manager import tool_manager
                lower_input = user_input.lower()
                if any(keyword in lower_input for keyword in ['天气', 'weather', '气温', '温度']):
                    response = tool_manager.get_weather_response(user_input)
                    print(f"AI: {response}")
                else:
                    response = tool_manager.handle_general_query(user_input)
                    print(f"AI: {response}")
            else:
                # 未知意图或其他，默认走RAG流程
                chatbot.ask_stream(user_input, session_id=current_session_id)

        except Exception as e:
            logger.info(f"\n❌ 错误: {e}")
