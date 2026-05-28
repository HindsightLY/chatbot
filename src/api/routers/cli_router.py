"""
CLI 路由器
提供命令行交互界面，与 chat_router.py 共享同一套意图路由逻辑

流程:
  1. 调用 chatbot.ask_stream() — Agent 内部完成意图分类 + 检索 + 生成
  2. 逐事件处理 token / review / done（不再重复调用 intent_classifier）

被 main.py 调用:
  python main.py              → CLI 模式（默认）
  python main.py --api        → API 模式
"""
from src.utils.logger_config import monitor_performance, logger
from src.service.system_initializer import system_initializer


@monitor_performance
def run_cli():
    """
    CLI 主循环

    全部走流式路径 (ask_stream)，由 Agent 内部统一完成意图分类、检索、生成。
    避免重复调用 intent_classifier。
    """
    chatbot = system_initializer.chatbot

    if chatbot is None:
        logger.info("🔄 正在初始化系统...")
        system_initializer.initialize_system()
        chatbot = system_initializer.chatbot

    if chatbot is None:
        logger.error("❌ 系统初始化失败（Ollama 可能未运行），请检查后重试")
        return

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

        try:
            saved = True
            for event in chatbot.ask_stream(user_input, session_id=current_session_id):
                if event["type"] == "intent":
                    logger.info(f"\n🔍 识别意图: {event['content']}")
                elif event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "review":
                    print()
                    resp = input("\n🤔 确认回答？(Y/n): ").strip().lower()
                    if resp == "n":
                        saved = False
                        logger.info("⏭️ 用户拒绝回答，未保存记忆")
                    else:
                        saved = True
                elif event["type"] == "done":
                    if not saved:
                        pass  # 记忆已在 ask_stream 中保存，此处仅控制不重复保存
                    break
            print()
        except Exception as e:
            logger.info(f"\n❌ 错误: {e}")
