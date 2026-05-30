"""
CLI 路由器 — 命令行交互界面

与 chat_router.py 共享同一套意图路由逻辑（通过 system_initializer.agent）。

流程:
  1. run_cli() 检查 system_initializer.agent 是否为 None
  2. 如果为 None，调用 system_initializer.initialize_system()
  3. 进入 while True 循环，逐行读取用户输入
  4. 每轮调用 agent.run_stream()，逐事件处理 intent / token / done
  5. 'quit' 或 'exit' 退出

注意事项:
  - run_stream 流式路径不再产生 review 事件，记忆在 agent 内部自动保存
  - 所有事件由 agent 内部统一完成意图分类 + 检索 + 生成 + 保存，无需外部干预

被 main.py 调用:
  python main.py         → CLI 模式（默认）
  python main.py --api   → API 模式
"""
from src.utils.logger_config import monitor_performance, logger
from src.service.system_initializer import system_initializer


@monitor_performance
def run_cli():
    """
    CLI 主循环

    全部走流式路径 (agent.run_stream)，由 Agent 内部统一完成意图分类、检索、生成。
    事件处理:
      - intent → 打印意图标签
      - token  → 直接输出（不换行）
      - done   → 换行，继续下一轮
    """
    agent = system_initializer.agent

    if agent is None:
        logger.info("🔄 正在初始化系统...")
        system_initializer.initialize_system()
        agent = system_initializer.agent

    if agent is None:
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
            # run_stream 依次产出 intent → token... → done
            for event in agent.run_stream(user_input=user_input, session_id=current_session_id):
                if event["type"] == "intent":
                    logger.info(f"\n🔍 识别意图: {event['content']}")
                elif event["type"] == "token":
                    print(event["content"], end="", flush=True)
                elif event["type"] == "done":
                    break
            print()
        except Exception as e:
            logger.info(f"\n❌ 错误: {e}")
