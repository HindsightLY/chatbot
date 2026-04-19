import os
import time
import argparse  # ✅ 新增：用于解析命令行参数
import uvicorn  # ✅ 新增：ASGI 服务器
from pydantic import BaseModel  # ✅ 新增：数据校验
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from src.chatbot import CloudProductChatbot
from src.document_loader import DocumentLoader
from src.intent_classifier import IntentClassifier
from src.logger_config import logger, monitor_performance
from src.vector_store import VectorStoreManager

# --- 全局变量：用于在 API 和 CLI 之间共享核心组件 ---
vector_store = None
intent_classifier = None
chatbot = None


def initialize_system():
    """
    ✅ 提取公共初始化逻辑
    无论是 CLI 还是 API，都需要先加载向量和模型
    """
    global vector_store, intent_classifier, chatbot

    # 1. 路径配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    disease_dir = os.path.join(project_root, "data", "disease")

    # 2. 加载文档与向量库
    doc_loader = DocumentLoader(data_dir=disease_dir)
    vector_manager = VectorStoreManager()

    store = vector_manager.load_vector_store()
    if store is None:
        logger.info("🔄 正在创建向量存储（首次运行或索引丢失）...")
        documents = doc_loader.load_documents()
        if not documents:
            raise Exception("❌ 没有加载到任何文档，无法初始化系统")
        vector_store = vector_manager.create_vector_store(documents)
        logger.info("✅ 向量存储创建完成！")
    else:
        vector_store = store
        logger.info("✅ 成功加载现有向量库！")

    # 3. 初始化意图识别与聊天机器人
    intent_classifier = IntentClassifier()
    chatbot = CloudProductChatbot(vector_store)


# ✅ 新增：API 请求的数据模型
class ChatRequest(BaseModel):
    query: str
    session_id: str = "default_user"  # 前端可以传入 session_id 来维持多轮对话


class ChatResponse(BaseModel):
    intent: str
    answer: str


# ✅ 新增：FastAPI 实例
app = FastAPI(title="医疗咨询AI API", description="基于RAG的医疗问答接口")


@app.on_event("startup")
def startup_event():
    """
    FastAPI 启动时自动运行
    """
    logger.info("🚀 正在初始化医疗AI系统...")
    initialize_system()
    logger.info("🎉 系统初始化完成，API 就绪！")

def generate_response_stream(query, history):
    # 示例：模拟逐个字符生成响应
    response = "这是一个模拟的流式响应。"
    for char in response:
        yield char
        time.sleep(0.05)  # 模拟延迟

@app.post("/api/chat", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    ✅ 核心 API 接口
    """
    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    # 1. 意图识别
    intent = intent_classifier.classify(user_input)

    # 2. 根据意图分流 (复用 CLI 的逻辑)
    try:
        if intent == "medical_inquiry":
            # 医疗意图：走 RAG
            # 注意：ask_stream 是打印流，这里我们需要获取完整文本
            # 为了最小化修改，我们直接调用 chatbot 的底层逻辑或模拟获取
            # 由于原 chatbot.ask_stream 直接 print，我们需要一个能返回字符串的方法
            # 这里简单处理：直接调用 ask_stream 但捕获输出不太优雅，
            # 建议 chatbot 增加一个 get_response 方法，但为了“最小化修改”，
            # 我们假设 ask_stream 内部逻辑可以复用，或者我们在这里简单包装。

            # 方案：为了不改 chatbot.py 太多，我们在这里简单模拟“获取回答”
            # 实际上，最好是在 chatbot.py 中把生成逻辑提取出来。
            # 但根据“最小化修改”原则，我们直接调用 ask_stream 并让它打印到控制台，
            # 同时为了返回给前端，我们需要 chatbot 能返回字符串。

            # ⚠️ 关键修改点：我们需要 chatbot 返回字符串而不是只打印
            # 查看 chatbot.py，发现有一个 get_answer 方法！完美利用它。
            result = chatbot.get_answer(user_input, session_id=session_id)
            answer = result.get('answer', '抱歉，我没有找到相关信息。')

        elif intent == "chat_general":
            # 闲聊意图
            result = chatbot.get_answer(user_input, session_id=session_id)
            answer = result.get('answer', '你好！')
        else:
            # 未知意图
            result = chatbot.get_answer(user_input, session_id=session_id)
            answer = result.get('answer', '我不太明白你的意思。')

        return ChatResponse(intent=intent, answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def api_chat_stream(request: ChatRequest):
    """
    新增流式接口
    """
    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    # 1. 意图识别 (这里我们先做一次同步识别，决定流的内容)
    # 注意：为了流式传输，我们不能在这里阻塞太久，所以先简单处理
    # 实际上，意图识别很快，可以接受
    intent = intent_classifier.classify(user_input)

    # 我们将意图放在 SSE 的第一个消息里，或者作为元数据
    # 这里为了简单，我们直接在流开始时发送一个包含意图的JSON，然后是文本流
    # 但为了最简改动，我们只流式传输文本，前端通过其他方式获取意图
    # 或者：我们构建一个生成器，先生成意图，再生成文本

    async def generate_response():
        # 发送意图 (可选：作为第一个数据块)
        # yield f"data: {json.dumps({'type': 'intent', 'data': intent})}\n\n"

        # 发送文本流
        try:
            # 注意：ask_stream 现在是一个生成器
            for char in chatbot.ask_stream(user_input, session_id=session_id):
                if char:
                    # 【修改点】：直接 yield 字符，不包装成 JSON
                    yield char
                    # 如果你还是想保留 SSE 格式，但希望 Postman 好看一点，
                    # 可以确保 ensure_ascii=False，但这在 SSE 里依然会分行
                    # yield f"data: {json.dumps({'data': char}, ensure_ascii=False)}\n\n"

            # 发送结束标记
            # yield f"data: {json.dumps({'type': 'done', 'data': 'stream_end'})}\n\n"
            # 【修改点】：最后给一个结束标记（可选）
            yield "[DONE]"

        except Exception as e:
            # yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/event-stream")

# --- 原有的 CLI 逻辑 ---
@monitor_performance # ✅ 添加这一行，监控整个 CLI 循环的性能
def run_cli():
    """
    原有的命令行交互逻辑
    """
    # 注意：initialize_system 已经初始化了全局变量，
    # 但为了保持原有逻辑独立（如果用户直接运行旧代码），这里可以保留部分逻辑
    # 但既然我们要集成，直接依赖 initialize_system 更好。

    # 如果是通过 python main.py 运行，且没有指定 --api，则走这里
    if vector_store is None:
        initialize_system()

    global intent_classifier, chatbot

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
                # ✅医疗意图：走RAG检索流程
                chatbot.ask_stream(user_input, session_id=current_session_id)
            elif intent == "chat_general":
                # ✅闲聊意图：不检索，直接让模型基于通用知识回答
                logger.info("AI (通用模式): ")
                chatbot.ask_stream(user_input, session_id=current_session_id)
            else:
                # ✅未知意图或其他，默认走RAG流程
                chatbot.ask_stream(user_input, session_id=current_session_id)

        except Exception as e:
            logger.info(f"\n❌ 错误: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="医疗咨询AI启动器")
    parser.add_argument("--api", action="store_true", help="以 API 模式启动 (FastAPI)")
    parser.add_argument("--host", default="0.0.0.0", help="API 监听地址")
    parser.add_argument("--port", type=int, default=8000, help="API 监听端口")

    args = parser.parse_args()

    if args.api:
        # ✅ 启动 API 模式
        # 注意：initialize_system 会在 on_event("startup") 中调用
        # 这里直接启动 uvicorn
        uvicorn.run("src.main:app", host=args.host, port=args.port, reload=False)
    else:
        # ✅ 启动 CLI 模式 (默认)
        run_cli()
