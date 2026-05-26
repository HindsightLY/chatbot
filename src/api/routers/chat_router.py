"""
聊天 API 路由器
FastAPI 路由层，负责请求分发

路由策略（意图驱动）:
  POST /api/chat        — 普通问答
  POST /api/chat/stream — SSE 流式问答
  POST /api/chat/daily_news — 新闻查询

核心分发逻辑:
  intent_classifier.classify()
    ├─ medical_inquiry → MedicalChatbot.get_answer()
    ├─ chat_general    → ToolManager (天气/闲聊)
    ├─ system_query    → ToolManager.handle_general_query()
    └─ unknown         → 兜底 RAG
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config.app_config import APP_CONFIG
from src.tools.news_tool import get_daily_news, NewsResponse, NewsRequest
from src.utils.logger_config import logger
from src.utils.text_utils import is_weather_query
from src.service.system_initializer import system_initializer
from src.service.tool_manager import tool_manager


class ChatRequest(BaseModel):
    """聊天请求"""
    query: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    """聊天响应"""
    intent: str
    answer: str


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    普通问答接口

    流程: 意图分类 → 路由 → 调用对应处理器 → 返回结果
    """
    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    try:
        intent = intent_classifier.classify(user_input)
        logger.info(f"🔍 识别意图: {intent}")

        if intent == "medical_inquiry":
            result = chatbot.get_answer(user_input, session_id)
            answer = result.get('answer', '抱歉，我没有找到相关信息。')

        elif intent == "chat_general":
            if is_weather_query(user_input):
                answer = tool_manager.get_weather_response(user_input)
            else:
                answer = tool_manager.handle_general_query(user_input)

        elif intent == "system_query":
            answer = tool_manager.handle_general_query(user_input)

        else:
            result = chatbot.get_answer(user_input, session_id)
            answer = result.get('answer', '抱歉，我没有找到相关信息。')

        return ChatResponse(intent=intent, answer=answer)

    except Exception as e:
        logger.error(f"API处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@router.post("/stream")
async def api_chat_stream(request: ChatRequest):
    """
    SSE 流式问答接口

    注意: 当前所有处理器均为同步阻塞调用，返回后才逐字符 yield，
    并非真正的流式生成。未来可替换为 llm.stream() 实现逐 token 输出。
    """
    from fastapi.responses import StreamingResponse

    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    intent = intent_classifier.classify(user_input)

    async def generate_response():
        try:
            if intent == "medical_inquiry":
                for char in chatbot.ask_stream(user_input, session_id=session_id):
                    if char:
                        yield char
            elif intent == "chat_general":
                if is_weather_query(user_input):
                    response = tool_manager.get_weather_response(user_input)
                else:
                    response = tool_manager.handle_general_query(user_input)
                for char in response:
                    yield char
            else:
                for char in chatbot.ask_stream(user_input, session_id=session_id):
                    if char:
                        yield char

            yield "[DONE]"

        except Exception as e:
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@router.post("/daily_news", response_model=NewsResponse)
async def daily_news(request: NewsRequest = None):
    """
    每日新闻查询

    通过聚合数据 API 获取各类型新闻。
    类型列表见 APP_CONFIG.valid_news_types。
    """
    if request is None:
        request = NewsRequest()

    news_type = request.news_type

    if news_type not in APP_CONFIG.valid_news_types:
        news_type = "top"

    try:
        news_result = get_daily_news(news_type)

        if news_result["success"]:
            return NewsResponse(
                success=True,
                news=news_result["news"],
                total=news_result["total"]
            )
        else:
            return NewsResponse(
                success=False,
                error=news_result.get("reason", "获取新闻失败")
            )

    except Exception as e:
        logger.error(f"获取新闻时发生错误: {e}")
        return NewsResponse(
            success=False,
            error=f"获取新闻时发生错误: {str(e)}"
        )
