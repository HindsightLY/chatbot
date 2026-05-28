"""
聊天 API 路由器
FastAPI 路由层，负责请求分发

路由策略（意图驱动）:
  POST /api/chat        — 普通问答（JSON 响应）
  POST /api/chat/stream — SSE 流式问答（逐 token 推送）
  POST /api/chat/daily_news — 新闻查询

SSE 协议格式:
  data: {"intent":"medical_inquiry"}\n\n     ← 首个事件：意图
  data: "\u4f60"\n\n                         ← 逐 token，JSON 编码
  data: "\u597d"\n\n
  data: [DONE]\n\n                           ← 终止信号
"""
import asyncio
import json
from functools import partial
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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
    """非流式问答（单次 JSON 响应）"""
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
    SSE 流式问答（真·流式，逐 token 推送）

    SSE 事件序列:
      1. data: {"intent":"<类型>"}
      2. data: "<token>"   (逐 token)
      ...
      N. data: [DONE]
    """
    chatbot = system_initializer.chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    async def event_stream():
        loop = asyncio.get_event_loop()
        gen = chatbot.ask_stream(user_input, session_id)

        async def _next():
            return await loop.run_in_executor(None, partial(next, gen))

        try:
            while True:
                try:
                    event = await _next()
                except StopIteration:
                    break

                if event["type"] == "intent":
                    yield f"data: {json.dumps({'intent': event['content']}, ensure_ascii=False)}\n\n"
                elif event["type"] == "token":
                    yield f"data: {json.dumps(event['content'], ensure_ascii=False)}\n\n"
                elif event["type"] == "done":
                    break
                elif event["type"] == "error":
                    logger.error(f"流式处理错误: {event['content']}")
                    yield f"data: {json.dumps({'error': event['content']}, ensure_ascii=False)}\n\n"
                    break

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式处理错误: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
