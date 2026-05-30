"""
聊天 API 路由器 — FastAPI 路由层

路由策略（意图驱动）:
  POST /api/chat        — 同步问答（JSON 响应）
  POST /api/chat/stream — SSE 流式问答（逐 token 推送）
  POST /api/chat/daily_news — 新闻查询
  GET  /api/chat/health — 健康检查

SSE 协议格式:
  data: {"intent":"medical_inquiry"}\n\n     ← 意图事件（首个）
  data: "\u5934"\n\n                         ← 逐 token，JSON 字符串编码
  data: "\u75db"\n\n
  data: [DONE]\n\n                           ← 终止信号

设计说明:
  - 不再使用 system_initializer.chatbot（MedicalChatbot 已移除）
  - 路由层直接使用 system_initializer.agent
  - 人工审核 review 端点已移除（SSE 流式路径不再产生 review 事件）
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


class ChatRequest(BaseModel):
    """聊天请求体"""
    query: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    """聊天响应体"""
    intent: str
    answer: str


router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.get("/health")
async def api_health():
    """
    系统健康检查接口

    用于 Docker HEALTHCHECK 和前端监控。
    返回各组件初始化状态（是否 None）。
    """
    si = system_initializer
    return {
        "status": "ok" if si.initialized else "initializing",
        "components": {
            "vector_store": si.vector_store is not None,
            "memory_store": si.memory_store is not None,
            "intent_classifier": si.intent_classifier is not None,
            "tool_manager": si.tool_manager is not None,
            "agent": si.agent is not None,
        },
        "errors": si.init_errors if si.init_errors else [],
    }


@router.post("", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    同步问答接口（非流式）

    请求 -> 意图分类 -> 路由分发 -> 返回 JSON {intent, answer}

    Args:
        request: ChatRequest {query, session_id}

    Returns:
        ChatResponse {intent, answer}
    """
    agent = system_initializer.agent

    if not agent:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    try:
        intent = system_initializer.intent_classifier.classify(user_input)
        logger.info(f"🔍 识别意图: {intent}")

        if intent == "medical_inquiry" or intent == "unknown":
            # 医疗问题/未知 → Agent 同步推理（走 LangGraph 图）
            answer = agent.run(user_input=user_input, session_id=session_id)
        elif intent == "chat_general":
            if is_weather_query(user_input):
                # 天气查询 → ToolManager 天气流水线
                answer = system_initializer.tool_manager.get_weather_response(user_input)
            else:
                # 闲聊 → ToolManager 通用 LLM
                answer = system_initializer.tool_manager.handle_general_query(user_input)
        elif intent == "system_query":
            # 系统查询 → ToolManager 通用 LLM
            answer = system_initializer.tool_manager.handle_general_query(user_input)
        else:
            # 兜底 → Agent 同步推理
            answer = agent.run(user_input=user_input, session_id=session_id)

        return ChatResponse(intent=intent, answer=answer)

    except Exception as e:
        logger.error(f"API处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@router.post("/stream")
async def api_chat_stream(request: ChatRequest):
    """
    SSE 流式问答接口

    使用 Server-Sent Events 协议逐 token 推送 LLM 生成结果。
    同步 Generator 通过 loop.run_in_executor() 转换为异步迭代。

    SSE 事件序列:
      1. data: {"intent":"medical_inquiry"}
      2. data: "token1"
      3. data: "token2"
      ...
      N. data: [DONE]

    Args:
        request: ChatRequest {query, session_id}

    Returns:
        StreamingResponse (media_type="text/event-stream")
    """
    agent = system_initializer.agent

    if not agent:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    async def event_stream():
        """异步 SSE 事件生成器"""
        loop = asyncio.get_event_loop()
        gen = agent.run_stream(user_input=user_input, session_id=session_id)

        async def _next():
            """在线程池中执行 next(gen)，避免阻塞事件循环"""
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

            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"流式处理错误: {e}")
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/daily_news", response_model=NewsResponse)
async def daily_news(request: NewsRequest = None):
    """
    每日新闻查询接口

    通过聚合数据 API 获取各类型新闻。
    类型列表见 APP_CONFIG.valid_news_types。

    Args:
        request: NewsRequest {news_type}，默认 "top"

    Returns:
        NewsResponse {success, news[], total, error}
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
