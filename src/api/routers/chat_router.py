"""
聊天API路由器
处理所有聊天相关的API请求
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from config.app_config import APP_CONFIG
from src.tools.news_tool import get_daily_news, NewsResponse, NewsRequest
from src.logger_config import logger
from src.service.system_initializer import system_initializer
from src.service.tool_manager import tool_manager


# API请求/响应模型
class ChatRequest(BaseModel):
    """聊天请求数据模型"""
    query: str
    session_id: str = "default_user"


class ChatResponse(BaseModel):
    """聊天响应数据模型"""
    intent: str
    answer: str


# 创建路由器实例
router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def api_chat(request: ChatRequest):
    """
    聊天API接口

    Args:
        request: 聊天请求

    Returns:
        ChatResponse: 聊天响应
    """
    # 获取系统组件
    vector_store = system_initializer.vector_store
    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    try:
        # 1. 意图识别
        intent = intent_classifier.classify(user_input)
        logger.info(f"🔍 识别意图: {intent}")

        # 按意图处理
        if intent == "medical_inquiry":
            # 医疗意图：走 RAG
            result = chatbot.get_answer(user_input, session_id)
            answer = result.get('answer', '抱歉，我没有找到相关信息。')

        elif intent == "chat_general":
            # 处理天气查询
            lower_input = user_input.lower()
            is_weather_query = any(keyword in lower_input for keyword in ['天气', 'weather', '气温', '温度'])
            if is_weather_query:
                answer = tool_manager.get_weather_response(user_input)
            else:
                # 闲聊意图：使用通用 LLM
                answer = tool_manager.handle_general_query(user_input)

        elif intent == "system_query":
            # 系统查询，走通用 LLM
            answer = tool_manager.handle_general_query(user_input)

        else:
            # 未知意图，默认走RAG
            result = chatbot.get_answer(user_input, session_id)
            answer = result.get('answer', '抱歉，我没有找到相关信息。')

        return ChatResponse(intent=intent, answer=answer)

    except Exception as e:
        logger.error(f"API处理错误: {e}")
        raise HTTPException(status_code=500, detail=f"处理请求时发生错误: {str(e)}")


@router.post("/stream")
async def api_chat_stream(request: ChatRequest):
    """
    流式聊天API接口

    Args:
        request: 聊天请求

    Returns:
        StreamingResponse: 流式响应
    """
    from fastapi.responses import StreamingResponse

    # 获取系统组件
    vector_store = system_initializer.vector_store
    intent_classifier = system_initializer.intent_classifier
    chatbot = system_initializer.chatbot

    if not chatbot:
        raise HTTPException(status_code=500, detail="系统未初始化")

    user_input = request.query
    session_id = request.session_id

    # 意图识别
    intent = intent_classifier.classify(user_input)

    async def generate_response():
        try:
            # 根据意图决定使用哪种响应方式
            if intent == "medical_inquiry":
                # 医疗意图：使用RAG流式响应
                for char in chatbot.ask_stream(user_input, session_id=session_id):
                    if char:
                        yield char
            elif intent == "chat_general":
                # 通用意图：根据具体情况决定
                lower_input = user_input.lower()
                is_weather_query = any(keyword in lower_input for keyword in ['天气', 'weather', '气温', '温度'])

                if is_weather_query:
                    # 天气查询：生成一次性响应然后流式输出
                    response = tool_manager.get_weather_response(user_input)
                    for char in response:
                        yield char
                else:
                    # 闲聊：生成一次性响应然后流式输出
                    response = tool_manager.handle_general_query(user_input)
                    for char in response:
                        yield char
            else:
                # 其他意图：使用RAG流式响应
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
    每日新闻API接口

    Args:
        request: 新闻请求，如果为None则使用默认值

    Returns:
        NewsResponse: 新闻响应
    """
    if request is None:
        request = NewsRequest()

    news_type = request.news_type

    # 验证新闻类型
    if news_type not in APP_CONFIG.valid_news_types:
        news_type = "top"  # 默认为top

    try:
        # 获取新闻数据
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