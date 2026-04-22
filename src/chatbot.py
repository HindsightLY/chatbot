"""
聊天机器人模块
提供基于RAG的医疗咨询功能
"""
import re
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from .logger_config import logger


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    """
    内存聊天历史记录类
    用于存储单个会话的聊天历史
    """

    def __init__(self):
        super().__init__()
        self.messages = []

    def add_user_message(self, message: str):
        """添加用户消息"""
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        """添加AI消息"""
        self.messages.append(AIMessage(content=message))

    def clear(self):
        """清空历史记录"""
        self.messages = []


# 全局存储历史
store = {}


class MedicalChatbot:
    """
    医疗聊天机器人
    基于RAG技术提供医疗咨询服务
    """

    def __init__(self, vector_store):
        """
        初始化医疗聊天机器人

        Args:
            vector_store: 向量存储实例
        """
        self.vector_store = vector_store

        # 使用Ollama LLM
        self.llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.1,
            base_url="http://localhost:11434"
        )

        # 定义prompt（支持对话历史）
        self.prompt = PromptTemplate.from_template(
            """你是一位专业医疗顾问。请根据以下医学资料回答用户问题。
            若资料中无直接匹配，请基于医学常识谨慎推断，但需注明"可能"、"常见原因包括"等措辞。

            【对话历史】
            {chat_history}

            【医学资料】
            {context}

            【用户问题】
            {input}

            请直接给出清晰、专业的回答，分点说明可能疾病、症状关联与建议。
            """
        )

        # 创建文档链
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # 创建retriever
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 6, "score_threshold": 0.3}
        )

        # 创建最终检索链
        self.qa_chain = create_retrieval_chain(
            self.retriever,
            self.document_chain
        )

        # 定义获取历史的函数
        def get_session_history(session_id: str):
            if session_id not in store:
                store[session_id] = InMemoryChatMessageHistory()
            return store[session_id]

        # 创建带记忆的链
        self.memory_chain = RunnableWithMessageHistory(
            self.qa_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history"  # 注意这里的键名需要和PromptTemplate中的一致
        )

    def _extract_city_from_input(self, input_text: str, common_cities: list) -> str:
        """
        从输入文本中提取城市名

        Args:
            input_text: 输入文本
            common_cities: 常见城市列表

        Returns:
            str: 提取出的城市名，如果未找到则返回None
        """
        # 方法1：直接匹配城市列表
        for city in common_cities:
            if city in input_text:
                return city

        # 方法2：尝试匹配"在XX"、"去XX"、"XX的"等模式
        patterns = [
            r'在([^\s，。？！,\.!?]+?)[的\s]',  # 匹配"在XX的"
            r'去([^\s，。？！,\.!?]+?)[的\s]',  # 匹配"去XX的"
            r'([^\s，。？！,\.!?]+?)的天气',  # 匹配"XX的天气"
            r'([^\s，。？！,\.!?]+?)天气',  # 匹配"XX天气"
        ]

        for pattern in patterns:
            match = re.search(pattern, input_text)
            if match:
                potential_city = match.group(1)
                # 检查是否是常见城市之一
                for city in common_cities:
                    if city in potential_city or potential_city in city:
                        return city

        return None

    def get_answer_with_tools(self, user_input: str, session_id: str):
        """
        使用工具（如天气、时间）辅助生成回答

        Args:
            user_input: 用户输入
            session_id: 会话ID

        Returns:
            str: 工具辅助生成的回答
        """
        from src.tools.weather_tool import get_current_weather, get_current_time

        # 定义常见城市列表（可扩展）
        common_cities = [
            "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "天津",
            "重庆", "成都", "武汉", "西安", "青岛", "大连", "厦门", "宁波",
            "长沙", "郑州", "济南", "福州", "合肥", "太原", "石家庄", "沈阳",
            "长春", "哈尔滨", "昆明", "南宁", "海口", "兰州", "银川", "西宁",
            "乌鲁木齐", "拉萨", "呼和浩特", "香港", "澳门", "台北"
        ]

        # 检测关键词触发工具
        if any(keyword in user_input for keyword in ["天气", "气温", "下雨", "阴天", "晴天", "多云", "温度"]):
            # 尝试从用户输入中提取城市
            city = self._extract_city_from_input(user_input, common_cities)
            if city:
                weather_resp = get_current_weather(city)
                return f"【工具回复】{weather_resp}"
            else:
                return "抱歉，我没有识别到您询问的城市，请明确告诉我您想查询哪个城市的天气。"

        elif any(keyword in user_input for keyword in ["时间", "现在几点", "几点了", "日期", "今天", "现在"]):
            time_resp = get_current_time()
            return f"【工具回复】{time_resp}"

        # 否则走默认 LLM 回答
        return self.get_answer(user_input, session_id)

    def ask_stream(self, question, session_id="default"):
        """
        流式回答，逐字符输出

        Args:
            question: 问题
            session_id: 会话ID

        Yields:
            str: 逐字符的响应内容
        """
        logger.info("💬 医疗顾问回复：")

        inputs = {"input": question}

        try:
            # 添加历史记录
            if session_id in store:
                history = store[session_id]
                history.add_user_message(question)
            else:
                history = InMemoryChatMessageHistory()
                store[session_id] = history
                history.add_user_message(question)

            # 调用带记忆的链
            for chunk in self.memory_chain.stream(
                    inputs,
                    config={"configurable": {"session_id": session_id}}
            ):
                # 修复：检查chunk中是否有answer键
                if isinstance(chunk, dict):
                    # 尝试不同的键名
                    content = None
                    for key in ["answer", "output", "result", "response"]:
                        if key in chunk:
                            content = chunk[key]
                            break
                elif hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = str(chunk)

                # 【关键改动】：逐字符 Yield
                # 如果 content 是字符串，我们逐个字符输出以实现"打字机"效果
                if content:
                    if isinstance(content, str):
                        for char in content:
                            # 这里可以添加微小的延迟来模拟"打字"效果，也可以不加，由前端控制
                            # time.sleep(0.01) # 可选：如果希望后端控制速度，取消注释
                            yield char  # <-- 将字符发送给调用方
                    elif hasattr(content, '__iter__'):  # 如果是可迭代对象
                        yield str(content)

            logger.info("\n\n" + "=" * 50)
            logger.info("✅ 回答完毕")

        except Exception as e:
            logger.info(f"\n❌ 流式输出失败: {e}")
            # 备用方案：非流式输出
            try:
                result = self.memory_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})
                answer = result.get('answer', result.get('output', result.get('result', '无结果')))
                logger.info(f"\n📝 最终回复: {answer}")

                # 保存AI的回答到历史记录
                if session_id in store:
                    store[session_id].add_ai_message(answer)

            except Exception as fallback_error:
                logger.info(f"备用方案也失败: {fallback_error}")

    def get_answer(self, question, session_id="default"):
        """
        非流式获取答案的方法

        Args:
            question: 问题
            session_id: 会话ID

        Returns:
            dict: 包含答案的字典
        """
        inputs = {"input": question}
        result = self.memory_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})

        # 保存对话到历史
        if session_id in store:
            store[session_id].add_user_message(question)
            ai_response = result.get('answer', result.get('output', result.get('result', '无结果')))
            store[session_id].add_ai_message(ai_response)

        return result