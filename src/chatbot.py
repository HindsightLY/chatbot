from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import time


# 使用LangChain内置的消息历史管理
class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        super().__init__()
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def clear(self):
        self.messages = []


# 全局存储历史
store = {}


class CloudProductChatbot:
    def __init__(self, vector_store):
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

    def ask_stream(self, question, session_id="default"):
        print("💬 医疗顾问回复：", end="", flush=True)

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
                # 如果 content 是字符串，我们逐个字符输出以实现“打字机”效果
                if content:
                    if isinstance(content, str):
                        for char in content:
                            # 这里可以添加微小的延迟来模拟“打字”效果，也可以不加，由前端控制
                            # time.sleep(0.01) # 可选：如果希望后端控制速度，取消注释
                            yield char  # <-- 将字符发送给调用方
                    elif hasattr(content, '__iter__'):  # 如果是可迭代对象
                        yield str(content)

            print("\n\n" + "=" * 50)
            print("✅ 回答完毕")

        except Exception as e:
            print(f"\n❌ 流式输出失败: {e}")
            # 备用方案：非流式输出
            try:
                result = self.memory_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})
                answer = result.get('answer', result.get('output', result.get('result', '无结果')))
                print(f"\n📝 最终回复: {answer}")

                # 保存AI的回答到历史记录
                if session_id in store:
                    store[session_id].add_ai_message(answer)

            except Exception as fallback_error:
                print(f"备用方案也失败: {fallback_error}")

    def get_answer(self, question, session_id="default"):
        """非流式获取答案的方法"""
        inputs = {"input": question}
        result = self.memory_chain.invoke(inputs, config={"configurable": {"session_id": session_id}})

        # 保存对话到历史
        if session_id in store:
            store[session_id].add_user_message(question)
            ai_response = result.get('answer', result.get('output', result.get('result', '无结果')))
            store[session_id].add_ai_message(ai_response)

        return result