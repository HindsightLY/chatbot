from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.retrievers import BM25Retriever  # 可选：如需纯关键词检索
import time


class CloudProductChatbot:
    def __init__(self, vector_store):
        self.vector_store = vector_store

        # ✅ 使用 Ollama LLM（您已有 qwen2.5:7b）
        self.llm = OllamaLLM(
            model="qwen2.5:7b",
            temperature=0.1,
            base_url="http://localhost:11434"
        )

        # ✅ 1. 初始化内存存储（用于保存对话历史）
        self.store = {}

        # ✅ 定义 prompt（支持中文医疗场景）
        self.prompt = PromptTemplate.from_template(
            """你是一位专业医疗顾问。请根据以下医学资料回答用户问题。
            若资料中无直接匹配，请基于医学常识谨慎推断，但需注明“可能”、“常见原因包括”等措辞。

            【医学资料】
            {context}
            
            【用户问题】
            {input}

            请直接给出清晰、专业的回答，分点说明可能疾病、症状关联与建议。
            """
        )

        # ✅ 创建文档链（将检索到的文档合并为 context）
        self.document_chain = create_stuff_documents_chain(self.llm, self.prompt)

        # ✅ 创建 retriever（从 vector_store）
        self.retriever = vector_store.as_retriever(
            search_kwargs={"k": 6, "score_threshold": 0.3}
        )

        # ✅ 创建最终检索链
        self.qa_chain = create_retrieval_chain(
            self.retriever,
            self.document_chain
        )

    def ask_stream(self, question):
        print("💬 医疗顾问回复：", end="", flush=True)

        # ✅ 新增：定义一个缓冲区，用来存放接收到的文本片段
        buffer = ""

        try:
            # ✅ 使用 .stream() 获取流式数据
            for chunk in self.qa_chain.stream({"input": question}):
                if "answer" in chunk:
                    token = chunk["answer"]

                    # ✅ 核心逻辑：将新收到的文本追加到缓冲区
                    buffer += token

                    # ✅ 将缓冲区内容拆解为单字列表
                    # list("你好") -> ['你', '好']
                    chars = list(buffer)

                    # ✅ 逐字打印
                    for char in chars:
                        print(char, end="", flush=True)

                        # ✅ 模拟打字机延迟（增加真实感）
                        if char in '，。！？；：,.!?;:':
                            time.sleep(0.08)  # 标点符号停顿久一点
                        elif char == ' ':
                            time.sleep(0.05)
                        else:
                            time.sleep(0.01)  # 普通字停顿短一点

                    # ✅ 打印完后清空缓冲区
                    buffer = ""

            print("\n\n" + "=" * 50)
            print("✅ 回答完毕")

        except Exception as e:
            print(f"\n❌ 执行异常: {e}")
            # fallback
            result = self.qa_chain.invoke({"input": question})
            ans = result.get("answer", "")
            print(ans)
