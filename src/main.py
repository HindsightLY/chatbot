import os
import time
from src.document_loader import DocumentLoader
from src.intent_classifier import IntentClassifier
from src.vector_store import VectorStoreManager
from src.chatbot import CloudProductChatbot


def main():
    # ✅ 优化：基于当前文件(__file__)的绝对路径定位
    # __file__ 指向当前文件 (main.py)
    # dirname(__file__) 指向 src 目录
    # dirname(dirname(__file__)) 指向项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    disease_dir = os.path.join(project_root, "data", "disease")

    print(f"📁 项目根目录: {project_root}")
    print(f"📁 医疗文档目录: {disease_dir}")

    # 检查文档目录
    if not os.path.exists(disease_dir):
        print(f"❌ 错误：医疗文档目录不存在！请确认：\n {disease_dir}")
        print("💡 请确保：")
        print(" 1. data/disease/ 目录存在")
        print(" 2. 里面至少有一个 .txt 文件（如 扁桃体炎.txt、感冒.txt）")
        return

    doc_loader = DocumentLoader(data_dir=disease_dir)
    vector_manager = VectorStoreManager()

    # ✅ 核心逻辑：先尝试加载
    vector_store = vector_manager.load_vector_store()

    if vector_store is None:
        print("🔄 正在创建向量存储（首次运行或索引丢失）...")
        documents = doc_loader.load_documents()
        if not documents:
            print("❌ 没有加载到任何文档，程序退出")
            return
        vector_store = vector_manager.create_vector_store(documents)
        print("✅ 向量存储创建完成！")
    else:
        print("✅ 成功加载现有向量库！")

    # ✅ 初始化意图分类器
    intent_classifier = IntentClassifier()

    chatbot = CloudProductChatbot(vector_store)

    print("\n" + "=" * 60)
    print("🤖 医疗疾病咨询AI已启动 (已加载意图识别)")
    print("💡 输入 'quit' 或 'exit' 退出")
    print("=" * 60)

    while True:
        print("\n📝 您: ", end="")
        user_input = input().strip()
        if user_input.lower() in ['quit', 'exit']:
            break
        if not user_input:
            continue

        # ✅ 1. 意图识别
        intent = intent_classifier.classify(user_input)
        print(f"\n🔍 识别意图: {intent}")

        try:
            # ✅ 2. 根据意图分流处理
            if intent == "medical_inquiry":
                # ✅ 医疗意图：走 RAG 检索流程 (调用你原来的 ask_stream)
                chatbot.ask_stream(user_input)
            elif intent == "chat_general":
                # ✅ 闲聊意图：不检索，直接让模型基于通用知识回答
                print("💬 AI (通用模式): ", end="", flush=True)
                # ✅ 这里可以写一个简化的流式输出，或者直接调用 LLM
                # 为了演示，我们直接拼接一个回复
                response = f"你好！我是医疗助手。你刚才的问候我收到了。关于医疗问题，随时可以问我哦。\n\n(检测到你正在闲聊，本次未进行知识库检索)"
                for char in response:
                    print(char, end="", flush=True)
                    time.sleep(0.01)
            else:
                # ✅ 未知意图或其他
                print("💬 AI: ", end="", flush=True)
                chatbot.ask_stream(user_input)  # 默认走主流程

        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    main()
