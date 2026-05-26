"""
系统初始化服务
负责按依赖顺序组装系统核心组件

初始化顺序（有严格依赖关系）:
  1. VectorStoreManager   — 向量库（无依赖）
  2. IntentClassifier     — 意图分类器（无依赖，仅需 LLM）
  3. MedicalChatbot       — 聊天机器人（依赖 vector_store）
"""
from typing import Tuple
from src.utils.logger_config import logger
from config.app_config import APP_CONFIG
from src.service.document_loader import DocumentLoader
from src.service.vector_store import VectorStoreManager
from src.service.intent_classifier import IntentClassifier
from src.service.chatbot import MedicalChatbot


class SystemInitializer:
    """
    系统初始化器，管理所有核心组件的生命周期

    chat_router.py 和 cli_router.py 通过访问本类的属性来获取组件实例。
    组件的实际初始化由 initialize_system() 延迟触发。
    """

    def __init__(self):
        self.vector_store = None
        self.intent_classifier = None
        self.chatbot = None

    def initialize_system(self) -> Tuple[object, object, object]:
        """
        按依赖顺序初始化所有组件

        Returns:
            (vector_store, intent_classifier, chatbot)

        Raises:
            Exception: 文档不存在或向量库创建失败时终止
        """
        logger.info("🔄 开始初始化医疗AI系统...")
        logger.info(f"📁 项目根目录: {APP_CONFIG.project_root}")
        logger.info(f"📁 数据目录: {APP_CONFIG.data_dir}")
        logger.info(f"📁 疾病文档目录: {APP_CONFIG.disease_dir}")
        logger.info(f"📁 向量索引目录: {APP_CONFIG.vector_persist_dir}")

        self.vector_store = self._initialize_vector_store()
        self.intent_classifier = self._initialize_intent_classifier()
        self.chatbot = self._initialize_chatbot()

        logger.info("✅ 系统初始化完成！")

        return self.vector_store, self.intent_classifier, self.chatbot

    def _initialize_vector_store(self):
        """
        初始化向量存储:
          存在缓存 → 直接加载
          不存在    → 从文档目录加载文件 → 分块 → 创建 FAISS 索引
        """
        logger.info("📦 初始化向量存储...")

        vector_manager = VectorStoreManager()
        store = vector_manager.load_vector_store()

        if store is None:
            logger.info("🔄 未找到现有向量库，正在创建新的向量存储...")

            doc_loader = DocumentLoader()
            documents = doc_loader.load_and_split_documents()

            if not documents:
                raise Exception("❌ 没有加载到任何文档，无法初始化向量存储")

            store = vector_manager.create_vector_store(documents)
            logger.info("✅ 新向量存储创建完成！")
        else:
            logger.info("✅ 成功加载现有向量库！")

        return store

    def _initialize_intent_classifier(self):
        """
        初始化意图分类器

        使用与 RAG 相同的 LLM，未来可替换为轻量化分类模型。
        """
        logger.info("🎯 初始化意图分类器...")
        classifier = IntentClassifier(model_name=APP_CONFIG.llm_model_name)
        logger.info("✅ 意图分类器初始化完成！")
        return classifier

    def _initialize_chatbot(self):
        """
        初始化聊天机器人

        依赖 vector_store 已就绪，chatbot 内部再创建 HybridChatMemory。
        """
        logger.info("🤖 初始化聊天机器人...")
        chatbot = MedicalChatbot(self.vector_store)
        logger.info("✅ 聊天机器人初始化完成！")
        return chatbot


# 全局单例，由 main.py 的 startup 事件或 cli_router 触发初始化
system_initializer = SystemInitializer()
