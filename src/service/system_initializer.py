"""
系统初始化服务
负责系统的整体初始化逻辑
"""
from typing import Tuple
from src.logger_config import logger
from config.app_config import APP_CONFIG
from src.document_loader import DocumentLoader
from src.vector_store import VectorStoreManager
from src.intent_classifier import IntentClassifier
from src.chatbot import MedicalChatbot


class SystemInitializer:
    """
    系统初始化器
    负责初始化所有核心组件
    """

    def __init__(self):
        """初始化系统初始化器"""
        self.vector_store = None
        self.intent_classifier = None
        self.chatbot = None

    def initialize_system(self) -> Tuple[object, object, object]:
        """
        初始化整个系统

        Returns:
            Tuple: (vector_store, intent_classifier, chatbot)

        Raises:
            Exception: 初始化失败时抛出异常
        """
        logger.info("🔄 开始初始化医疗AI系统...")
        logger.info(f"📁 项目根目录: {APP_CONFIG.project_root}")
        logger.info(f"📁 数据目录: {APP_CONFIG.data_dir}")
        logger.info(f"📁 疾病文档目录: {APP_CONFIG.disease_dir}")
        logger.info(f"📁 向量索引目录: {APP_CONFIG.vector_persist_dir}")

        # 1. 初始化向量存储
        self.vector_store = self._initialize_vector_store()

        # 2. 初始化意图分类器
        self.intent_classifier = self._initialize_intent_classifier()

        # 3. 初始化聊天机器人
        self.chatbot = self._initialize_chatbot()

        logger.info("✅ 系统初始化完成！")

        return self.vector_store, self.intent_classifier, self.chatbot

    def _initialize_vector_store(self) -> object:
        """
        初始化向量存储

        Returns:
            object: 向量存储实例
        """
        logger.info("📦 初始化向量存储...")

        # 创建向量存储管理器
        vector_manager = VectorStoreManager()

        # 尝试加载现有向量库
        store = vector_manager.load_vector_store()

        if store is None:
            logger.info("🔄 未找到现有向量库，正在创建新的向量存储...")

            # 加载文档
            doc_loader = DocumentLoader()
            documents = doc_loader.load_and_split_documents()

            if not documents:
                raise Exception("❌ 没有加载到任何文档，无法初始化向量存储")

            # 创建新的向量库
            store = vector_manager.create_vector_store(documents)
            logger.info("✅ 新向量存储创建完成！")
        else:
            logger.info("✅ 成功加载现有向量库！")

        return store

    def _initialize_intent_classifier(self) -> object:
        """
        初始化意图分类器

        Returns:
            object: 意图分类器实例
        """
        logger.info("🎯 初始化意图分类器...")
        classifier = IntentClassifier(model_name=APP_CONFIG.llm_model_name)
        logger.info("✅ 意图分类器初始化完成！")
        return classifier

    def _initialize_chatbot(self) -> object:
        """
        初始化聊天机器人

        Returns:
            object: 聊天机器人实例
        """
        logger.info("🤖 初始化聊天机器人...")
        chatbot = MedicalChatbot(self.vector_store)
        logger.info("✅ 聊天机器人初始化完成！")
        return chatbot


# 全局系统初始化器实例
system_initializer = SystemInitializer()