"""
系统初始化服务
负责按依赖顺序组装系统核心组件

初始化顺序（有严格依赖关系）:
  1. VectorStoreManager   — ChromaDB 向量库（无依赖）
  2. MemoryStore          — Redis 对话记忆（无依赖）
  3. IntentClassifier     — 意图分类器（无依赖，仅需 LLM）
  4. ToolManager          — 工具管理器（无依赖）
  5. MedicalAgent         — LangGraph Agent（依赖 1/2/3/4）
  6. MedicalChatbot       — 聊天机器人（依赖 5）
"""
from typing import Tuple
from src.utils.logger_config import logger
from config.app_config import APP_CONFIG
from src.service.document_loader import DocumentLoader
from src.service.vector_store import VectorStoreManager
from src.service.memory_store import MemoryStore
from src.service.intent_classifier import IntentClassifier
from src.service.tool_manager import ToolManager
from src.service.agent import MedicalAgent
from src.service.chatbot import MedicalChatbot


class SystemInitializer:
    """
    系统初始化器，管理所有核心组件的生命周期

    chat_router.py 和 cli_router.py 通过访问本类的属性来获取组件实例。
    组件的实际初始化由 initialize_system() 延迟触发。
    """

    def __init__(self):
        self.vector_store = None
        self.memory_store = None
        self.intent_classifier = None
        self.tool_manager = None
        self.agent = None
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
        logger.info(f"📁 ChromaDB 索引目录: {APP_CONFIG.chroma_persist_dir}")

        self.vector_store = self._initialize_vector_store()
        self.memory_store = self._initialize_memory_store()
        self.intent_classifier = self._initialize_intent_classifier()
        self.tool_manager = self._initialize_tool_manager()
        self.agent = self._initialize_agent()
        self.chatbot = self._initialize_chatbot()

        logger.info("✅ 系统初始化完成！")

        return self.vector_store, self.intent_classifier, self.chatbot

    def _initialize_vector_store(self):
        """
        初始化 ChromaDB 向量存储:
          存在缓存 → 直接加载
          不存在    → 从文档目录加载文件 → 分块 → 创建 Chroma 集合
        """
        logger.info("📦 初始化 ChromaDB 向量存储...")

        vector_manager = VectorStoreManager()
        store = vector_manager.vector_store

        if store is None:
            logger.info("🔄 未找到现有向量库，正在创建新的向量存储...")

            doc_loader = DocumentLoader()
            documents = doc_loader.load_and_split_documents()

            if not documents:
                raise Exception("❌ 没有加载到任何文档，无法初始化向量存储")

            store = vector_manager.create_vector_store(documents)
            logger.info("✅ 新 ChromaDB 创建完成！")
        else:
            logger.info("✅ 成功加载现有 ChromaDB！")

        # 返回 VectorStoreManager 而不是原始 Chroma 实例，
        # 以便上游通过 .vector_store 属性访问
        return vector_manager

    def _initialize_memory_store(self):
        """初始化 Redis 对话记忆存储"""
        logger.info("💾 初始化 Redis 对话记忆...")
        store = MemoryStore()
        logger.info("✅ Redis 记忆存储初始化完成！")
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

    def _initialize_tool_manager(self):
        """初始化工具管理器"""
        logger.info("🛠️ 初始化工具管理器...")
        tm = ToolManager()
        logger.info("✅ 工具管理器初始化完成！")
        return tm

    def _initialize_agent(self):
        """
        初始化 LangGraph Agent

        依赖 vector_store / memory_store / intent_classifier / tool_manager 已就绪。
        """
        logger.info("🤖 初始化 LangGraph Agent...")
        agent = MedicalAgent(
            vector_store=self.vector_store,
            memory_store=self.memory_store,
            intent_classifier=self.intent_classifier,
            tool_manager=self.tool_manager
        )
        logger.info("✅ Agent 初始化完成！")
        return agent

    def _initialize_chatbot(self):
        """
        初始化聊天机器人

        依赖 agent 已就绪。
        """
        logger.info("💬 初始化聊天机器人...")
        chatbot = MedicalChatbot(agent=self.agent)
        logger.info("✅ 聊天机器人初始化完成！")
        return chatbot


# 全局单例，由 main.py 的 startup 事件或 cli_router 触发初始化
system_initializer = SystemInitializer()
