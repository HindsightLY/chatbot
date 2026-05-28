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

错误恢复策略:
  - 每个组件独立 try/except，单组件失败不影响其他组件
  - 缺失 Ollama/Redis 时记录警告而非崩溃
  - 调用方通过检查属性是否为 None 判断组件状态
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
    各组件独立容错，单组件失败不影响系统启动。
    """

    def __init__(self):
        self.vector_store = None
        self.memory_store = None
        self.intent_classifier = None
        self.tool_manager = None
        self.agent = None
        self.chatbot = None
        self.initialized = False
        self.init_errors = []

    def initialize_system(self) -> Tuple[object, object, object]:
        """
        按依赖顺序初始化所有组件，单组件失败不影响其他组件

        Returns:
            (vector_store, intent_classifier, chatbot)
        """
        logger.info("🔄 开始初始化医疗AI系统...")
        logger.info(f"📁 项目根目录: {APP_CONFIG.project_root}")
        logger.info(f"📁 数据目录: {APP_CONFIG.data_dir}")
        logger.info(f"📁 疾病文档目录: {APP_CONFIG.disease_dir}")
        logger.info(f"📁 ChromaDB 索引目录: {APP_CONFIG.chroma_persist_dir}")

        self.init_errors = []

        import urllib.request
        import urllib.error
        ollama_url = APP_CONFIG.llm_base_url.rstrip("/") + "/api/tags"
        try:
            urllib.request.urlopen(ollama_url, timeout=3)
            logger.info(f"✅ Ollama 服务连接成功: {APP_CONFIG.llm_base_url}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            logger.warning(f"⚠️ Ollama 服务未响应 ({APP_CONFIG.llm_base_url}): {e}")
            logger.warning("⚠️ 请确保已启动 Ollama: ollama serve")
            self.init_errors.append(f"Ollama 未运行 ({APP_CONFIG.llm_base_url})")

        if self.vector_store is None:
            self.vector_store = self._initialize_vector_store()
        if self.memory_store is None:
            self.memory_store = self._initialize_memory_store()
        if self.intent_classifier is None:
            self.intent_classifier = self._initialize_intent_classifier()
        if self.tool_manager is None:
            self.tool_manager = self._initialize_tool_manager()
        if self.agent is None:
            self.agent = self._initialize_agent()
        if self.chatbot is None:
            self.chatbot = self._initialize_chatbot()

        if self.init_errors:
            logger.warning(f"⚠️ 系统初始化完成，但存在 {len(self.init_errors)} 个错误:")
            for err in self.init_errors:
                logger.warning(f"  - {err}")
        else:
            logger.info("✅ 系统初始化完成！")

        self.initialized = True
        return self.vector_store, self.intent_classifier, self.chatbot

    def _initialize_vector_store(self):
        """
        初始化 ChromaDB 向量存储:
          存在缓存且非空 → 直接加载
          不存在/为空     → 从文档目录加载文件 → 分块 → 创建 Chroma 集合
        """
        logger.info("📦 初始化 ChromaDB 向量存储...")
        try:
            vector_manager = VectorStoreManager()
            store = vector_manager.vector_store

            need_rebuild = False
            if store is None:
                need_rebuild = True
            else:
                try:
                    count = store._collection.count()
                    if count == 0:
                        need_rebuild = True
                        logger.info("📁 ChromaDB 集合为空，将重新创建")
                    else:
                        logger.info(f"✅ 成功加载现有 ChromaDB，包含 {count} 个文档")
                except Exception:
                    need_rebuild = True
                    logger.info("📁 ChromaDB 状态异常，将重新创建")

            if need_rebuild:
                logger.info("🔄 正在创建新的向量存储...")
                doc_loader = DocumentLoader()
                documents = doc_loader.load_and_split_documents()
                if not documents:
                    raise Exception("❌ 没有加载到任何文档，无法初始化向量存储")
                store = vector_manager.create_vector_store(documents)
                logger.info("✅ 新 ChromaDB 创建完成！")

            return vector_manager
        except Exception as e:
            err_msg = f"向量存储初始化失败: {e}"
            logger.error(err_msg)
            self.init_errors.append(err_msg)
            return None

    def _initialize_memory_store(self):
        """初始化 Redis 对话记忆存储（容错：Redis 不可用时返回 None）"""
        logger.info("💾 初始化 Redis 对话记忆...")
        try:
            store = MemoryStore()
            if store._client is None:
                logger.warning("⚠️ Redis 未连接，对话记忆将降级为空")
            else:
                logger.info("✅ Redis 记忆存储初始化完成！")
            return store
        except Exception as e:
            err_msg = f"Redis 记忆存储初始化失败: {e}"
            logger.warning(f"⚠️ {err_msg}，对话记忆将降级")
            self.init_errors.append(err_msg)
            return MemoryStore.__new__(MemoryStore)

    def _initialize_intent_classifier(self):
        """
        初始化意图分类器（容错：LLM 不可用时记录警告）
        """
        logger.info("🎯 初始化意图分类器...")
        try:
            classifier = IntentClassifier(model_name=APP_CONFIG.llm_model_name)
            logger.info("✅ 意图分类器初始化完成！")
            return classifier
        except Exception as e:
            err_msg = f"意图分类器初始化失败 (Ollama 可能未运行): {e}"
            logger.warning(f"⚠️ {err_msg}")
            self.init_errors.append(err_msg)
            return None

    def _initialize_tool_manager(self):
        """初始化工具管理器（容错：LLM 不可用时记录警告）"""
        logger.info("🛠️ 初始化工具管理器...")
        try:
            tm = ToolManager()
            logger.info("✅ 工具管理器初始化完成！")
            return tm
        except Exception as e:
            err_msg = f"工具管理器初始化失败 (Ollama 可能未运行): {e}"
            logger.warning(f"⚠️ {err_msg}")
            self.init_errors.append(err_msg)
            return None

    def _initialize_agent(self):
        """
        初始化 LangGraph Agent

        依赖 vector_store / memory_store / intent_classifier / tool_manager 已就绪。
        必要组件缺失时返回 None。
        """
        if self.intent_classifier is None:
            logger.warning("⚠️ 意图分类器未就绪，跳过 Agent 初始化")
            self.init_errors.append("Agent 初始化跳过：意图分类器未就绪")
            return None
        logger.info("🤖 初始化 LangGraph Agent...")
        try:
            agent = MedicalAgent(
                vector_store=self.vector_store,
                memory_store=self.memory_store,
                intent_classifier=self.intent_classifier,
                tool_manager=self.tool_manager
            )
            logger.info("✅ Agent 初始化完成！")
            return agent
        except Exception as e:
            err_msg = f"Agent 初始化失败: {e}"
            logger.error(f"❌ {err_msg}")
            self.init_errors.append(err_msg)
            return None

    def _initialize_chatbot(self):
        """初始化聊天机器人（依赖 agent 已就绪）"""
        if self.agent is None:
            logger.warning("⚠️ Agent 未就绪，跳过 Chatbot 初始化")
            return None
        logger.info("💬 初始化聊天机器人...")
        try:
            chatbot = MedicalChatbot(agent=self.agent)
            logger.info("✅ 聊天机器人初始化完成！")
            return chatbot
        except Exception as e:
            err_msg = f"Chatbot 初始化失败: {e}"
            logger.error(f"❌ {err_msg}")
            self.init_errors.append(err_msg)
            return None


# 全局单例，由 main.py 的 startup 事件或 cli_router 触发初始化
system_initializer = SystemInitializer()
