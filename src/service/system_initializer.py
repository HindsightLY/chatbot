"""
系统初始化服务 — 按依赖顺序组装系统核心组件

关键设计:
  - 每个组件独立 try/except，单组件失败不影响其他组件
  - Ollama/Redis 缺失时记录警告而非崩溃
  - 调用方通过检查属性是否为 None 判断组件状态

初始化顺序（严格的依赖关系）:
  1. VectorStoreManager   — ChromaDB 向量库（无依赖）
  2. MemoryStore          — Redis 对话记忆（无依赖）
  3. IntentClassifier     — 意图分类器（依赖 Ollama）
  4. ToolManager          — 工具管理器（无依赖）
  5. MedicalAgent         — LangGraph Agent（依赖 1/2/3/4）

注: MedicalChatbot 包装层已移除，路由层直接使用 MedicalAgent。
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


class SystemInitializer:
    """
    系统初始化器 — 管理所有核心组件的生命周期。

    chat_router.py / cli_router.py 通过访问本类的属性获取组件实例:
      system_initializer.agent
      system_initializer.vector_store
      system_initializer.intent_classifier
      等
    """

    def __init__(self):
        self.vector_store = None          # VectorStoreManager 实例
        self.memory_store = None          # MemoryStore 实例（Redis）
        self.intent_classifier = None     # IntentClassifier 实例
        self.tool_manager = None          # ToolManager 实例
        self.agent = None                 # MedicalAgent 实例
        self.initialized = False          # 初始化完成标记
        self.init_errors = []             # 初始化错误列表

    def initialize_system(self) -> Tuple[object, object, object]:
        """
        按依赖顺序初始化所有核心组件。

        容错策略:
          - 每个组件初始化独立 try/except，单组件失败不影响其他组件
          - Ollama/Redis 不可用时记录警告，系统以降级模式运行
          - 初始化前先检查 Ollama 服务是否可达（仅记录，不阻塞后续初始化）

        Returns:
            (vector_store, intent_classifier, agent) 三个核心组件引用
        """
        logger.info("🔄 开始初始化医疗AI系统...")
        logger.info(f"📁 项目根目录: {APP_CONFIG.project_root}")
        logger.info(f"📁 数据目录: {APP_CONFIG.data_dir}")
        logger.info(f"📁 疾病文档目录: {APP_CONFIG.disease_dir}")
        logger.info(f"📁 ChromaDB 索引目录: {APP_CONFIG.chroma_persist_dir}")

        self.init_errors = []

        # ── 检查 Ollama 服务是否可用（不阻塞后续初始化） ──
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

        # ── 按依赖顺序初始化各组件 ──
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

        if self.init_errors:
            logger.warning(f"⚠️ 系统初始化完成，但存在 {len(self.init_errors)} 个错误:")
            for err in self.init_errors:
                logger.warning(f"  - {err}")
        else:
            logger.info("✅ 系统初始化完成！")

        self.initialized = True
        return self.vector_store, self.intent_classifier, self.agent

    def _initialize_vector_store(self):
        """
        初始化 ChromaDB 向量存储。

        策略:
          - 缓存存在且非空 → 直接加载现有集合（BM25 索引首次搜索时懒加载）
          - 缓存不存在/为空 → 从 data/disease/ 加载文档 → 分块 → 创建集合 + BM25
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
        """
        初始化 Redis 对话记忆存储。

        容错:
          - Redis 连接失败时记录警告，返回无连接的 MemoryStore 实例
          - 后续所有读写操作静默返回空结果，不影响主流程
          - 即使 Redis 不可用也返回 MemoryStore 实例，避免调用方做 None 检查
        """
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
        初始化意图分类器。

        内部使用三引擎级联:
          1. BERT 分类器（sentence-transformers，~50ms，可离线，可选配置）
          2. LLM 分类器（OllamaLLM，~2s，依赖 Ollama 服务）
          3. 关键词规则兜底（~1ms）
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
        """初始化工具管理器（天气查询 + 通用闲聊 LLM）"""
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
        初始化 LangGraph Agent。

        依赖: vector_store, memory_store, intent_classifier, tool_manager
        已就绪。intent_classifier 不可用时跳过（无法做意图路由）。
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


# 全局单例，由 main.py 的 startup 事件或 cli_router 触发初始化
system_initializer = SystemInitializer()
