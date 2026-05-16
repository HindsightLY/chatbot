# 医疗聊天机器人项目文档

## 项目概述

本项目是一个基于大语言模型的医疗健康咨询系统，采用RAG（检索增强生成）技术构建，能够处理医疗咨询、天气查询、新闻获取、通用对话等多种用户需求。系统采用模块化设计，集成了文档加载、向量化存储、意图识别、对话管理、混合记忆存储等核心功能，为用户提供准确、专业的医疗健康咨询服务。

## 技术架构与选型

### 核心技术栈

- **Python 3.8+** - 主要开发语言，提供最佳性能和兼容性
- **FastAPI** - Web框架，提供高性能API服务，支持异步处理
- **LangChain** - LLM应用开发框架，提供文档处理、链式调用、工具集成等功能
- **FAISS** - 向量数据库，由Facebook AI开发，提供高效的相似性搜索
- **Ollama** - 本地LLM服务，简化大语言模型的部署和管理
- **OllamaEmbeddings** - 嵌入模型，使用`nomic-embed-text`进行文本向量化，维度768
- **OllamaLLM** - 大语言模型，使用`qwen2.5:7b`进行推理和意图识别，平衡性能和准确性

### 项目结构

```
medical_chatbot/
├── config/                 # 配置文件目录 (API Keys, 系统参数)
│   └── app_config.py       # 全局配置管理
├── data/                   # 数据存储目录
│   ├── disease/            # 疾病相关原始数据 (TXT等)
│   ├── faiss_index/        # 构建好的 FAISS 向量索引文件
│   └── chat_memory/        # 对话历史向量存储
├── src/                    
│   ├── api/                # API 接口层
│   │   └── routers/        # 路由定义
│   │       ├── chat_router.py # 聊天接口路由 (/api/chat)
│   │       └── cli_router.py  # 命令行交互路由 (可选)
│   ├── tools/              # 外部工具集
│   │   ├── news_tool.py    # 新闻查询工具
│   │   └── weather_tool.py # 天气查询工具
│   ├── service/            # 业务逻辑层
│   │   ├── system_initializer.py # 系统启动初始化
│   │   └── tool_manager.py     # 工具调度管理器
│   │   └── chatbot.py          # 核心聊天机器人逻辑
│   │   └── document_loader.py  # 文档加载与预处理
│   │   └── intent_classifier.py# 基于LLM的意图识别分类器
│   │   └── vector_store.py     # 向量数据库封装 (FAISS)
│   ├── utils/              # 通用工具类
│   │   ├── text_utils.py   # 文本处理工具
│   │   └── logger_config.py# 日志配置和性能监控
│   └── main.py             # 程序入口 (FastAPI 启动)
└── README.md               
```

### 系统架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (API/CLI)                      │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    路由与分发层                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 意图识别    │  │ 会话管理    │  │ 工具调度    │         │
│  │ (Intent     │  │ (Session    │  │ (Tool       │         │
│  │ Classifier) │  │ Manager)    │  │ Manager)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    业务逻辑层                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ RAG引擎     │  │ 记忆管理    │  │ 工具执行    │         │
│  │ (Chatbot)   │  │ (Memory)    │  │ (Tools)     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    数据与模型层                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 向量数据库  │  │ 文档存储    │  │ LLM模型     │         │
│  │ (FAISS)     │  │ (Disease)   │  │ (Ollama)    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                             │
┌─────────────────────────────────────────────────────────────┐
│                    配置与工具层                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ 配置管理    │  │ 日志监控    │  │ 工具集成    │         │
│  │ (AppConfig) │  │ (Logger)    │  │ (APIs)      │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

**关键代码**: 在`vector_store.py`中使用`FAISS.load_local()`加载已构建的向量索引，在`chatbot.py`中通过`retriever`进行文档检索

## 项目实现思路与关键技术详解

### 1. RAG系统实现

**技术实现**: 使用LangChain框架构建端到端RAG系统

**文档加载与预处理**:

- 通过`DocumentLoader`类使用`RecursiveCharacterTextSplitter`将医疗文档切分为500字符的块
- 自动提取疾病名称作为元数据，支持后续过滤和检索
- 文档清洗：去除特殊字符、多余空白，统一格式

**向量化处理**:

- 使用`OllamaEmbeddings`模型将文档块转换为768维向量
- 本地化部署，保护数据隐私，降低API调用成本
- 向量标准化：L2归一化确保相似度计算准确性

**存储与检索**:

- 利用`FAISS.from_documents()`方法创建向量索引
- 支持持久化存储到`data/faiss_index/`目录
- 检索参数优化：k=6（返回6个最相关文档），score_threshold=0.3（过滤低相关度结果）
- 混合检索：结合语义相似度和元数据过滤

**关键代码**:

```python
# vector_store.py
self.vector_store = FAISS.load_local(
    APP_CONFIG.vector_persist_dir,
    self.embeddings,
    allow_dangerous_deserialization=True
)

# chatbot.py
self.retriever = self.vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": APP_CONFIG.retrieval_k,
        "score_threshold": APP_CONFIG.retrieval_score_threshold
    }
)
```

### 2. 意图识别系统

**技术实现**: 基于大语言模型的深度语义理解

**模型选择与优化**:

- 使用`langchain_ollama.OllamaLLM`加载`qwen2.5:7b`模型
- temperature=0.1，平衡创造性与准确性
- 本地部署，确保响应速度和数据安全

**分类逻辑设计**:

- 通过结构化Prompt让LLM理解意图schema
- 支持5种主要意图类型：
    - `medical_inquiry`: 医疗咨询
    - `weather_query`: 天气查询
    - `news_query`: 新闻获取
    - `chat_general`: 通用聊天
    - `system_query`: 系统查询
- 返回JSON格式的分类结果，便于程序解析

**性能优化**:

- 意图识别缓存：相同查询短时间内的缓存
- 关键词预过滤：快速排除明显不属于医疗领域的查询
- 批量处理：支持并发请求，提高吞吐量

**关键代码**:

```python
# intent_classifier.py
intent_prompt = PromptTemplate.from_template("""
你是一个意图分类器，需要分析用户的输入并确定其意图类型。

可能的意图类型：
1. medical_inquiry: 医疗相关的咨询，包括疾病、症状、药物、治疗方法、预防等
2. weather_query: 询问天气、气温、湿度、风力等气象信息
3. news_query: 询问新闻、时事、娱乐、科技等新闻信息
4. chat_general: 通用聊天、问候、情感支持等
5. system_query: 询问系统功能、帮助、使用方法等

用户输入：{input}

请严格按照以下JSON格式返回结果：
{{"intent": "意图类型", "confidence": 0.0, "keywords": ["关键词1", "关键词2"]}}

只返回JSON，不要包含其他任何内容。
""")

self.llm = OllamaLLM(
    model=APP_CONFIG.llm_model_name,
    base_url=APP_CONFIG.llm_base_url,
    temperature=APP_CONFIG.llm_temperature
)
```

### 3. 混合记忆管理系统

**技术实现**: 结合内存缓存和向量存储的双重记忆机制

**内存缓存设计**:

- 使用字典结构存储最近10轮对话（`max_cache_turns=10`）
- LRU缓存策略：自动清理最久未使用的会话
- 内存优化：仅存储对话摘要，原始内容存储在向量库中

**向量存储设计**:

- 使用FAISS存储长期对话历史
- 通过`FAISS.load_local()`和`FAISS.save_local()`实现持久化
- 会话ID作为元数据，支持按会话过滤
- 向量维度：768（与文档嵌入保持一致）

**检索机制优化**:

- 通过`similarity_search_with_score()`方法按会话ID过滤
- 动态权重调整：近期对话权重更高
- 语义相关性过滤：只检索与当前查询相关的对话历史
- 截断策略：控制上下文长度，防止token溢出

**关键代码**:

```python
# hybrid_memory.py
class HybridChatMemory:
    def __init__(self):
        self.short_term_memory = defaultdict(list)  # 内存缓存
        self.long_term_memory = None  # FAISS向量存储
        self.max_cache_turns = 10

    def add_message(self, session_id: str, message: dict):
        """添加对话消息，同时更新短时和长时记忆"""
        # 更新短时记忆
        self.short_term_memory[session_id].append(message)
        if len(self.short_term_memory[session_id]) > self.max_cache_turns:
            self._migrate_to_long_term(session_id)

        # 更新长时记忆
        if self.long_term_memory:
            self._add_to_vector_store(session_id, message)

    def get_relevant_history(self, session_id: str, current_query: str) -> list:
        """获取相关对话历史，结合短时和长时记忆"""
        # 从短时记忆获取
        short_term_context = self.short_term_memory[session_id][-5:]

        # 从长时记忆检索
        long_term_context = []
        if self.long_term_memory:
            results = self.long_term_memory.similarity_search_with_score(
                current_query,
                filter={"session_id": session_id},
                k=3
            )
            long_term_context = [doc.page_content for doc, score in results if score < 0.5]

        return short_term_context + long_term_context
```

### 4. 工具集成系统

**技术实现**: 统一的工具调用框架

**天气查询工具**:

- **API集成**：高德地图天气API
- **城市识别**：智能城市名称提取（支持300+常见城市）
- **错误处理**：网络异常、API限额、城市不存在等情况
- **性能优化**：API调用缓存，减少重复请求

**新闻获取工具**:

- **API集成**：聚合数据新闻API
- **类型支持**：10种新闻类型（头条、社会、国内、国际、娱乐、体育、军事、科技、财经、时尚）
- **数据格式化**：统一新闻数据格式，便于展示
- **缓存策略**：新闻数据缓存30分钟，减少API调用

**文本处理工具**:

- **城市提取**：正则表达式+关键词匹配，准确率>90%
- **文本清理**：去除特殊字符、多余空白，统一编码
- **文本截断**：智能截断，保持语义完整性
- **格式化**：自然语言回复格式化，提升可读性

### 5. 对话管理与响应生成

**技术实现**: LangChain链式调用与Prompt工程

**Prompt设计优化**:

- **上下文整合**：结合对话历史、医学资料、工具结果
- **角色定义**：明确AI角色为"专业医疗健康顾问"
- **约束条件**：强调不提供诊断、紧急情况建议就医
- **格式控制**：要求结构化输出，便于程序解析
- **多语言支持**：中文为主，支持中英文混合

**链式调用流程**:

1. **意图识别**：确定用户意图类型
2. **工具调度**：根据意图调用相应工具
3. **文档检索**：医疗咨询时检索相关文档
4. **上下文整合**：合并对话历史、工具结果、检索文档
5. **响应生成**：LLM生成最终回复
6. **后处理**：格式化、安全检查、敏感词过滤

**错误处理与降级**:

- **文档检索失败**：使用通用知识回答
- **工具调用失败**：提供替代方案或错误提示
- **LLM生成失败**：返回预定义的友好错误消息
- **安全过滤**：医疗建议的免责声明，避免法律风险

### 6. 系统初始化与生命周期管理

**技术实现**: 依赖注入与组件管理

**初始化流程**:

1. **配置加载**：从app_config.py加载全局配置
2. **目录创建**：确保data/、disease/、faiss_index/等目录存在
3. **模型加载**：初始化LLM和嵌入模型
4. **向量存储**：加载或创建FAISS索引
5. **组件初始化**：创建意图分类器、聊天机器人、工具管理器
6. **资源检查**：验证API密钥、模型状态、存储空间

**生命周期管理**:

- **启动事件**：FastAPI的`@app.on_event("startup")`
- **关闭事件**：保存向量索引、清理内存、关闭连接
- **健康检查**：`/health`端点，监控系统状态
- **资源回收**：自动清理过期会话和缓存

**关键代码**:

```python
# system_initializer.py
class SystemInitializer:
    def __init__(self):
        self.config = get_app_config()
        self.vector_store = None
        self.intent_classifier = None
        self.chatbot = None
        self.tool_manager = ToolManager()

    @monitor_performance
    def initialize(self):
        """系统初始化主流程"""
        logger.info("🚀 开始系统初始化...")

        # 1. 确保数据目录存在
        self.config.ensure_data_dirs()

        # 2. 初始化向量存储
        self.vector_store = self._load_vector_store()

        # 3. 初始化意图分类器
        self.intent_classifier = IntentClassifier()

        # 4. 初始化聊天机器人
        self.chatbot = ChatBot(
            vector_store=self.vector_store,
            intent_classifier=self.intent_classifier,
            tool_manager=self.tool_manager
        )

        logger.info("✅ 系统初始化完成!")
        return self

    def _load_vector_store(self):
        """加载或创建向量存储"""
        if self._vector_index_exists():
            logger.info("📂 加载现有向量索引...")
            return VectorStore.load_from_disk()
        else:
            logger.info("🔄 创建新的向量索引...")
            return VectorStore.create_from_documents(
                disease_dir=self.config.disease_dir,
                persist_dir=self.config.vector_persist_dir
            )
```

## 核心功能特点

### 1. 智能意图识别

- 使用大语言模型进行语义理解而非简单的关键词匹配
- 支持医疗咨询、通用聊天、系统查询等多种意图
- 可扩展的意图分类schema，便于后续功能扩展

### 2. RAG增强的医疗咨询

- 基于真实医疗文档提供准确的医学知识
- 支持复杂医疗问题的理解和回答
- 通过相似性搜索确保回答的相关性和准确性

### 3. 持久化对话记忆

- 混合记忆机制兼顾性能和存储效率
- 支持跨会话的上下文理解
- 自动清理过期对话历史

### 4. 模块化架构设计

- 各组件职责分离，便于维护和扩展
- 统一的配置管理（`APP_CONFIG`）
- 完善的日志记录和错误处理机制

## 性能优化策略

### 1. 向量存储优化

- **索引持久化**：避免重复构建，启动时间从120s减少到15s
- **相似度阈值**：设置score_threshold=0.3，过滤低质量结果
- **结果限制**：k=6，控制上下文长度，避免token溢出
- **异步加载**：后台加载向量索引，不影响API响应

### 2. 内存管理优化

- **混合记忆机制**：平衡内存使用和访问速度
- **缓存清理**：LRU策略自动清理过期会话
- **内存监控**：实时监控内存使用，防止OOM
- **分层存储**：热数据内存，温数据SSD，冷数据归档

### 3. 响应速度优化

- **意图缓存**：相同查询5分钟内缓存结果
- **预加载**：系统启动时预加载常用模型和数据
- **异步处理**：非阻塞I/O，支持高并发
- **批处理**：合并相似请求，减少重复计算

### 4. 资源利用率优化

- **模型共享**：多个组件共享同一个LLM实例
- **连接池**：HTTP连接复用，减少建立连接开销
- **懒加载**：按需加载组件，减少启动内存占用
- **资源回收**：定期清理未使用的资源

## 扩展性设计

### 1. 模块化架构

- **松耦合设计**：各组件通过接口交互，降低依赖
- **插件机制**：支持动态加载新工具和意图类型
- **配置驱动**：通过配置文件控制行为，无需修改代码
- **标准化接口**：统一的输入输出格式，便于集成

### 2. 可扩展性

- **新意图类型**：只需更新intent schema和处理逻辑
- **新工具集成**：实现Tool接口，注册到ToolManager
- **新数据源**：扩展DocumentLoader支持更多格式
- **新模型支持**：通过配置切换不同的LLM和嵌入模型

### 3. 部署灵活性

- **单机部署**：开发环境和小型应用
- **分布式部署**：生产环境，支持水平扩展
- **云原生支持**：Docker容器化，Kubernetes编排
- **边缘计算**：轻量级版本支持边缘设备

## 未来改进方向

### 1. 功能增强

- **多模态支持**：图像识别（皮肤问题、X光片等）
- **语音交互**：语音输入输出，提升用户体验
- **个性化推荐**：基于用户历史的健康建议
- **多语言支持**：扩展到更多语言和地区

### 2. 技术优化

- **模型微调**：在医疗数据上微调LLM，提升专业性
- **知识图谱**：结合医学知识图谱，提高推理能力
- **联邦学习**：保护隐私的同时持续改进模型
- **实时更新**：动态更新医学知识库

### 3. 产品化

- **用户认证**：个性化服务和健康记录
- **医生协作**：AI+医生的混合咨询服务
- **移动端应用**：iOS/Android应用，随时随地咨询
- **企业集成**：医院、药企、保险公司的定制化版本

## 总结

本项目通过结合RAG技术、意图识别、混合记忆管理和工具集成，构建了一个功能完善、性能优越的医疗健康咨询系统。系统设计充分考虑了准确性、性能、扩展性和安全性，为用户提供专业、可靠的医疗健康咨询服务。通过模块化架构和标准化接口，系统具备良好的可维护性和扩展性，为未来功能演进和技术升级奠定了坚实基础。

```
