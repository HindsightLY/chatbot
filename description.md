# 医疗聊天机器人项目文档

## 项目概述

本项目是一个基于大语言模型的医疗健康咨询系统，采用RAG（检索增强生成）技术构建，能够处理医疗咨询、天气查询、通用对话等多种用户需求。系统采用模块化设计，集成了文档加载、向量化存储、意图识别、对话管理、混合记忆存储等核心功能。

## 技术架构与选型

### 核心技术栈
- **Python 3.8+** - 主要开发语言
- **FastAPI** - Web框架，提供高性能API服务
- **LangChain** - LLM应用开发框架，提供文档处理、链式调用等功能
- **FAISS** - 向量数据库，用于高效相似性搜索
- **Ollama** - 本地LLM服务，部署和管理大语言模型
- **OllamaEmbeddings** - 嵌入模型，使用nomic-embed-text进行文本向量化
- **OllamaLLM** - 大语言模型，使用qwen2.5:7b进行推理和意图识别

### 项目结构
```
medical_chatbot/
├── config/                 # 配置文件目录 (API Keys, 系统参数)
├── data/                   # 数据存储目录
│   ├── disease/            # 疾病相关原始数据 (TXT等)
│   ├── faiss_index/        # 构建好的 FAISS 向量索引文件
│   └── chat_memory/        # 对话历史向量存储
├── src/                    # 源代码核心目录
│   ├── api/                # API 接口层
│   │   └── routers/        # 路由定义
│   │       └── chat_router.py  # 聊天接口路由 (/api/chat)
│   ├── service/            # 业务逻辑层
│   │   ├── system_initializer.py # 系统启动初始化
│   │   └── tool_manager.py     # 工具调度管理器
│   ├── chatbot.py          # 核心聊天机器人逻辑
│   ├── document_loader.py  # 文档加载与预处理
│   ├── intent_classifier.py# 基于LLM的意图识别分类器
│   ├── main.py             # 程序入口 (FastAPI 启动)
│   └── vector_store.py     # 向量数据库封装 (FAISS)
└── README.md               # 项目说明文档
```

## 项目实现思路与关键技术详解

### 1. RAG系统实现
**技术实现**: 使用LangChain框架构建RAG系统
- **文档加载**: 通过`DocumentLoader`类使用LangChain的文本分割器将医疗文档切分为块
- **向量化**: 使用`OllamaEmbeddings`将文档块转换为向量表示
- **存储**: 利用`FAISS.from_documents()`方法创建向量索引并持久化存储
- **检索**: 通过`vector_store.as_retriever()`创建检索器，使用`similarity_search`方法查找相关文档

**关键代码**: 在`vector_store.py`中使用`FAISS.load_local()`加载已构建的向量索引，在`chatbot.py`中通过`retriever`进行文档检索

### 2. 意图识别系统
**技术实现**: 基于大语言模型的意图分类
- **模型选择**: 使用`langchain_ollama.OllamaLLM`加载qwen2.5:7b模型
- **分类逻辑**: 通过结构化Prompt让LLM理解意图schema，返回JSON格式的分类结果
- **实现方法**: 在`IntentClassifier.classify()`方法中，使用特定格式的Prompt模板，要求模型返回预定义意图类别之一

**关键代码**: 
```python
self.llm = OllamaLLM(model=model_name, base_url=APP_CONFIG.llm_base_url)
# 通过自定义Prompt进行意图分类，返回medical_inquiry、chat_general、system_query等类别
```

### 3. 混合记忆管理系统
**技术实现**: 结合内存缓存和向量存储的双重记忆机制
- **内存缓存**: 使用字典结构存储最近10轮对话（`max_cache_turns=10`）
- **向量存储**: 使用FAISS存储长期对话历史，通过`FAISS.load_local()`和`FAISS.save_local()`实现持久化
- **检索机制**: 通过`similarity_search_with_score()`方法按会话ID过滤并搜索相关历史对话

**关键代码**: 在`HybridChatMemory`类中实现了内存缓存与FAISS向量存储的结合，使用`add_documents()`方法添加对话历史，使用`similarity_search_with_score()`进行相关性检索

### 4. 对话管理与响应生成
**技术实现**: LangChain链式调用与Prompt工程
- **Prompt设计**: 使用`PromptTemplate.from_template()`创建包含对话历史、医学资料和用户问题的结构化模板
- **链式调用**: 通过LangChain的链式操作符`|`组合retriever、prompt和llm
- **响应生成**: 使用`create_stuff_documents_chain`创建文档处理链，将检索到的文档内容整合到响应中

**关键代码**:
```python
self.qa_chain = {
    "context": self.retriever,
    "input": lambda x: x["input"],
    "relevant_history": lambda x: self.hybrid_memory.get_relevant_history(x["session_id"], x["input"])
} | self.prompt | self.llm
```

### 5. 系统初始化流程
**技术实现**: 依赖注入与组件管理
- **初始化顺序**: 通过`SystemInitializer`类按序初始化向量存储、意图分类器、聊天机器人
- **资源管理**: 自动检测是否存在已构建的向量索引，若不存在则自动创建
- **生命周期管理**: 在FastAPI的`@app.on_event("startup")`事件中执行系统初始化

**关键代码**: 在`system_initializer.py`中通过`load_vector_store()`和`create_vector_store()`方法管理向量存储生命周期

### 6. API服务与路由
**技术实现**: FastAPI框架提供RESTful API
- **路由定义**: 使用FastAPI的`APIRouter`定义聊天接口
- **请求处理**: 接收JSON格式的查询请求，包含用户问题和会话ID
- **响应格式**: 返回包含意图类型和答案的JSON响应

**关键代码**: 在`chat_router.py`中定义POST接口`/api/chat`，处理用户查询并返回结构化响应

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
- FAISS索引持久化，避免重复构建
- 设置相似度阈值(`score_threshold: 0.3`)提高检索质量
- 限制返回文档数量(`k: 6`)控制响应长度

### 2. 内存管理优化
- 混合记忆存储机制平衡内存使用和访问速度
- 限制每会话缓存轮数防止内存溢出
- 自动将过期对话迁移到向量存储

### 3. 缓存策略
- 本地向量索引避免重复计算
- 对话历史分层存储策略
- 配置信息全局单例模式

## 扩展性设计

### 1. 插件化意图识别
- 可扩展的意图schema定义
- 易于添加新的意图类别
- 统一的意图处理接口

### 2. 工具集成能力
- 预留工具管理接口
- 支持外部API集成（如天气查询）
- 统一的工具调用规范

### 3. 多模态支持预留
- 文档加载器支持多种格式
- 向量存储支持不同类型的embedding
- 响应生成器可适配不同LLM

这套系统设计体现了现代AI应用的核心架构思想：通过向量化存储实现知识检索，利用大语言模型进行理解和生成，结合记忆机制提升对话连贯性，形成完整的智能对话解决方案。