# 医疗聊天机器人 — 项目设计文档

## 整体架构

```
用户 (API / CLI)
    │
    ▼
IntentClassifier  ── 意图分类 ──→  medical_inquiry  ──→ MedicalChatbot (RAG)
    │                              chat_general      ──→ ToolManager (天气/闲聊)
    │                              system_query      ──→ ToolManager (通用LLM)
    │                              unknown           ──→ 兜底 RAG
    │
    ▼
SystemInitializer  — 按依赖顺序组装组件:
  1. VectorStoreManager   (无依赖)
  2. IntentClassifier     (仅 LLM)
  3. MedicalChatbot       (依赖 vector_store)
```

## 模块详解

### 1. 配置管理 — `config/app_config.py`

所有可调参数集中在 `AppConfig` 类中，包括 LLM 模型名、嵌入模型名、检索参数（k=6, score_threshold=0.3）、分块参数（chunk_size=500, chunk_overlap=100）、高德 API Key、城市列表、新闻类型等。

全局单例 `APP_CONFIG` 被所有模块直接引用。

### 2. 文档加载 — `src/service/document_loader.py`

- 扫描 `data/disease/` 下的 `.txt` / `.md` / `.mdx` 文件
- 使用 `RecursiveCharacterTextSplitter` 递归按 `\n\n → \n → 。 → ！ → ？ → ； → " " → ""` 分块
- 每块保留源文件元数据（source, file_path, file_type）并附加块索引

### 3. 向量存储 — `src/service/vector_store.py`

- **创建**: `FAISS.from_documents()` → 嵌入 → L2 归一化 → 持久化到 `data/faiss_index/`
- **加载**: `FAISS.load_local()` 从磁盘重建索引
- **检索**: `similarity_search_with_relevance_scores()` + score_threshold 过滤
- 嵌入模型使用 `nomic-embed-text`（768 维）
- 提供 `@property vector_store` 懒加载

### 4. 意图分类 — `src/service/intent_classifier.py`

使用 LLM（同 RAG 的 qwen2.5:7b）通过结构化 Prompt 输出 JSON 格式的分类结果。

```python
# 分类 Schema
intents = ["medical_inquiry", "chat_general", "system_query"]
# 规则: 医疗+闲聊混合时优先医疗
# 输出: {"intent": "medical_inquiry"}
```

返回 `"unknown"` 时由上游兜底走 RAG。

### 5. RAG 链 — `src/service/chatbot.py`

#### MedicalChatbot

使用 LCEL 构建管道:

```
{context: retriever, input, relevant_history} | prompt | llm
```

- **retriever**: 从 VectorStoreManager 获取，k=6, score_threshold=0.3
- **prompt**: 包含 `{context}`(检索文档)、`{relevant_history}`(对话历史)、`{input}`(当前问题)
- **llm**: OllamaLLM(qwen2.5:7b, temperature=0.1)

#### HybridChatMemory

两级存储:
- **内存缓存**: 每个 session 最近 10 轮对话（字典存储）
- **FAISS 存储**: 超出 10 轮后最早记录迁移到向量库，按 session_id 过滤

`get_relevant_history()` 合并两路结果: 内存最近记录 + FAISS 语义相似历史。

### 6. 工具管理 — `src/service/tool_manager.py`

- **天气查询流水线**: `extract_city_from_text()`(正则) → `extract_city_by_llm()`(LLM 兜底) → `search_weather()`(高德 API) → LLM 润色输出
- **通用闲聊**: 直接走 LLM 生成
- 城市提取采用两阶段策略确保召回率

### 7. 路由分发 — `src/api/routers/chat_router.py`

三个端点:
| 端点 | 功能 |
|---|---|
| `POST /api/chat` | 普通问答，返回 `{intent, answer}` |
| `POST /api/chat/stream` | SSE 流式问答（当前为全量返回后逐字符 yield） |
| `POST /api/chat/daily_news` | 新闻查询（聚合数据 API） |

### 8. CLI 模式 — `src/api/routers/cli_router.py`

与 API 共享同一套 `system_initializer` → `intent_classifier` → `chatbot`/`tool_manager` 路由逻辑。

使用 `@monitor_performance` 装饰器记录执行耗时。

### 9. 初始化 — `src/service/system_initializer.py`

按固定顺序初始化三个组件:
1. `_initialize_vector_store()` — 加载已有索引或新建
2. `_initialize_intent_classifier()` — 创建分类器
3. `_initialize_chatbot()` — 传入 vector_store

API 模式由 FastAPI `@app.on_event("startup")` 触发；CLI 模式由 `run_cli()` 内按需触发。

## 外部依赖

| 依赖 | 用途 |
|---|---|
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `pydantic` | 配置 & 请求/响应模型 |
| `langchain-core` | PromptTemplate、Document |
| `langchain-community` | FAISS 向量存储 |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter |
| `langchain-ollama` | OllamaLLM、OllamaEmbeddings |
| `faiss-cpu` | 向量索引 |
| `requests` | 高德天气 API |
