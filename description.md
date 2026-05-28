# 医疗聊天机器人 — 项目设计文档

## 整体架构

```
用户 (API / CLI)
    │
    ▼
IntentClassifier  ── 意图分类 ──→  medical_inquiry  ──→ MedicalAgent (LangGraph)
    │                              chat_general      ──→ ToolManager (天气/闲聊)
    │                              system_query      ──→ ToolManager (通用LLM)
    │                              unknown           ──→ 兜底 Agent(RAG)
    │
    ▼
SystemInitializer  — 按依赖顺序组装组件:
  1. VectorStoreManager   (ChromaDB, 无依赖)
  2. MemoryStore          (Redis, 无依赖)
  3. IntentClassifier     (仅 LLM)
  4. ToolManager          (无依赖)
  5. MedicalAgent         (依赖 1/2/3/4)
  6. MedicalChatbot       (依赖 5)
```

## 模块详解

### 1. 配置管理 — `config/app_config.py`

所有可调参数集中在 `AppConfig` 类中，包括 LLM 模型名、嵌入模型名、检索参数（k=6, score_threshold=0.3）、分块参数（chunk_size=500, chunk_overlap=100）、Redis 连接（127.0.0.1:6379, TTL=24h）、ChromaDB 持久化路径等。

全局单例 `APP_CONFIG` 被所有模块直接引用。

### 2. 文档加载 — `src/service/document_loader.py`

- 扫描 `data/disease/` 下的 `.txt` / `.md` / `.mdx` 文件
- 使用 `RecursiveCharacterTextSplitter` 递归按 `\n\n → \n → 。 → ！ → ？ → ； → " " → ""` 分块
- 每块保留源文件元数据（source, file_path, file_type）并附加块索引
- 与旧版行为完全一致

### 3. 向量存储 — `src/service/vector_store.py`（ChromaDB 替代 FAISS）

- **创建**: `Chroma.from_documents()` → 嵌入 → 持久化到 `data/chroma_db/`
- **加载**: `Chroma(persist_directory=...)` 从磁盘重建集合
- **检索**: `similarity_search_with_relevance_scores()` + score_threshold 过滤
- 嵌入模型使用 `nomic-embed-text`（768 维），与旧版一致
- 提供 `@property vector_store` 懒加载，延迟初始化 ChromaDB

### 4. 对话记忆 — `src/service/memory_store.py`（Redis 替代 FAISS+内存）

**存储结构**:
- Key: `chat:{session_id}:messages`（Redis List）
- Value: JSON `{"role": "user"/"assistant", "content": "...", "timestamp": "..."}`
- TTL: 24 小时自动过期（每次写入刷新）

**核心方法**:
| 方法 | 功能 |
|---|---|
| `add_message(session_id, role, content)` | 追加消息到列表末尾 |
| `get_recent_messages(session_id, n)` | 返回最近 N 条消息 |
| `get_history_text(session_id, n)` | 拼接为 prompt 可读文本 |
| `clear_session(session_id)` | 清除会话全部历史 |

**故障回退**: Redis 不可用时静默降级，返回空历史，不影响主流程。

### 5. Agent — `src/service/agent.py`（LangGraph 替代 LCEL）

使用 `StateGraph` 构建的医疗咨询 Agent：

```
节点:
  classify_intent → 意图分类 (复用 IntentClassifier)
  retrieve_docs   → ChromaDB 相似度检索
  generate_answer → LLM 生成 (上下文 + 文档 + 历史)
  weather_query   → 天气 API 查询
  general_chat    → 通用 LLM 回复
  save_memory     → Redis 持久化

条件路由:
  classify_intent → medical_inquiry → retrieve_docs → generate_answer
                  → chat_general+天气 → weather_query
                  → chat_general/其他 → general_chat
                  → unknown → retrieve_docs (兜底)
```

状态定义 (`AgentState`):
```python
{
    "messages": List[Dict],      # 当前轮消息
    "session_id": str,           # 会话 ID
    "intent": str,               # 识别到的意图
    "context_docs": List[str],   # 检索到的文档文本
    "answer": str                # 生成的回答
}
```

### 6. 聊天机器人 — `src/service/chatbot.py`

简化为 `MedicalAgent` 的包装器，保持 `get_answer()` / `ask_stream()` 接口不变。

```python
class MedicalChatbot:
    def get_answer(self, question: str, session_id: str) -> dict:
        answer = self.agent.run(user_input=question, session_id=session_id)
        return {"answer": answer}
```

### 7. 意图分类 — `src/service/intent_classifier.py`

与旧版完全一致。使用 LLM（同 RAG 的 qwen2.5:7b）通过结构化 Prompt 输出 JSON。

```python
intents = ["medical_inquiry", "chat_general", "system_query"]
# 输出: {"intent": "medical_inquiry"}
```

### 8. 工具管理 — `src/service/tool_manager.py`

与旧版完全一致:
- **天气查询流水线**: `extract_city_from_text()`(正则) → `extract_city_by_llm()`(LLM 兜底) → `search_weather()`(高德 API) → LLM 润色输出
- **通用闲聊**: 直接走 LLM 生成

### 9. 路由分发 — `src/api/routers/chat_router.py`

三个端点:
| 端点 | 功能 |
|---|---|
| `POST /api/chat` | 普通问答，返回 `{intent, answer}` |
| `POST /api/chat/stream` | SSE 流式问答（全量返回后逐字符 yield） |
| `POST /api/chat/daily_news` | 新闻查询（聚合数据 API） |

**SSE 协议格式**:
```
data: {"intent":"medical_inquiry"}\n\n     ← 首个事件：意图
data: "你"\n\n                              ← 每字符一个事件
data: "好"\n\n
data: [DONE]\n\n                            ← 终止信号
```

### 10. CLI 模式 — `src/api/routers/cli_router.py`

与 API 共享同一套 `system_initializer` → 组件路由逻辑。使用 `@monitor_performance` 装饰器记录执行耗时。

### 11. 初始化 — `src/service/system_initializer.py`

按固定顺序初始化六个组件:
1. `_initialize_vector_store()` — VectorStoreManager (ChromaDB)
2. `_initialize_memory_store()` — MemoryStore (Redis)
3. `_initialize_intent_classifier()` — IntentClassifier
4. `_initialize_tool_manager()` — ToolManager
5. `_initialize_agent()` — MedicalAgent (注入 1-4)
6. `_initialize_chatbot()` — MedicalChatbot (注入 5)

API 模式由 FastAPI `@app.on_event("startup")` 触发；CLI 模式由 `run_cli()` 内按需触发。

## 外部依赖

| 依赖 | 用途 | 变更 |
|---|---|---|
| `fastapi` | Web 框架 | — |
| `uvicorn` | ASGI 服务器 | — |
| `pydantic` | 配置 & 请求/响应模型 | — |
| `langchain-core` | PromptTemplate、Document | — |
| `langchain-text-splitters` | RecursiveCharacterTextSplitter | — |
| `langchain-ollama` | OllamaLLM、OllamaEmbeddings | — |
| `langchain-chroma` | ChromaDB 向量存储 | **新增** |
| `chromadb` | 向量索引 | **新增** |
| `redis` | 对话记忆 | **新增** |
| `langgraph` | Agent 框架 | **新增** |
| `requests` | 高德天气 API | — |
| ~~`langchain-community`~~ | ~~FAISS 向量存储~~ | **移除** |
| ~~`faiss-cpu`~~ | ~~向量索引~~ | **移除** |
