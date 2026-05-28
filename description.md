# 医疗聊天机器人 — 项目设计文档

## 整体架构

```
                         ┌─────────────────────────────────┐
                         │   用户接口层 (Presentation)      │
  ┌──────────────────┐   │  ┌──────────┐  ┌─────────────┐  │
  │ 浏览器 (index.html) │──┼─→│ FastAPI  │  │ CLI (input) │  │
  │  SSE 流式渲染      │   │  │ Router   │  │ 逐行交互    │  │
  └──────────────────┘   │  └────┬─────┘  └──────┬──────┘  │
                         └───────┼────────────────┼─────────┘
                                 │                │
                                 ▼                ▼
                         ┌──────────────────────────┐
                         │   意图路由层               │
                         │  IntentClassifier         │
                         │   (LLM → JSON 分类)       │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼──────────────────┐
                    ▼                 ▼                  ▼
            medical_inquiry    chat_general        system_query
                    │                 │                  │
                    ▼                 ▼                  ▼
          ┌─────────────────┐ ┌──────────────┐  ┌──────────────┐
          │ MedicalAgent    │ │ ToolManager  │  │ ToolManager  │
          │ (LangGraph)     │ │ 天气/闲聊    │  │ 通用 LLM     │
          │                 │ │              │  │              │
          │  classify_intent│ │ 城市提取     │  │ LLM 直接回答 │
          │  → retrieve_docs│ │  → 高德 API  │  │              │
          │  → generate     │ │  → LLM 润色  │  │              │
          │  → save_memory  │ │              │  │              │
          └────────┬────────┘ └──────────────┘  └──────────────┘
                   │
                   ▼
          ┌─────────────────────┐
          │   持久化层           │
          │  ┌──────┐ ┌──────┐  │
          │  │Redis │ │Chroma│  │
          │  │记忆  │ │向量  │  │
          │  └──────┘ └──────┘  │
          └─────────────────────┘
```

## 模块详解

### 1. 配置管理 — `config/app_config.py`

所有可调参数集中于 `AppConfig`（Pydantic BaseModel），全局单例 `APP_CONFIG`。

**配置分类**:

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| LLM | `llm_model_name` | `qwen2.5:7b` | 生成 & 分类共用 |
| LLM | `llm_base_url` | `http://localhost:11434` | Ollama 服务地址 |
| LLM | `llm_temperature` | `0.1` | 低温度保证一致性 |
| 嵌入 | `embedding_model_name` | `nomic-embed-text` | 768 维 |
| 检索 | `retrieval_k` | `6` | Top-K |
| 检索 | `retrieval_score_threshold` | `0.3` | 相关性过滤 |
| 分块 | `chunk_size` / `chunk_overlap` | `500` / `100` | 字符级 |
| Redis | `redis_host` / `port` / `db` | `127.0.0.1:6379/0` | — |
| Redis | `redis_ttl` | `86400` | 24h 自动过期 |
| Chroma | `chroma_collection_name` | `medical_docs` | 集合名 |
| 高德 | `amap_api_key` | 内嵌 | 天气 API |
| 新闻 | `valid_news_types` | 10 种 | 聚合数据 API |

> **⚠️ 安全建议**: `amap_api_key` 应迁移至环境变量或 `.env` 文件。

---

### 2. 文档加载 — `src/service/document_loader.py`

**职责**: 将 `data/disease/` 下的原始文本文件分块为语义完整的 Document 列表。

**流程**:
```
扫描目录(*.txt/*.md/*.mdx)
  → 逐文件读取(UTF-8)
  → 构建 Document(保留 source/file_path/file_type 元数据)
  → RecursiveCharacterTextSplitter
      separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
  → 每块附加 chunk_index / total_chunks / chunk_size
```

**分块策略说明**:
- 按中文标点符号逐级递归，保证句子不会被截断在词中间
- 500 字符块大小在"上下文长度"和"粒度"之间取得平衡
- 100 字符重叠保证跨块语义不丢失

---

### 3. 向量存储 — `src/service/vector_store.py`

**职责**: ChromaDB 创建、加载、检索。

**关键设计**:

```
VectorStoreManager
  ├── _create_embedding_model()
  │     → OllamaEmbeddings(nomic-embed-text, 768d)
  │
  ├── @property vector_store (懒加载)
  │     → load_vector_store() 从磁盘重建
  │     → 若存在则返回 Chroma 实例，否则返回 None
  │
  ├── create_vector_store(documents)
  │     → Chroma.from_documents() → 持久化到 data/chroma_db/
  │     → collection_name = "medical_docs"
  │
  └── similarity_search(query, k=6, score_threshold=0.3)
        → similarity_search_with_relevance_scores()
        → score_threshold 过滤低相关结果
```

**为什么选 ChromaDB 而非 FAISS**:
| 维度 | FAISS (旧) | ChromaDB (新) |
|------|-----------|--------------|
| 持久化 | 手动 pickle | 自动持久化目录 |
| 集合管理 | 单文件 | 多集合隔离 |
| 元数据过滤 | 需自行实现 | 内建支持 |
| 生产部署 | 单机内存 | 可选 HTTP 模式 |

---

### 4. 对话记忆 — `src/service/memory_store.py`

**职责**: 基于 Redis 的多轮会话记忆存储。

**数据结构**:
```
key:   chat:{session_id}:messages       # Redis List
value: {"role":"user","content":"...","timestamp":"2026-05-28T..."}
TTL:   86400s (每次 rpush 时刷新)
```

**核心方法**:

| 方法 | 功能 | 时间复杂度 |
|------|------|-----------|
| `add_message(session_id, role, content)` | 追加到列表末尾 | O(1) |
| `get_recent_messages(session_id, n=10)` | `lrange(key, -n, -1)` 取最近 N 条 | O(1)+O(n) |
| `get_history_text(session_id, n=10)` | 拼接为 prompt 可读文本（含角色标签） | O(n) |
| `clear_session(session_id)` | `delete(key)` 清除整轮历史 | O(1) |

**故障回退**: Redis 不可用时 `_connect()` 将 `_client` 置为 `None`，
所有读写方法静默返回空结果。主流程不受影响。

**为什么用 Redis List 而非 FAISS**:
- 对话记忆不需要语义检索（不需要按内容搜索历史）
- List 结构天然按时序追加，正好对应对话流
- TTL 自动过期，无需手动清理
- 相比原 HybridChatMemory（FAISS+内存），外部化存储支持多进程共享

---

### 5. Agent — `src/service/agent.py`

**职责**: LangGraph StateGraph 驱动的多节点 Agent，封装 RAG 全流程。

**节点定义**:

```
classify_intent
  → _classify_intent(state):
      messages[-1] → IntentClassifier → intent

conditional edge (route_by_intent)
  ├── medical_inquiry / unknown
  │     → retrieve_docs
  │         → _retrieve_docs(state):
  │             messages[-1] → ChromaDB.similarity_search()
  │           generate_answer
  │         → _generate_answer(state):
  │             {history} + {context} + {input} → OllamaLLM.invoke()
  │           save_memory
  │         → _save_memory(state):
  │             user_input + answer → MemoryStore.add_message()
  │
  ├── chat_general + is_weather_query
  │     → weather_query → _weather_query(state):
  │         get_weather_response() → 与 history 合并后 LLM 润色
  │
  └── chat_general / system_query
        → general_chat → _general_chat(state):
            general_prompt.format(history, input) → OllamaLLM.invoke()
```

**状态定义 (AgentState TypedDict)**:
```python
{
    "messages": List[Dict],      # 当前轮消息 [{"role": "user", "content": ...}]
    "session_id": str,           # 会话隔离标识
    "intent": str,               # 分类结果
    "context_docs": List[str],   # ChromaDB 检索到的文档正文
    "answer": str                # LLM 最终回答
}
```

**Prompt 模板**:

| 模板 | 用途 | 输入 |
|------|------|------|
| `self.prompt` | RAG 医疗问答 | `{history}` + `{context}` + `{input}` |
| `self.general_prompt` | 闲聊 / 天气润色 | `{history}` + `{input}` |

两套模板均包含 `{history}`（来自 Redis 的最近 10 轮对话），
确保多轮记忆中提到的信息（姓名、既往症状等）可在后续轮次引用。

**流式推理 (run_stream)**:
`run_stream()` 不依赖 LangGraph 图执行，直接走条件分支 + `OllamaLLM.stream()`，
在 `yield` 层面实现 per-token 推送。按意图分类结果直接分流：

```
意图分类
  │
  ├── medical_inquiry / unknown
  │     → ChromaDB 检索
  │     → self.prompt + history + context
  │     → ollama.stream(formatted_prompt)
  │
  ├── chat_general + 天气
  │     → get_weather_response() → self.general_prompt + history
  │     → ollama.stream()
  │
  ├── chat_general / system_query
  │     → self.general_prompt + history + input
  │     → ollama.stream()
  │
  └── save_memory → add_message(user + assistant)
```

**LangGraph 图执行 (run)**:
`run()` 走完整 LangGraph 图，适合需要图拓扑的复杂场景（未来可加 human-in-the-loop、并行节点等）。

---

### 6. 聊天机器人 — `src/service/chatbot.py`

`MedicalAgent` 的薄包装，保持外部接口稳定：

```python
class MedicalChatbot:
    def get_answer(self, question, session_id) -> dict:
        return {"answer": self.agent.run(question, session_id)}

    def ask_stream(self, question, session_id) -> Generator[dict]:
        yield from self.agent.run_stream(question, session_id)
```

**Yields**（流式接口）:
```python
{"type": "intent", "content": "medical_inquiry"}  # 意图事件（首个）
{"type": "token",  "content": "根"}               # LLM 输出片段
{"type": "token",  "content": "据"}               # ...
{"type": "done"}                                   # 结束信号
```

---

### 7. 意图分类 — `src/service/intent_classifier.py`

**原理**: 用 LLM（qwen2.5:7b）通过结构化 Prompt 输出 JSON。

**Schema**:
```python
{
    "intents": [
        {"name": "medical_inquiry", "description": "医疗相关问题", "keywords": ["病", "症状", "药"]},
        {"name": "chat_general",    "description": "闲聊",         "keywords": ["你好", "名字", "天气"]},
        {"name": "system_query",    "description": "系统问题",     "keywords": ["功能", "系统", "帮助"]}
    ]
}
```

**规则**:
1. 综合分析语义，不只是关键词匹配
2. 医疗+闲聊混合输入 → 优先识别为 `medical_inquiry`
3. 输出 JSON `{"intent": "medical_inquiry"}`，解析失败返回 `"unknown"`

**性能**:
- 每次分类调用 LLM 一次，约 0.5-2s（取决于模型和硬件）
- `unknown` 兜底走 RAG 流程，保证不会遗漏可能的医疗查询

---

### 8. 工具管理 — `src/service/tool_manager.py`

**天气查询流水线**:
```
用户输入"北京今天天气"
  → extract_city_from_text()  —— 正则快速匹配 "北京"
  └─ 失败 → extract_city_by_llm() —— LLM 语义兜底
    → search_weather(city)     —— 高德地图 API
    → general_llm.invoke()     —— 将 API 数据润色为自然语言
```

**通用闲聊**:
```
general_prompt_template.format(query, additional_context)
  → general_llm.invoke() → 自然语言回答
```

---

### 9. API 路由 — `src/api/routers/chat_router.py`

**端点**:

| 端点 | 方法 | 功能 | 返回格式 |
|------|------|------|----------|
| `/api/chat` | POST | 普通问答 | JSON `{intent, answer}` |
| `/api/chat/stream` | POST | SSE 流式问答 | `text/event-stream` |
| `/api/chat/daily_news` | POST | 新闻查询 | JSON `{success, news[], total}` |

**SSE 协议**:
```
data: {"intent":"medical_inquiry"}       ← 意图事件（首个）
data: "根"                               ← 逐 token（JSON 编码字符串）
data: "据"
...
data: [DONE]                             ← 终止信号
```

**流式实现细节**:
- 使用 `StreamingResponse(media_type="text/event-stream")`
- `asyncio.get_event_loop().run_in_executor(None, partial(next, gen))`
  将同步 Generator 转换为异步迭代，避免阻塞事件循环
- 每轮生成后自动写入 Redis（在 `agent.py` 的 `save_memory` 或 `run_stream` 末尾）

---

### 10. CLI 模式 — `src/api/routers/cli_router.py`

**流程**:
```
system_initializer.initialize_system()
  → while True:
      input() → classify → intent → agent.run_stream() / tool_manager
```

与 API 模式共享 `SystemInitializer` 单例和完整的意图路由逻辑。

---

### 11. 系统初始化 — `src/service/system_initializer.py`

**初始化顺序**（严格依赖）：

```
1. VectorStoreManager     ── 无依赖
2. MemoryStore            ── 无依赖
3. IntentClassifier       ── 仅依赖 Ollama 服务
4. ToolManager            ── 无依赖
5. MedicalAgent           ── 依赖 1/2/3/4
6. MedicalChatbot         ── 依赖 5
```

**触发方式**:
- API 模式: FastAPI `@app.on_event("startup")` 自动调用
- CLI 模式: `run_cli()` 内首次访问组件时按需触发

---

### 12. 前端 — `src/static/index.html`

单页面 HTML，核心机制：

```
用户输入 → fetch() POST /api/chat/stream
  → response.body.getReader() 读取 SSE 流
  → 按 \n\n 拆分事件 → data: 前缀解析
  → intent 事件 → 显示意图标签
  → token 事件 → 追加到 AI 气泡
  → [DONE] → 移除光标动画
```

**特点**:
- 纯原生 JS，无框架依赖
- `ReadableStream` 实现真正的流式渲染
- localStorage 持久化 `session_id`，支持多轮对话
- 支持中断 (AbortController) 和清空对话

---

## 数据流全览

```
                    ┌──────────────┐
                    │  启动/初始化  │
                    │  ensure_data_dirs()│
                    │  SystemInitializer│
                    └──────┬───────┘
                           ▼
              ┌────────────────────────┐
              │  离线索引阶段           │
              │  (首次或 data/disease   │
              │   有更新时)             │
              │                        │
              │  DocumentLoader        │
              │  → RecursiveCharacter  │
              │    TextSplitter        │
              │  → Chroma.from_docs()  │
              │  → persist to disk     │
              └────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
   ┌──────────────────┐     ┌────────────────────┐
   │  在线推理阶段     │     │  Redis 存储        │
   │                  │     │  用户 → 助手对话    │
   │  User Input      │     │  TTL 24h           │
   │  → Classify      │     └────────────────────┘
   │  → Route         │
   │  → Retrieve(Chro)│
   │  → Generate(LLM) │
   │  → Save Memory   │
   │  → Response      │
   └──────────────────┘
```

## 外部依赖

| 依赖 | 用途 | 许可证 |
|------|------|--------|
| `fastapi` | Web 框架 | MIT |
| `uvicorn` | ASGI 服务器 | BSD-3 |
| `langchain-core` | LLM 抽象层 | MIT |
| `langchain-text-splitters` | 文档分块 | MIT |
| `langchain-ollama` | Ollama 集成 | MIT |
| `langchain-chroma` | ChromaDB LangChain 接口 | MIT |
| `chromadb` | 向量数据库 | Apache-2.0 |
| `redis` | Python Redis 客户端 | MIT |
| `langgraph` | Agent 状态图框架 | MIT |
| `requests` / `httpx` | HTTP 客户端 | Apache-2.0 |
| `pydantic` | 数据校验 | MIT |

**外部服务**:
| 服务 | 用途 | 依赖 |
|------|------|------|
| Ollama (本地) | LLM 推理 & 文本嵌入 | 必须 |
| Redis (本地) | 对话记忆缓存 | 必须 |
| 高德天气 API | 实时天气数据 | 可选（API Key） |
| 聚合数据 API | 新闻资讯 | 可选（API Key） |

## 性能考量

### 延迟
- **意图分类**: ~0.5-2s（每次调用 LLM）
- **向量检索**: ~50-200ms（ChromaDB 内存模式）
- **LLM 生成**: ~3-10s（7B 模型，取决于硬件）
- **端到端**: ~4-12s（SSE 逐步呈现，首 token 约 2-3s）

### 优化方向
1. **缓存频繁意图分类结果**（同一轮会话中意图通常不变）
2. **提前检索**（在 LLM 生成的同时预取下一轮相关文档）
3. **批处理嵌入**（减少 Ollama 嵌入调用的网络开销）
4. **LLM 量化**（qwen2.5:7b → qwen2.5:7b-Q4_K_M 降低显存占用）
