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
| 检索 | `retrieval_score_threshold` | `0.3` | 相关性过滤阈值 |
| 检索 | `use_hybrid_search` | `True` | 启用混合检索 (BM25+Dense+RRF) |
| 检索 | `hybrid_prefetch_k` | `20` | 混合检索预取数 |
| 检索 | `rrf_k` | `60` | RRF 融合常数 |
| 检索 | `use_reranking` | `True` | 启用 Cross-Encoder 重排序 |
| 检索 | `rerank_top_k` | `6` | 重排序后取 Top-K |
| 分块 | `chunk_size` / `chunk_overlap` | `500` / `100` | 固定分块参数（回退） |
| 分块 | `use_semantic_chunking` | `True` | 启用语义分块 |
| 分块 | `semantic_chunk_min_size` | `200` | 语义块最小字符数 |
| 分块 | `semantic_chunk_max_size` | `800` | 语义块最大字符数 |
| Redis | `redis_host` / `port` / `db` | `127.0.0.1:6379/0` | — |
| Redis | `redis_ttl` | `86400` | 24h 自动过期 |
| Chroma | `chroma_collection_name` | `medical_docs` | 集合名 |
| 高德 | `amap_api_key` | 内嵌 | 天气 API |
| 新闻 | `valid_news_types` | 10 种 | 聚合数据 API |

> **⚠️ 安全建议**: `amap_api_key` 应迁移至环境变量或 `.env` 文件。

---

### 2. 文档加载 — `src/service/document_loader.py`

**职责**: 将 `data/disease/` 下的原始文本文件分块为语义完整的 Document 列表。

**分块策略（二选一）**:

#### 语义分块（默认，推荐）

```
文本 → 预分句 (RecursiveCharacterTextSplitter, chunk_size=50)
  → OllamaEmbeddings 逐句嵌入
  → 计算相邻句余弦距离
  → 在距离 > 百分位阈值处断开
  → 合并为语义块 (min=200, max=800)
```

`SemanticChunker` 类实现:
1. 先用极小单元(50字符)将文本拆为句子
2. 用 `nomic-embed-text` 对每个句子编码为 768 维向量
3. 计算相邻句之间的余弦距离矩阵
4. 取距离的第 80 百分位作为断点阈值
5. 低于阈值的相邻句合并为同一语义块
6. 二次合并保证每块在 [200, 800] 字符范围内

**优势**: 同一话题的句子被自然合并，话题切换处自动分段。

#### 固定分块（回退）

```
RecursiveCharacterTextSplitter
  separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
  chunk_size=500, chunk_overlap=100
```

当嵌入模型不可用时自动降级。

---

### 3. 向量存储 — `src/service/vector_store.py`

**职责**: ChromaDB 创建、加载、混合检索、重排序。

**检索流水线**:

```
用户查询
  │
  ├─ Step 1: 稠密检索 (ChromaDB)
  │     similarity_search_with_relevance_scores(query, k=20)
  │     → 召回 Top-20 向量近邻
  │
  ├─ Step 2: 稀疏检索 (BM25)
  │     jieba 分词 → BM25Okapi.get_scores()
  │     → 召回 Top-20 关键词匹配
  │
  ├─ Step 3: RRF 融合
  │     score = Σ 1/(rrf_k + rank)
  │     → 合并去重得 20~40 个候选
  │
  ├─ Step 4: Cross-Encoder 重排序（可选）
  │     cross-encoder/ms-marco-MiniLM-L-6-v2
  │     → 对 (query, doc) 对逐一打分
  │     → 按相关性得分降序排列
  │
  └─ Step 5: 返回 Top-K (k=6)
```

**关键设计**:

```
VectorStoreManager
  ├── _create_embedding_model()
  │     → OllamaEmbeddings(nomic-embed-text, 768d)
  │
  ├── @property vector_store (懒加载)
  │     → load_vector_store() 从磁盘重建
  │     → 成功则自动构建 BM25 索引
  │
  ├── create_vector_store(documents)
  │     → Chroma.from_documents() → 持久化到 data/chroma_db/
  │     → 自动构建 BM25 索引
  │
  ├── similarity_search(query, k=6, score_threshold=0.3)
  │     → 纯稠密向量检索（兼容旧接口）
  │
  ├── hybrid_search(query, k=6)
  │     → 混合检索 + 可选重排序（主入口）
  │
  └── @property reranker
        → 懒加载 Cross-Encoder 模型
        → 缺依赖时静默降级
```

**BM25 索引**: 使用 `jieba` 中文分词 + `rank_bm25` 库，在 ChromaDB 创建/加载时自动构建，
所有文档的文本内容被分词后存入倒排索引。

**Cross-Encoder 重排序**: 使用 `sentence-transformers` 库加载
`cross-encoder/ms-marco-MiniLM-L-6-v2` 模型。模型首次使用时自动下载。
若 `sentence-transformers` 未安装，跳过重排序步骤（不影响主流程）。

**为什么选 ChromaDB 而非 FAISS**:
| 维度 | FAISS (旧) | ChromaDB (新) |
|------|-----------|--------------|
| 持久化 | 手动 pickle | 自动持久化目录 |
| 集合管理 | 单文件 | 多集合隔离 |
| 元数据过滤 | 需自行实现 | 内建支持 |
| 混合检索 | ❌ 需自行实现 BM25 | ✅ 可集成 rank-bm25 |
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

**职责**: LangGraph StateGraph 驱动的多节点 Agent，封装 RAG + 工具调用 + 人机协同全流程。

**节点定义**:

```
classify_intent (仅日志/UI展示)
  → _classify_intent(state):
      messages[-1] → IntentClassifier → intent (并存 state 供前端)

call_model (bind_tools — LLM 自主决策)
  → _call_model(state):
      ChatOllama.bind_tools([search_medical_knowledge, get_weather, chat_general])
      LLM 自主决定:
        ├─ 医疗问题 → 调用 search_medical_knowledge
        ├─ 天气问题 → 调用 get_weather
        └─ 其他     → 直接回答 (chat_general)

conditional edge (should_continue)
  ├── 有 tool_calls → tool_node
  │     → ToolNode 执行工具 → 结果回填到 messages
  │     → 返回 call_model (循环，LLM 用工具结果生成最终答案)
  │
  └── 无 tool_calls → human_review (interrupt_after)
        → 暂停等待外部确认或自动批准

human_review (人机协同)
  → _human_review(state):
      interrupt_after 暂停，等待 resume() 或 update_state()
      外部可通过 POST /api/chat/review 或 CLI 输入确认

save_memory (记忆持久化)
  → _save_memory(state):
      user_input + answer → MemoryStore.add_message()
```

**状态定义 (AgentState TypedDict)**:
```python
{
    "messages": List[Any],       # 消息列表 (HumanMessage / AIMessage)
    "session_id": str,           # 会话隔离标识
    "intent": str,               # 分类结果
    "context_docs": List[str],   # ChromaDB 检索到的文档正文
    "answer": str,               # LLM 最终回答
    "human_approved": bool,      # 审核结果
    "review_skipped": bool       # 是否跳过审核
}
```

**工具定义** (`src/tools/medical_tools.py`):

使用 `@tool` 装饰器定义标准 LangChain 工具：

| 工具 | 函数 | 触发条件 |
|------|------|---------|
| `search_medical_knowledge(query)` | ChromaDB 混合检索 | 医疗相关问题 |
| `get_weather(location)` | 高德天气 API | 天气查询 |
| `chat_general(query)` | ChatOllama 直接回答 | 闲聊/系统查询 |

**Prompt 模板**:

| 模板 | 用途 | 输入 |
|------|------|------|
| `self.prompt` | RAG 医疗问答 | `{history}` + `{context}` + `{input}` |
| `self.general_prompt` | 闲聊 / 天气润色 | `{history}` + `{input}` |

两套模板均包含 `{history}`（来自 Redis 的最近 10 轮对话），
确保多轮记忆中提到的信息（姓名、既往症状等）可在后续轮次引用。

**流式推理 (run_stream)** — 沿用条件分支 + `ChatOllama.stream()`，增加审核事件：

```
意图分类
  │
  ├── medical_inquiry / unknown
  │     → ChromaDB 检索
  │     → self.prompt + history + context
  │     → ChatOllama.stream(formatted_prompt)
  │
  ├── chat_general + 天气
  │     → get_weather_response() → self.general_prompt + history
  │     → ChatOllama.stream()
  │
  ├── chat_general / system_query
  │     → self.general_prompt + history + input
  │     → ChatOllama.stream()
  │
  ├── review event
  │     → yield {"type": "review", "content": full_answer}
  │     → CLI: 等待用户 Y/n 确认
  │     → SSE: 前端可调用 /api/chat/review 记录审核结果
  │
  └── save_memory → add_message(user + assistant)
```

**LangGraph 图执行 (run)** — 完整图拓扑，含工具调用 + 人机协同：

```
classify_intent → call_model → conditional
  ├─ tool_calls → tool_node → call_model (循环)
  └─ 无工具调用 → human_review (interrupt_after) → save_memory → END
```

**编译参数**:
```python
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_after=["human_review"]  # 在 human_review 节点后暂停
)
```

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

### 7. 意图分类 — `src/service/intent_classifier.py` + `bert_classifier.py`

**双引擎分类器**: BERT 优先（~50ms），LLM 兜底（~2s），关键词规则保底。

#### 引擎 1: BERT 分类器 (默认，推荐)

**文件**: `src/service/bert_classifier.py`

**原理**: sentence-transformers 多头匹配

```
用户查询
  → sentence-transformers 编码为 768 维向量
  → 与每个意图的预编码示例句向量计算余弦相似度
  → 取最高分意图（阈值 < 0.45 时返回 unknown）
```

**预定义示例**:

| 意图 | 示例句数 | 代表句 |
|------|---------|--------|
| medical_inquiry | 15 | "感冒了怎么办"、"头疼是什么原因" |
| chat_general | 15 | "你好"、"今天天气怎么样" |
| system_query | 10 | "这个系统有什么用"、"帮助" |

**性能**: ~50ms/次（首次加载模型约 10s）

**依赖**: `sentence-transformers` + `paraphrase-multilingual-MiniLM-L12-v2`

#### 引擎 2: LLM 分类器 (回退)

**文件**: `src/service/intent_classifier.py`

当 `sentence-transformers` 未安装或 BERT 模型加载失败时，自动降级到 LLM。

**原理**: OllamaLLM(qwen2.5:7b) + JSON Schema Prompt

```python
# Schema
{
    "intents": [
        {"name": "medical_inquiry", "description": "医疗相关问题", ...},
        {"name": "chat_general",    "description": "闲聊",         ...},
        {"name": "system_query",    "description": "系统问题",     ...}
    ]
}
```

**性能**: ~2s/次

#### 引擎 3: 关键词规则 (终极兜底)

当 LLM 也失败时（Ollama 不可用），使用关键词权重打分。

#### 三引擎回退链

```
BERT 分类 (~50ms)
  ├─ 成功 → 返回意图
  └─ 失败 → LLM 分类 (~2s)
              ├─ 成功 → 返回意图
              └─ 失败 → 关键词规则 (<1ms) → 返回意图
```

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

## Docker 部署

项目提供 Docker Compose 一键部署脚本。

### 文件结构

```
Dockerfile           — 多阶段构建（builder + runtime）
docker-compose.yml   — 三服务编排（app + redis + ollama）
.dockerignore        — 构建忽略清单
```

### 容器架构

```
┌─────────────────────────────────────────────────────┐
│  docker-compose.yml                                  │
│                                                      │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐ │
│  │   app:8000   │   │  redis   │   │   ollama     │ │
│  │ FastAPI +    │──→│ :6379    │   │ :11434       │ │
│  │ LangGraph +  │   │ 对话记忆  │   │ LLM 推理     │ │
│  │ ChromaDB     │   └──────────┘   │ 文本嵌入     │ │
│  │ BERT 分类    │                  └──────────────┘ │
│  └──────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

### 使用方式

```bash
# 1. 启动所有服务
docker compose up -d

# 2. 拉取 LLM 模型
docker exec medical_chatbot_ollama ollama pull qwen2.5:7b
docker exec medical_chatbot_ollama ollama pull nomic-embed-text

# 3. 查看应用日志
docker compose logs -f app

# 4. 健康检查
curl http://localhost:8000/api/chat/health

# 5. 停止
docker compose down
```

### 关键设计

| 特性 | 实现方式 |
|------|---------|
| 环境变量 | `REDIS_HOST=redis` / `OLLAMA_BASE_URL=http://ollama:11434` |
| 数据持久化 | `./data:/app/data`（ChromaDB + 疾病文档） |
| GPU 加速 | NVIDIA Container Toolkit（docker-compose 中配置 device 保留） |
| 健康检查 | `GET /api/chat/health` + Docker HEALTHCHECK |
| 可选依赖 | `sentence-transformers` 可在容器内 `pip install` |

---

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
| 阶段 | 估算耗时 | 说明 |
|------|---------|------|
| 意图分类 (BERT) | ~50ms | sentence-transformers |
| 意图分类 (LLM 回退) | ~0.5-2s | OllamaLLM 调用 |
| 稠密检索 | ~50-200ms | ChromaDB 内存模式 |
| BM25 检索 | ~10-50ms | 纯 CPU 计算 |
| RRF 融合 | ~1-5ms | 内存计算 |
| Cross-Encoder 重排序 | ~100-500ms | CPU 推理（MiniLM） |
| LLM 生成（首 token） | ~1-3s | 7B 模型 |
| LLM 生成（后续 token） | ~30-60ms/token | 取决于推理硬件 |
| **端到端（SSE）** | **~2-5s 首 token, 4-12s 完成** | |

### 优化方向
1. **缓存频繁意图分类结果**
2. **提前检索**（在 LLM 生成的同时预取下一轮相关文档）
3. **批处理嵌入**（减少 Ollama 嵌入调用的网络开销）
4. **LLM 量化**（qwen2.5:7b → qwen2.5:7b-Q4_K_M 降低显存占用）
5. **重排序模型量化**（MiniLM → ONNX 加速）
