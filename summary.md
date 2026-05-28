# RAG 全流程梳理（基于本项目代码）

RAG（Retrieval-Augmented Generation）核心思想：**让 LLM 在回答前先去检索外部知识库**，
将检索到的相关资料注入 prompt，保证回答的准确性和时效性。

---

## 三阶段

### 1. 数据准备（Ingestion）

```
非结构化文本(.txt/.md)
  → DocumentLoader.load_documents()
      — 扫描 data/disease/ 目录
      — 每文件构建 LangChain Document(保留 metadata)
  → 分块策略（二选一）:
      • 语义分块 (默认): SemanticChunker — 预分句 → 嵌入 → 余弦距离断点 → 合并语义块
      • 固定回退: RecursiveCharacterTextSplitter(chunk_size=500, overlap=100)
  → VectorStoreManager.create_vector_store()
      — OllamaEmbeddings(nomic-embed-text) → 768 维向量
      — Chroma.from_documents() → 持久化到 data/chroma_db/
      — jieba 分词 → BM25 索引构建
```

**文件**: `document_loader.py:DocumentLoader` / `SemanticChunker` → `vector_store.py`

### 2. 检索（Retrieval）

```
用户问题"头痛怎么办"
  → vector_store.hybrid_search(query, k=6)
      Step 1 — 稠密:  ChromaDB 余弦相似度召回 Top-20
      Step 2 — 稀疏:  BM25.jieba 分词 → BM25Okapi 召回 Top-20
      Step 3 — 融合:  RRF score = Σ 1/(60 + rank)，合并去重 → 20~40 候选
      Step 4 — 重排:  Cross-Encoder (MiniLM) 对 (query, doc) 打分 → 降序
      Step 5 — 返回:  Top-6 最终结果
  → 返回 List[Document] → extract page_content → context_text
```

**文件**: `vector_store.py:hybrid_search()` → `agent.py:_retrieve_docs()`

### 3. 生成（Generation）

```
PromptTemplate(
    history = MemoryStore.get_history_text(session_id)   # Redis 最近 10 轮
    context = hybrid_search 结果                           # 混合检索+重排序后文档
    input   = 用户当前问题
)
  → OllamaLLM.invoke() / .stream()
  → 回答文本
```

**文件**: `agent.py:_generate_answer()` / `run_stream()`

---

## Agent 架构（LangGraph StateGraph）

```
classify_intent (日志/UI)
  │
  └── call_model (ChatOllama.bind_tools)
        │
        ├── LLM 调用工具 → tool_node
        │     ├─ search_medical_knowledge  (ChromaDB 混合检索)
        │     ├─ get_weather               (高德天气 API)
        │     └─ chat_general              (ChatOllama 直接回答)
        │     → 工具结果 → call_model (循环)
        │
        └── LLM 直接回答 → human_review (interrupt_after)
              → save_memory (Redis 写入) → END
```

**路径选择**:
- **流式路径** (`run_stream`): 条件分支 + `ChatOllama.stream()`，生成后 yield review 事件
- **同步路径** (`run`): 完整 LangGraph 图 + `ChatOllama.invoke()` + 工具调用 + 人机协同

**多轮记忆**: 所有分支生成 prompt 前都从 Redis 拉取最近 10 轮对话历史，
确保用户在第一轮提到的信息（姓名、既往症状）可被后续轮次引用。

---

## 关键参数

| 参数 | 值 | 用途 |
|------|-----|------|
| `use_semantic_chunking` | `True` | 启用语义分块 |
| `use_hybrid_search` | `True` | 启用混合检索 |
| `use_reranking` | `True` | 启用 Cross-Encoder 重排序 |
| `hybrid_prefetch_k` | 20 | 稠密/稀疏各预取数 |
| `rrf_k` | 60 | RRF 融合常数 |
| `rerank_top_k` | 6 | 重排序后文档数 |
| `retrieval_k` | 6 | 最终返回文档数 |
| `chunk_size` / `chunk_overlap` | 500 / 100 | 固定分块参数（回退） |
| `semantic_chunk_min/max_size` | 200 / 800 | 语义块尺寸范围 |
| `score_threshold` | 0.3 | 稠密检索过滤阈值 |
| `temperature` | 0.1 | LLM 生成温度 |
| `max_history_turns` | 10 | 记忆保留轮数 |
| `redis_ttl` | 86400s (24h) | 记忆过期时间 |

---

## 运行方式

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
redis-server                     # 启动 Redis

python src/main.py --api         # Web 界面 (http://localhost:8000)
python src/main.py               # 命令行交互
```

---

## RAG vs 纯 LLM

| 维度 | 纯 LLM | RAG（本项目） |
|------|--------|-------------|
| 知识时效性 | 训练数据截止日期 | 可实时更新知识库 |
| 幻觉控制 | 高（编造事实） | 低（检索结果约束） |
| 领域专业度 | 通用知识 | 可注入特定医学文档 |
| 可解释性 | 黑盒 | 可追溯引用来源 |
| 更新成本 | 需要重新训练 | 只需更新文档目录 |
