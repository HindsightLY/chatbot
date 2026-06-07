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
  → (可选) HyDE 查询转换:
      LLM 生成假设医学回答 → 替代原始查询进行稠密检索
      Step 0 — 假设文档嵌入（仅影响 Step 1 稠密检索）
  → vector_store.hybrid_search(query/假设文档, k=6)
      Step 1 — 稠密:  ChromaDB 余弦相似度召回 Top-20
      Step 2 — 稀疏:  BM25.jieba 分词 → BM25Okapi 召回 Top-20 (始终用原始查询)
      Step 3 — 融合:  RRF score = Σ 1/(60 + rank)，合并去重 → 20~40 候选
      Step 4 — 重排:  Cross-Encoder (MiniLM) 对 (query, doc) 打分 → 降序
      Step 5 — 返回:  Top-6 最终结果
  → 返回 List[Document] → extract page_content → context_text
```

**文件**: `vector_store.py:hybrid_search()` → `agent.py:run_stream()`（medical_inquiry 分支）

### 3. 生成（Generation）

```
history = MemoryStore.get_history_text(session_id)   # Redis 最近 10 轮
  + (可选) MemorySummarizer.get_relevant_summaries() # 语义检索相关历史摘要

PromptTemplate(
    history = enhanced_history_box_above
    context = hybrid_search 结果                       # 混合检索+重排序后文档
    input   = 用户当前问题
)
  → ChatOllama.invoke() / .stream()
  → 回答文本
```

**文件**: `agent.py:run_stream()` → `memory_summarizer.py:get_enhanced_history()`

---

## Agent 架构

### 流式路径 (`run_stream`) — 不走 LangGraph 图，直接条件分支 + `ChatOllama.stream()`

```
意图分类 (IntentClassifier)
  │
  ├── medical_inquiry / unknown
  │     → (可选) HyDE → ChromaDB 混合检索
  │     → history = enhanced_history（含语义检索到的摘要）
  │     → self.prompt + history + context → LLM.stream()
  │
  ├── chat_general + 天气
  │     → tool_manager.get_weather_response()
  │     → history = enhanced_history
  │     → self.general_prompt + history → LLM.stream()
  │
  ├── chat_general / system_query
  │     → history = enhanced_history
  │     → self.general_prompt + history → LLM.stream()
  │
  └── 其他（兜底）
        → self.prompt + "未找到相关医学资料" + history → LLM.stream()

  ↓ 所有分支结束后自动执行
  自动保存记忆 → add_message(user + assistant)
    → check_and_summarize()    (分层记忆摘要，≥20 轮时触发)
  → yield {"type": "done"}
```

**产出事件**（仅三个类型）:
| 事件 | 格式 | 说明 |
|------|------|------|
| `intent` | `{"type":"intent","content":"medical_inquiry"}` | 意图分类结果（首个事件） |
| `token` | `{"type":"token","content":"头痛"}` | LLM 逐 token 输出 |
| `done` | `{"type":"done"}` | 生成完成信号 |

> 注意：流式路径**不产生 `review` 事件**，不经过人工审核。
> 记忆在 LLM 生成完毕后自动持久化到 Redis，无需前端额外确认。

### 同步路径 (`run`) — 完整 LangGraph StateGraph + 工具调用 + 人机协同

```
classify_intent (日志/UI)
  │
  └── call_model (ChatOllama.bind_tools)
        │ ← history = enhanced_history（含语义检索到的摘要）
        │
        ├── LLM 调用工具 → tool_node
        │     ├─ search_medical_knowledge  (HyDE 转换 → ChromaDB 混合检索)
        │     ├─ get_weather               (高德天气 API)
        │     └─ chat_general              (ChatOllama 直接回答)
        │     → 工具结果 → call_model (循环)
        │
        └── LLM 直接回答 → human_review (interrupt_after)
              → save_memory (Redis 写入) → END
              → check_and_summarize()    (分层记忆摘要)
```

**路径选择**:
- **流式路径** (`run_stream`): 不走 LangGraph 图，直接条件分支 + `ChatOllama.stream()`，用于 SSE 流式场景
- **同步路径** (`run`): 走完整 LangGraph 图，支持工具调用 + `human_review` 人机协同中断，用于同步 API 或 CLI 非流式场景

**`human_review` 自动批准**: 同步图中 LLM 无工具调用时会在 `human_review` 节点暂停，
`run()` 方法检测到中断后自动调用 `graph.invoke(None)` 恢复执行（自动批准），
无需外部人工确认。

**分层记忆**: 所有分支使用 `MemorySummarizer.get_enhanced_history()` 替代裸 `get_history_text()`，
在最近 10 轮对话基础上附加语义检索到的历史摘要，实现长期记忆。

---

## 意图分类（三引擎）

| 引擎 | 延迟 | 依赖 | 说明 |
|------|------|------|------|
| BERT (默认) | ~50ms | `sentence-transformers` | 多头匹配，精度 ~85-90% |
| LLM (回退) | ~2s | Ollama | JSON Prompt，精度 ~95% |
| 关键词 (兜底) | <1ms | 无 | 权重打分，保底分类 |

---

## 关键参数

| 参数 | 值 | 用途 |
|------|-----|------|
| `use_bert_classifier` | `True` | 启用 BERT 意图分类 |
| `bert_model_name` | `paraphrase-multilingual-MiniLM-L12-v2` | BERT 分类模型 |
| `use_hyde` | `True` | 启用 HyDE 查询转换 |
| `use_hierarchical_memory` | `True` | 启用分层记忆 |
| `memory_summary_turns` | `20` | 多少轮对话后触发摘要 |
| `memory_retrieval_k` | `3` | 检索相关摘要数量 |
| `use_semantic_chunking` | `True` | 启用语义分块 |
| `use_hybrid_search` | `True` | 启用混合检索 |
| `use_reranking` | `True` | 启用 Cross-Encoder 重排序 |
| `hybrid_prefetch_k` | 20 | 稠密/稀疏各预取数 |
| `rrf_k` | 60 | RRF 融合常数 |
| `rerank_top_k` | 6 | 重排序后文档数 |
| `retrieval_k` | 6 | 最终返回文档数 |
| `semantic_chunk_min/max_size` | 200 / 800 | 语义块尺寸范围 |
| `score_threshold` | 0.3 | 稠密检索过滤阈值 |
| `temperature` | 0.1 | LLM 生成温度 |
| `max_history_turns` | 10 | 记忆保留轮数 |
| `redis_ttl` | 86400s (24h) | 记忆过期时间 |

---

## 运行方式

### 本地运行

```bash
pip install -r requirements.txt
pip install sentence-transformers   # 可选，启用 BERT 分类器
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
redis-server                        # 启动 Redis

python -m src.main --api            # Web 界面 (http://localhost:8000)
python -m src.main                  # 命令行交互
```

### Docker 运行

```bash
docker compose up -d
docker exec medical_chatbot_ollama ollama pull qwen2.5:7b
docker exec medical_chatbot_ollama ollama pull nomic-embed-text
# 访问 http://localhost:8000
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
