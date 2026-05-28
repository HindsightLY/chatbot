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
  → RecursiveCharacterTextSplitter
      — chunk_size=500, chunk_overlap=100
      — separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
  → VectorStoreManager.create_vector_store()
      — OllamaEmbeddings(nomic-embed-text) → 768 维向量
      — Chroma.from_documents() → 持久化到 data/chroma_db/
```

**文件**: `document_loader.py` → `vector_store.py`

### 2. 检索（Retrieval）

```
用户问题"头痛怎么办"
  → vector_store.similarity_search(query, k=6, score_threshold=0.3)
      — ChromaDB 余弦相似度召回 Top-6
      — score_threshold=0.3 过滤低相关文档
  → 返回 List[Document] → extract page_content → context_text
```

**文件**: `vector_store.py:similarity_search()` → `agent.py:_retrieve_docs()`

### 3. 生成（Generation）

```
PromptTemplate(
    history = MemoryStore.get_history_text(session_id)   # Redis 最近 10 轮
    context = similarity_search 结果                       # ChromaDB 检索文档
    input   = 用户当前问题
)
  → OllamaLLM.invoke() / .stream()
  → 回答文本
```

**文件**: `agent.py:_generate_answer()` / `run_stream()`

---

## Agent 架构（LangGraph StateGraph）

```
入口: classify_intent ──── 条件路由
  │
  ├── medical_inquiry / unknown → retrieve_docs (ChromaDB 检索)
  │     → generate_answer (PromptTemplate + OllamaLLM)
  │     → save_memory (Redis 写入)
  │
  ├── chat_general + 天气 → weather_query (高德 API + LLM 润色)
  │     → save_memory
  │
  └── chat_general / system_query → general_chat (general_prompt + LLM)
        → save_memory
```

**路径选择**:
- **流式路径** (`run_stream`): 直接走 Python 条件分支 + `OllamaLLM.stream()`，不经过图
- **同步路径** (`run`): 走完整 LangGraph 图 + `OllamaLLM.invoke()`

**多轮记忆**: 所有分支生成 prompt 前都从 Redis 拉取最近 10 轮对话历史，
确保用户在第一轮提到的信息（姓名、既往症状）可被后续轮次引用。

---

## 关键参数

| 参数 | 值 | 用途 |
|------|-----|------|
| `chunk_size` | 500 | 文档分块字符数 |
| `chunk_overlap` | 100 | 块间重叠字符数 |
| `retrieval_k` | 6 | 检索文档数 |
| `score_threshold` | 0.3 | 相似度过滤阈值 |
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
