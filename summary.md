# RAG 全流程梳理（基于本项目代码）

RAG 的核心思想：**让 LLM 在回答前先去查阅相关的外部资料**，保证回答的准确性和时效性。

---

## 三阶段概览

### 1. 数据准备（Ingestion）

**原理**: 非结构化文本 → 分块 → 向量化 → 存入向量库

**本项目实现** (`document_loader.py` + `vector_store.py`):
1. `DocumentLoader.load_and_split_documents()` 读取 `data/disease/` 下 `.txt/.md` 文件
2. `RecursiveCharacterTextSplitter` 递归分块（chunk_size=500, chunk_overlap=100）
3. `VectorStoreManager.create_vector_store()` 用 `OllamaEmbeddings(nomic-embed-text)` 转为 768 维向量
4. `Chroma.from_documents()` 构建集合并持久化到 `data/chroma_db/`

### 2. 检索（Retrieval）

**原理**: 用户问题 → 向量化 → 向量库相似度搜索 → 返回 Top-K 相关文档

**本项目实现** (`vector_store.py` + `agent.py`):
1. `similarity_search_with_relevance_scores(query, k=6)` 召回候选
2. `score_threshold=0.3` 过滤低相关结果
3. `MedicalAgent._retrieve_docs()` 节点在 LangGraph 中调用检索

### 3. 生成（Generation）

**原理**: 系统指令 + 检索结果 + 对话历史 + 用户问题 → LLM → 回答

**本项目实现** (`agent.py`):
1. `PromptTemplate` 组装 `{context}`(检索文档)、`{history}`(Redis 对话历史)、`{input}`(问题)
2. `MedicalAgent._generate_answer()` 节点调用 `OllamaLLM` 生成
3. `MemoryStore`(Redis) 提供最近 10 轮对话历史，自动过期清理

---

## LangGraph Agent 流转

```
用户输入 → classify_intent
  │
  ├─ medical_inquiry → retrieve_docs (ChromaDB)
  │     → generate_answer (LLM + context + history)
  │     → save_memory (Redis)
  │
  ├─ chat_general + 天气 → weather_query (高德 API)
  │     → save_memory (Redis)
  │
  ├─ chat_general / system_query → general_chat (LLM)
  │     → save_memory (Redis)
  │
  └─ unknown → retrieve_docs (兜底走 RAG)
```

分类基于 LLM（同 RAG 用同一个 qwen2.5:7b），通过结构化 Prompt 输出 JSON。

---

## 关键优化点

### 检索质量
- 分块策略：按中文标点符号递归分割，语义单元更完整
- 相关性阈值：score_threshold=0.3 过滤噪声
- ChromaDB 持久化：集合按 collection_name 组织，支持多集合隔离

### 记忆管理
- Redis 存储：List 结构按时间序追加，TTL 24h 自动过期
- 懒加载连接：首次使用时连接 Redis，失败时静默降级
- 会话隔离：key 按 `chat:{session_id}:messages` 区分

### 天气提取
- 先正则快速匹配常见模式（"北京天气"）
- 正则失败后用 LLM 语义兜底
- 输出后过滤 "未找到"、"没有" 等无效结果

---

## 运行方式

```bash
python src/main.py           # CLI 交互
python src/main.py --api     # FastAPI 服务
```

需确保 Ollama 和 Redis 服务均已启动。
