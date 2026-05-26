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
4. `FAISS.from_documents()` 构建索引并持久化到 `data/faiss_index/`

### 2. 检索（Retrieval）

**原理**: 用户问题 → 向量化 → 向量库相似度搜索 → 返回 Top-K 相关文档

**本项目实现** (`vector_store.py`):
1. `similarity_search_with_relevance_scores(query, k=6)` 召回候选
2. `score_threshold=0.3` 过滤低相关结果
3. `MedicalChatbot` 初始化时通过 `vector_store.as_retriever()` 创建检索器

### 3. 生成（Generation）

**原理**: 系统指令 + 检索结果 + 对话历史 + 用户问题 → LLM → 回答

**本项目实现** (`chatbot.py`):
1. `PromptTemplate` 组装 `{context}`(检索文档)、`{relevant_history}`(对话历史)、`{input}`(问题)
2. LCEL 管道: `{ dict } | prompt | llm` 端到端调用
3. `HybridChatMemory` 提供两级记忆: 内存(最近10轮) + FAISS(历史语义检索)

---

## 意图路由

```
用户输入 → IntentClassifier.classify()
  ├─ medical_inquiry → MedicalChatbot.get_answer()  ← RAG 主流程
  ├─ chat_general    → ToolManager (天气/闲聊)
  ├─ system_query    → ToolManager (通用 LLM)
  └─ unknown         → 兜底走 RAG
```

分类基于 LLM（同 RAG 用同一个 qwen2.5:7b），通过结构化 Prompt 输出 JSON。

---

## 关键优化点

### 检索质量
- 分块策略：按中文标点符号递归分割，语义单元更完整
- 相关性阈值：score_threshold=0.3 过滤噪声
- 元数据过滤：按 session_id 限定对话历史搜索范围

### 记忆管理
- 内存缓存：最近 10 轮零延迟读取
- 溢出迁移：超出上限的记录写入 FAISS，兼顾性能与持久化
- 两路合并：`get_relevant_history()` 合并内存最近记录 + FAISS 语义相关记录

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
