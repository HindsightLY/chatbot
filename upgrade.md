# 技术栈升级路线 — 对比行业方案与演进方向

> 本文档对比当前项目选型与行业主流方案，提出可量化的升级路径。
> 按优先级（P0=关键 / P1=推荐 / P2=可选）排序。
>
> ✅ = 已实现

---

## 1. 检索增强：从基础 RAG 到高级 RAG

### 现状

```
用户问题 → 混合检索 (BM25 + Dense + RRF) → Cross-Encoder 重排序 → LLM 生成
```

### 行业对比

| 技术 | 当前状态 | 行业方案 | 差距 |
|------|---------|---------|------|
| 检索方式 | ✅ BM25 + Dense (RRF 融合) | 混合检索 | ✅ 已实现 |
| 重排序 | ✅ MiniLM Cross-Encoder | Cohere Rerank / BGE-Reranker | ⚠️ 小模型, 可升级 |
| 查询转换 | ✅ HyDE (LLM 生成假设文档) | HyDE / Multi-Query / Query Rewrite | ✅ 已实现 |
| 分块策略 | ✅ 语义分块 (Semantic Chunker) | 语义分块 / 递归 LLM 分块 | ✅ 已实现 |

### 已实现的升级

**✅ P0 — 混合检索 (Hybrid Search)**

```
query → embedding → cosine_sim(doc.vector, query.vector)  ← 稠密 (ChromaDB Top-20)
      → jieba 分词 → BM25Okapi.get_scores()               ← 稀疏 (BM25 Top-20)
      → RRF: score = Σ 1/(rrf_k + rank)                    ← 融合
      → Cross-Encoder 重排序                               ← 重排
      → Top-6 最终结果
```

实现文件: `src/service/vector_store.py` — `hybrid_search()` 方法

**✅ P0 — 语义分块 (Semantic Chunking)**

```
文本 → 预分句 (chunk_size=50)
  → OllamaEmbeddings 逐句嵌入
  → 余弦距离 > 80% 百分位阈值处断开
  → 合并为语义块 (min=200, max=800)
```

实现文件: `src/service/document_loader.py` — `SemanticChunker` 类

**✅ P1 — Cross-Encoder 重排序**

```
检索 Top-20~40 → cross-encoder/ms-marco-MiniLM-L-6-v2 打分 → Top-6
```

```bash
pip install sentence-transformers   # 可选；未安装时静默跳过重排序
```

实现文件: `src/service/vector_store.py` — `reranker` 属性 + `hybrid_search()` 中的重排序步骤

**✅ P1 — HyDE 查询转换**

```
用户查询 → LLM 生成假设医学回答 → 假设文档嵌入 → 替代原始查询进行稠密检索
```

| 阶段 | 传统 (无 HyDE) | HyDE |
|------|---------------|------|
| 查询 | "头痛怎么办" | "患者出现头痛症状，可能原因包括紧张性头痛、偏头痛等..." |
| 嵌入空间 | 短查询 (5-20 字) | 长文档 (100-300 字) |
| 检索匹配 | 可能遗漏专业术语 | 与知识库文档语义更接近 |

**设计要点**:
- 仅用于稠密检索阶段 (ChromaDB 余弦相似度)
- BM25 稀疏检索仍使用原始查询（关键词匹配不受益于 HyDE）
- 图路径 (`call_model`): 工具 `search_medical_knowledge` 内部自动应用 HyDE
- 流式路径 (`run_stream`): medical_inquiry 分支在 `hybrid_search()` 前执行 HyDE
- LLM 生成失败时静默回退原始查询

实现文件: `src/service/hyde_transformer.py` — `HyDEQueryTransformer.transform()`

---

## 2. 向量数据库：从 ChromaDB 到生产级方案

### 现状

ChromaDB 持久化到本地磁盘，单进程模式。

### 行业对比

| 维度 | ChromaDB | Milvus | Qdrant | Weaviate |
|------|----------|--------|--------|----------|
| 部署模式 | 嵌入式/客户端 | 独立服务 | 独立服务 | 独立服务 |
| 分布式 | ❌ | ✅ | ✅ | ✅ |
| 混合检索 | 有限 | ✅ | ✅ | ✅ |
| 多租户 | ❌ | ✅ | ✅ | ✅ |
| 过滤索引 | 有限 | ✅ 标量索引 | ✅ | ✅ |
| 适用场景 | 原型/小规模 | 生产/大规模 | 生产/中等规模 | 生产/通用 |

### 升级建议

**P2 — 评估生产场景**

- 单机 < 100 万文档 → ChromaDB 足够
- 多机 / 高并发 / 需要滚动升级 → Qdrant（部署最简单）
- 需要复杂标量过滤 + 向量混合 → Milvus

---

## 3. 对话记忆：从 List 到分层记忆

### 现状

```
Redis List: chat:{sid}:messages → lrange → 拼接为文本
```

最近 10 轮对话全部拼入 prompt，超长时简单截断。

### 行业对比

| 维度 | 当前方案 | Mem0 | MemGPT/Letta | LangGraph Persistence |
|------|---------|------|-------------|----------------------|
| 短期记忆 | ✅ Redis List | Buffer | Buffer | LangGraph Checkpoint |
| 长期记忆 | ✅ Redis + 摘要 + 语义检索 | 向量化存储 | 递归摘要 | 自定义 |
| 记忆检索 | ✅ 语义搜索 (OllamaEmbeddings) | 语义搜索 | 实体抽取 | ❌ 需自行实现 |
| 记忆合并 | ✅ 摘要 + Ltrim | 主动合并 | 递归压缩 | ❌ |

### 已实现的升级

**✅ P1 — 分层记忆 (Hierarchical Memory)**

```
近期记忆 (Redis List):  最近 10 轮原始对话 → 拼接为 prompt 上下文
历史摘要 (Redis List:summaries):
  当消息数 > memory_summary_turns (默认 20)：
    1. 取最早 20 条消息 → LLM 摘要
    2. 摘要存入 Redis
    3. Ltrim 裁剪已摘要消息
检索 (OllamaEmbeddings):
  当前查询 → 嵌入 → 与所有摘要余弦相似度排序 → Top-K 相关摘要注入 prompt
```

**设计要点**:
- 摘要使用医学领域定制 Prompt，保留症状/诊断/用药等关键信息
- 语义检索使用 `OllamaEmbeddings(nomic-embed-text)` 计算余弦相似度
- 嵌入不可用时回退返回全部摘要的前 K 条（关键词匹配兜底）
- `get_enhanced_history()` 集成到 Agent 所有分支（run + run_stream）
- 图路径的 `_call_model` 和 `_save_memory` 均触发摘要检查

```python
# src/service/memory_summarizer.py — 核心方法
def get_enhanced_history(session_id, query) -> str:
    recent = memory_store.get_history_text(session_id)  # 近期对话
    summaries = retrieve_relevant_summaries(session_id, query)  # 语义检索摘要
    return recent + "\n【历史摘要】" + "\n".join(summaries)

def check_and_summarize(session_id):
    if len(messages) >= memory_summary_turns:
        summary = llm.invoke(SUMMARY_PROMPT.format(conversation=oldest))
        memory_store.store_summary(session_id, summary)
        memory_store.trim(session_id, keep_last=5)
```

实现文件: `src/service/memory_summarizer.py`

---

## 4. Agent 框架：当前 vs 最佳实践

### 现状

Agent 提供两种推理路径:

**流式路径 (`run_stream`)** — 不走 LangGraph 图，直接条件分支 + `ChatOllama.stream()`：
```
意图分类 → 按意图路由（medical_inquiry / chat_general / system_query）
  → 各分支均使用 enhanced_history（分层记忆）
  → LLM.stream() → 逐 token 产出
  → 自动保存记忆到 Redis → 结束
```

**同步路径 (`run`)** — 完整 LangGraph StateGraph：
```
StateGraph → classify_intent → call_model(bind_tools) → _routes_after_model
  ├─ 有 tool_calls → tool_node → call_model (循环)
  └─ 无 tool_calls → human_review (interrupt_after) → save_memory → END
```

关键区别：
| 维度 | 流式路径 | 同步路径 |
|------|---------|---------|
| 图执行 | 不走图，直接条件分支 | 走完整 LangGraph StateGraph |
| 工具调用 | 不支持（由外部路由处理） | 支持 bind_tools + ToolNode |
| 人机协同 | 无（自动保存记忆） | human_review 节点 + interrupt_after |
| 输出 | 逐 token yield | 返回完整回答字符串 |
| 使用场景 | SSE 流式 API | 同步 API / 内部调用 |

### 行业对比

| 功能 | 当前 | LangGraph 完整能力 | AutoGen / CrewAI |
|------|------|-------------------|------------------|
| Human-in-the-Loop | ✅ `interrupt_after`（同步图） | `interrupt_after` ✅ | ✅ |
| 并行节点 | ❌ | `add_node` + fan-out ✅ | ✅ 多 agent |
| 循环/重试 | ✅ `call_model → tool_node → call_model` | `add_conditional_edges` 自环 ✅ | ✅ |
| Tool Calling | ✅ `bind_tools()` + ToolNode + @tool | `bind_tools()` + ToolNode ✅ | ✅ 原生 |
| 多 Agent 协作 | ❌ | `Send()` API / Supervisor ✅ | ✅ 原生 |
| 持久化检查点 | ❌ | PostgreSQL / SQLite 检查点 ✅ | ❌ |

### 已实现的升级

**✅ P1 — 工具调用标准化**

```python
# src/tools/medical_tools.py
from langchain_core.tools import tool

@tool
def search_medical_knowledge(query: str) -> str:
    """搜索医学知识库，获取与疾病、症状、治疗方法相关的医学资料"""
    ...

@tool
def get_weather(location: str) -> str:
    """查询指定城市的实时天气信息"""
    ...

@tool
def chat_general(query: str) -> str:
    """回答用户的一般性问题、闲聊、问候等"""
    ...

# tools 列表：tools = [search_medical_knowledge, get_weather, chat_general]

# agent.py 中使用：
llm_with_tools = ChatOllama(...).bind_tools(tools)
tool_node = ToolNode(tools)
```

**✅ P1 — 人机协同 (Human-in-the-Loop，仅同步图路径)**

```python
# agent.py — 图构建
builder.add_node("human_review", self._human_review)
builder.add_edge("human_review", "save_memory")

# compile 时指定中断节点
graph = builder.compile(checkpointer=MemorySaver(), interrupt_after=["human_review"])
```

**实现细节**:
- `human_review` 节点通过 `interrupt_after` 暂停图执行，等待外部 `resume` 或 `update_state`
- `run()` 方法中检测到中断后默认自动批准（调用 `graph.invoke(None)` 恢复）
- CLI 模式中，用户可输入 Y/n 确认或拒绝回答
- 流式路径 (`run_stream`) **不经过 human_review**，生成完毕后自动保存记忆到 Redis
- `POST /api/chat/review` 端点已移除（SSE 流式不再产生 `review` 事件）

---

## 5. LLM 策略：从单一模型到模型路由

### 现状

```
意图分类: BERT (sentence-transformers) ~50ms  /  OllamaLLM 兜底
RAG 生成: ChatOllama (Qwen2.5:7B)
闲聊/天气: ChatOllama (Qwen2.5:7B)
```

已分离分类与生成负载。

### 行业方案

| 任务 | 当前 | 推荐替代 | 收益 |
|------|------|---------|------|
| 意图分类 | ✅ BERT 多头匹配 | 微调 BERT 小模型 (< 100M) | 延迟 ~50ms |
| 嵌入 | nomic-embed-text | BGE-M3 / mxbai-embed-large | 多语言更好 |
| RAG 生成 | qwen2.5:7b | Qwen2.5:7B 或 API 模型 | — |
| 闲聊 | qwen2.5:7b | Qwen2.5:0.5B / API 轻量模型 | 降本 |
| 摘要 | qwen2.5:7b | 小模型 / API | 降本 |

### 已实现的升级

**✅ P1 — BERT 意图分类器**

```python
# src/service/bert_classifier.py
# 原理: sentence-transformers 多头匹配

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
query_emb = model.encode(query, normalize_embeddings=True)
# 与每个意图的预编码示例句向量计算余弦相似度
# 取最相似意图作为分类结果
```

| 指标 | LLM 分类 (旧) | BERT 分类 (新) |
|------|-------------|--------------|
| 推理时间 | ~2s | ~50ms |
| 依赖 | Ollama 必选 | sentence-transformers (可选) |
| 准确率 | ~95% (7B) | ~85-90% (零样本) |
| 离线可用 | ❌ | ✅ |
| 可扩展 | 修改 prompt 即可 | 增删示例句即可 |

**三引擎回退链**: BERT → LLM → 关键词规则，保证任意环境下均有分类结果。

**下一步 — 模型路由 (Model Router)**

```python
model_router = {
    "medical_inquiry": "qwen2.5:7b",       # 强模型
    "chat_general":    "qwen2.5:0.5b",     # 弱模型
    "system_query":    "qwen2.5:7b",       # 中等
    "summary":         "qwen2.5:3b",       # 中等
}
```

---

## 6. 部署与基础设施

### 现状

```
Docker Compose 一键部署 (app + redis + ollama)
```

### 已实现的升级

**✅ P0 — Docker 容器化**

```dockerfile
# Dockerfile — 多阶段构建
FROM python:3.11-slim AS builder
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY src/ ./src/
COPY config/ ./config/
```

```yaml
# docker-compose.yml — 三服务编排
services:
  app:
    build: .
    ports: ["8000:8000"]
    environment:
      - REDIS_HOST=redis
      - OLLAMA_BASE_URL=http://ollama:11434
    volumes:
      - ./data:/app/data     # 持久化 ChromaDB
    depends_on: [redis, ollama]

  redis:
    image: redis:7-alpine
    volumes: [redis_data:/data]

  ollama:
    image: ollama/ollama:latest
    volumes: [ollama_data:/root/.ollama]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]   # GPU 加速

volumes:
  redis_data:
  ollama_data:
```

**使用方式**:
```bash
# 首次启动（自动拉取镜像 + 构建应用）
docker compose up -d

# 拉取医学模型（进入 ollama 容器）
docker exec -it medical_chatbot_ollama ollama pull qwen2.5:7b
docker exec -it medical_chatbot_ollama ollama pull nomic-embed-text

# 查看日志
docker compose logs -f app

# 停止
docker compose down
```

**Docker 设计要点**:
| 特性 | 实现 |
|------|------|
| 多阶段构建 | 分离 build/run 阶段，减小镜像体积 |
| 健康检查 | `GET /api/chat/health` + Docker HEALTHCHECK |
| 环境变量 | `REDIS_HOST`、`OLLAMA_BASE_URL` 支持容器互联 |
| 数据持久化 | `data/` 目录挂载（ChromaDB + 文档） |
| GPU 加速 | NVIDIA Container Toolkit（nvidia-docker） |
| 可选依赖 | `sentence-transformers` 由用户自行 pip install |

**P1 — 可观测性 (Observability)**

| 工具 | 用途 | 集成难度 |
|------|------|---------|
| LangSmith | LLM 调用追踪、Token 统计 | 低 (加 API Key) |
| LangFuse | 开源 LLM 可观测 | 低 |
| OpenTelemetry + Jaeger | 全链路追踪 | 中 |
| Prometheus + Grafana | 系统指标监控 | 中 |

```python
# LangSmith 集成（3 行代码）
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "ls_..."
```

**P2 — 模型部署**

```
当前: Ollama (单机，CPU/GPU 均可)
升级:
  ├── vLLM     → 高吞吐 GPU 推理 (PagedAttention)
  ├── TG        → HuggingFace 官方推理服务
  └── Ollama   → 继续使用 (小规模足够)
```

---

## 7. 评估与测试

### 现状

无评估系统。

### 行业方案

**P1 — RAG 评估框架**

```python
# RAGAS: 量化评估 RAG 质量
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

scores = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
# faithfulness: 回答是否忠于检索文档
# answer_relevancy: 回答是否针对问题
# context_precision: 检索文档是否相关
```

**P2 — 自动回归测试**

```python
# 构建黄金测试集
test_cases = [
    {"query": "感冒了吃什么药", "expected_intent": "medical_inquiry",
     "expected_keywords": ["感冒", "休息", "多喝水"]},
    {"query": "北京天气",      "expected_intent": "chat_general",
     "expected_keywords": ["北京", "天气"]},
]

for case in test_cases:
    intent = classifier.classify(case["query"])
    assert intent == case["expected_intent"], f"意图分类失败: {case['query']}"
```

---

## 8. 前端与用户体验

### 现状

单 HTML 文件，原生 JS，SSE 流式渲染。

### 行业方案

**P2 — 前端框架**

| 方案 | 优缺点 |
|------|--------|
| 保持原生 | 零依赖，极致轻量 |
| React + Next.js | 生态丰富，SSR 支持 |
| Vue + Nuxt | 中文文档友好 |
| Gradio | 快速搭建 ML 演示 |

**P2 — 增强特性**

- Markdown 渲染（代码块、表格、链接）
- 引用来源标注（悬停显示检索文档片段）
- 对话分支 / 追问建议
- 语音输入 (Web Speech API)
- WebSocket 替换 SSE（双向通信）

---

## 升级路线图总览

| 优先级 | 项目 | 预估工时 | 收益 |
|--------|------|---------|------|
| **P0** | ✅ 混合检索 (BM25 + Dense + RRF) | ✅ 已完成 | 检索召回率 +10-20% |
| **P0** | ✅ 语义分块 (Semantic Chunking) | ✅ 已完成 | 话题凝聚力提升 |
| **P1** | ✅ Cross-Encoder 重排序 | ✅ 已完成 | 准确率 +5-15% |
| **P1** | ✅ 工具调用标准化 (bind_tools + @tool) | ✅ 已完成 | Agent 灵活性 |
| **P1** | ✅ 人机协同 (interrupt_after，同步图) | ✅ 已完成 | 安全性 / 可控性 |
| **P0** | ✅ Docker 容器化 (Compose 三服务) | ✅ 已完成 | 部署标准化 |
| **P1** | ✅ BERT 分类器 (sentence-transformers) | ✅ 已完成 | 延迟 2s→50ms |
| **P1** | ✅ HyDE 查询转换 | ✅ 已完成 | 召回复盖率提升 |
| **P1** | ✅ 分层记忆 (摘要+语义检索) | ✅ 已完成 | 记忆质量 |
| **P1** | LangSmith 可观测性 | 0.5 天 | 调试效率 |
| **P1** | RAGAS 评估 | 1-2 天 | 量化质量 |
| **P2** | 模型路由 | 1-2 天 | 成本优化 |
| **P2** | WebSocket / 前端增强 | 3-5 天 | 用户体验 |
| **P2** | 生产级向量库 (Qdrant) | 3-5 天 | 扩展性 |
| **P2** | CI/CD + 回归测试 | 2-3 天 | 工程质量 |

---

## 总结

当前项目是一个**功能完善的原型**，已完成九项核心升级（语义分块、混合检索、重排序、
工具调用标准化、人机协同、BERT 分类器、Docker 容器化、HyDE 查询转换、分层记忆），
流式交互顺畅、双模式可用。
通向生产级系统的主要差距在于：

1. **可观测性**: 无 LLM 调用追踪 → 调试效率低
2. **评估**: 无量化指标 → 优化方向不明确
3. **并行节点**: 无 fan-out → 无法并行执行多个工具

建议按 P0 → P1 → P2 的顺序逐步演进，每完成一个阶段进行一次质量评估。
