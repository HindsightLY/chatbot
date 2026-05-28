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
| 查询转换 | ❌ 无 | HyDE / Multi-Query / Query Rewrite | ❌ |
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

### 下一步

**P1 — 查询转换 (Query Transformation)**

```python
# Multi-Query: 用 LLM 生成 N 个同义查询，分别检索后去重合并
def multi_query(query: str) -> List[str]:
    prompt = f"请为以下问题生成 3 个不同角度的同义问句：\n{query}"
    return llm.invoke(prompt).split("\n")

# HyDE: 先生成假设文档，用假设文档检索
def hyde(query: str) -> str:
    return llm.invoke(f"请针对以下问题写一段医学回答：\n{query}")
```

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
| 短期记忆 | Redis List | Buffer | Buffer | LangGraph Checkpoint |
| 长期记忆 | ❌ | 向量化存储 | 递归摘要 | 自定义 |
| 记忆检索 | ❌ | 语义搜索 | 实体抽取 | ❌ 需自行实现 |
| 记忆合并 | ❌ | 主动合并 | 递归压缩 | ❌ |

### 升级建议

**P1 — 分层记忆架构**

```
短期 (Current Session): Redis ZSET / list → 最近 3 轮完整对话
中期 (Last Sessions):    Redis + ChromaDB → 上一会话的摘要向量存储
长期 (User Profile):     ChromaDB → 用户偏好 / 禁忌 / 个人信息的语义检索
```

**P1 — 记忆摘要 (Memory Consolidation)**

当对话轮数超过阈值时，触发 LLM 摘要：

```python
# 每 20 轮对话后压缩历史
if turn_count % 20 == 0:
    summary = llm.invoke(f"请总结以下对话中提到的所有重要信息：\n{history_text}")
    memory_store.save_summary(session_id, summary)
    memory_store.trim_messages(session_id, keep_last=5)  # 只保留最近 5 轮
```

---

## 4. Agent 框架：当前 vs 最佳实践

### 现状

```python
StateGraph → classify_intent → call_model(bind_tools) → conditional
  ├─ 有工具调用 → tool_node → call_model (循环)
  └─ 无工具调用 → human_review → save_memory → END
```

已实现工具调用标准化 + 人机协同。

### 行业对比

| 功能 | 当前 | LangGraph 完整能力 | AutoGen / CrewAI |
|------|------|-------------------|------------------|
| Human-in-the-Loop | ✅ `interrupt_after` | `interrupt_after` ✅ | ✅ |
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

**✅ P1 — 人机协同 (Human-in-the-Loop)**

```python
# agent.py — 图构建
builder.add_node("human_review", self._human_review)
builder.add_edge("human_review", "save_memory")

# compile 时指定中断节点
graph = builder.compile(checkpointer=MemorySaver(), interrupt_after=["human_review"])
```

**实现细节**:
- `human_review` 节点通过 `interrupt_after` 暂停图执行，等待外部 `resume` 或 `update_state`
- CLI 模式中，用户可输入 Y/n 确认或拒绝回答
- SSE 流式路径中，生成回答后 yield `{"type": "review", "content": answer}` 事件
- 新增 `POST /api/chat/review` 端点记录审核结果
- 审核结果可扩展：approved / rejected + feedback
- 未来可对接前端审核 UI（点赞/踩/修改建议）

---

## 5. LLM 策略：从单一模型到模型路由

### 现状

```
Qwen2.5:7B 用于：意图分类 + RAG 生成 + 闲聊 + 天气润色
```

一个模型做所有事，成本高、延迟大。

### 行业方案

| 任务 | 当前 | 推荐替代 | 收益 |
|------|------|---------|------|
| 意图分类 | qwen2.5:7b | 微调 BERT 小模型 (< 100M) | 延迟 2s → 50ms |
| 嵌入 | nomic-embed-text | BGE-M3 / mxbai-embed-large | 多语言更好 |
| RAG 生成 | qwen2.5:7b | Qwen2.5:7B 或 API 模型 | — |
| 闲聊 | qwen2.5:7b | Qwen2.5:0.5B / API 轻量模型 | 降本 |
| 摘要 | qwen2.5:7b | 小模型 / API | 降本 |

### 升级建议

**P1 — 分类专用模型**

```python
# 当前: 每次分类调用 7B LLM (~1-2s)
# 升级: BERT 微调分类器 (~50ms)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=4)
```

标注 200-500 条对话数据即可获得 >90% 准确率。

**P2 — 模型路由 (Model Router)**

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
裸机 python src/main.py --api
日志: print / logging
无 Docker、无监控、无 CI/CD
```

### 行业方案

**P1 — Docker 容器化**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports: ["8000:8000"]
    depends_on: [redis, ollama]
  redis:
    image: redis:7-alpine
  ollama:
    image: ollama/ollama
    volumes: ["./ollama:/root/.ollama"]
```

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
| **P1** | ✅ 人机协同 (interrupt_after + review) | ✅ 已完成 | 安全性 / 可控性 |
| **P0** | Docker 容器化 | 1 天 | 部署标准化 |
| **P1** | 查询转换 (Multi-Query / HyDE) | 1-2 天 | 召回复盖率提升 |
| **P1** | 分层记忆 (摘要+语义检索) | 2-3 天 | 记忆质量 |
| **P1** | LangSmith 可观测性 | 0.5 天 | 调试效率 |
| **P1** | BERT 分类器替代 LLM 分类 | 2-3 天 | 延迟 2s→50ms |
| **P1** | RAGAS 评估 | 1-2 天 | 量化质量 |
| **P2** | 模型路由 | 1-2 天 | 成本优化 |
| **P2** | WebSocket / 前端增强 | 3-5 天 | 用户体验 |
| **P2** | 生产级向量库 (Qdrant) | 3-5 天 | 扩展性 |
| **P2** | CI/CD + 回归测试 | 2-3 天 | 工程质量 |

---

## 总结

当前项目是一个**功能完善的原型**，已完成五项核心升级（语义分块、混合检索、重排序、
工具调用标准化、人机协同），流式交互顺畅、双模式可用。通向生产级系统的主要差距在于：

1. **可观测性**: 无 LLM 调用追踪 → 调试效率低
2. **部署**: 无容器化 → 环境一致性差
3. **查询优化**: 无 Query Rewrite / Multi-Query → 召回复盖率不足
4. **评估**: 无量化指标 → 优化方向不明确
5. **并行节点**: 无 fan-out → 无法并行执行多个工具

建议按 P0 → P1 → P2 的顺序逐步演进，每完成一个阶段进行一次质量评估。
