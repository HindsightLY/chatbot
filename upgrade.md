# 技术栈升级路线 — 对比行业方案与演进方向

> 本文档对比当前项目选型与行业主流方案，提出可量化的升级路径。
> 按优先级（P0=关键 / P1=推荐 / P2=可选）排序。

---

## 1. 检索增强：从基础 RAG 到高级 RAG

### 现状

```
用户问题 → ChromaDB 相似度搜索 → LLM 生成
```

仅依赖单轮稠密向量检索，无查询优化、无重排序。

### 行业对比

| 技术 | 当前状态 | 行业方案 | 差距 |
|------|---------|---------|------|
| 检索方式 | 纯稠密向量 | 混合检索 (BM25 + Dense) | ❌ |
| 重排序 | 无 | Cohere Rerank / BGE-Reranker / Cross-Encoder | ❌ |
| 查询转换 | 无 | HyDE / Multi-Query / Query Rewrite | ❌ |
| 分块策略 | 固定 500 字符 | 语义分块 (Semantic Chunker) / 递归 LLM 分块 | ❌ |

### 升级建议

**P0 — 混合检索 (Hybrid Search)**

```
当前: query → embedding → cosine_sim(doc.vector, query.vector)

升级: query → embedding → cosine_sim(doc.vector, query.vector)  ← 稠密
           → BM25 → keyword_tfidf(doc.text, query.text)          ← 稀疏
           → weighted_sum(0.5*dense + 0.5*sparse)                ← 加权融合
```

ChromaDB 不原生支持 BM25，可通过 `rank_bm25` 库实现后融合(BFS)：
1. 稠密检索取 Top-20
2. BM25 检索取 Top-20
3. Reciprocal Rank Fusion (RRF) 合并排序 → 最终 Top-6

**P1 — 重排序 (Reranking)**

```
检索 Top-20 → BGE-Reranker-v2-m3 打分 → 取 Top-3 给 LLM
```

```bash
pip install sentence-transformers
# 模型: BAAI/bge-reranker-v2-m3 (1.5G, CPU 可运行)
```

收益：大幅减少注入 prompt 的噪声文档，提升回答准确率 5-15%。

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
StateGraph → 6 个节点 → conditional_edges → compile(MemorySaver)
```

仅用于简单的意图路由 + RAG 链。

### 行业对比

| 功能 | 当前 | LangGraph 完整能力 | AutoGen / CrewAI |
|------|------|-------------------|------------------|
| Human-in-the-Loop | ❌ | `interrupt_after` ✅ | ✅ |
| 并行节点 | ❌ | `add_node` + fan-out ✅ | ✅ 多 agent |
| 循环/重试 | ❌ | `add_conditional_edges` 自环 ✅ | ✅ |
| Tool Calling | ❌ 字符串 prompt | `bind_tools()` + ToolNode ✅ | ✅ 原生 |
| 多 Agent 协作 | ❌ | `Send()` API / Supervisor ✅ | ✅ 原生 |
| 持久化检查点 | ❌ | PostgreSQL / SQLite 检查点 ✅ | ❌ |

### 升级建议

**P1 — 工具调用标准化**

```python
# 当前: 字符串 prompt 嵌入工具描述
# 升级: LangChain Tool + bind_tools()

from langchain_core.tools import tool

@tool
def search_medical_docs(query: str) -> str:
    """搜索医学知识库"""
    return vector_store.search(query)

llm_with_tools = llm.bind_tools([search_medical_docs, get_weather])
tool_node = ToolNode([search_medical_docs, get_weather])
```

收益：LLM 自主决定是否、何时、以何参数调用工具，远比字符串 prompt 灵活。

**P2 — 人机协同 (Human-in-the-Loop)**

```python
def generate_answer(state):
    answer = llm.invoke(state["prompt"])
    return {"answer": answer}

# 在生成后中断，等待人工审核
graph.add_edge("generate_answer", "human_review")
graph.set_interrupt_after("generate_answer")
```

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
| **P0** | 混合检索 (BM25 + Dense) | 2-3 天 | 检索召回率 +10-20% |
| **P0** | 重排序 (Reranker) | 1-2 天 | 准确率 +5-15% |
| **P0** | Docker 容器化 | 1 天 | 部署标准化 |
| **P1** | 查询转换 (Multi-Query / HyDE) | 1-2 天 | 召回复盖率提升 |
| **P1** | 工具调用标准化 (bind_tools) | 1-2 天 | Agent 灵活性 |
| **P1** | 分层记忆 (摘要+语义检索) | 2-3 天 | 记忆质量 |
| **P1** | LangSmith 可观测性 | 0.5 天 | 调试效率 |
| **P1** | BERT 分类器替代 LLM 分类 | 2-3 天 | 延迟 2s→50ms |
| **P1** | RAGAS 评估 | 1-2 天 | 量化质量 |
| **P2** | 人机协同 (Human-in-the-Loop) | 2-3 天 | 安全性 |
| **P2** | 模型路由 | 1-2 天 | 成本优化 |
| **P2** | WebSocket / 前端增强 | 3-5 天 | 用户体验 |
| **P2** | 生产级向量库 (Qdrant) | 3-5 天 | 扩展性 |
| **P2** | CI/CD + 回归测试 | 2-3 天 | 工程质量 |

---

## 总结

当前项目是一个**功能完善的原型**：核心 RAG 流程完整、流式交互顺畅、双模式可用。
通向生产级系统的主要差距在于：

1. **检索质量**: 无混合检索、无重排序 → 最优先补齐
2. **可观测性**: 无 LLM 调用追踪 → 调试效率低
3. **部署**: 无容器化 → 环境一致性差
4. **评估**: 无量化指标 → 优化方向不明确
5. **Agent 能力**: 工具调用不规范 → 扩展困难

建议按 P0 → P1 → P2 的顺序逐步演进，每完成一个阶段进行一次质量评估。
