# Medical Chatbot - 智能医疗健康咨询系统

基于 **RAG + LangGraph Agent** 的智能医疗问答系统，支持 SSE 流式实时对话。
通过检索增强生成技术，结合医学文档知识库，提供专业、准确的医疗信息咨询。

## 快速开始

### 前置条件

- Python 3.9+
- [Ollama](https://ollama.ai) 服务已启动（默认 `http://localhost:11434`）
- Redis 服务已启动（默认 `127.0.0.1:6379`）

```bash
# 1. 安装依赖
ollama pull nomic-embed-text
ollama pull qwen2.5:7b

# 2. 安装 Python 包
pip install -r requirements.txt

# 3. 启动
python src/main.py --api    # 浏览器自动打开 http://localhost:8000
```

### CLI 模式

```bash
python src/main.py          # 命令行交互
```

## 项目结构

```
medical_chatbot/
├── config/
│   └── app_config.py           # 全局配置（LLM / Redis / ChromaDB / API Key）
├── data/
│   ├── disease/                # 医疗知识文档（放 .txt / .md 文件）
│   └── chroma_db/              # ChromaDB 向量索引持久化
├── src/
│   ├── main.py                 # 入口：API / CLI 双模式
│   ├── api/routers/
│   │   ├── chat_router.py      # FastAPI 路由 & SSE 流式接口
│   │   └── cli_router.py       # 命令行交互循环
│   ├── service/
│   │   ├── agent.py            # LangGraph Agent（状态图 + 流式推理）
│   │   ├── chatbot.py          # 聊天机器人包装器
│   │   ├── vector_store.py     # ChromaDB 创建 / 加载 / 检索
│   │   ├── memory_store.py     # Redis 对话记忆
│   │   ├── document_loader.py  # 文档加载与分块
│   │   ├── intent_classifier.py# LLM 意图分类
│   │   ├── tool_manager.py     # 天气 & 闲聊处理器
│   │   └── system_initializer.py# 组件依赖注入组装
│   ├── tools/
│   │   ├── weather_tool.py     # 高德天气 API
│   │   └── news_tool.py        # 聚合数据新闻 API
│   ├── static/
│   │   └── index.html          # 单页面 Web 前端
│   └── utils/
│       ├── text_utils.py       # 城市提取 / 文本清理
│       └── logger_config.py    # 日志 & 性能监控
├── description.md              # 详细设计文档
├── summary.md                  # RAG 流程总结
├── upgrade.md                  # 技术栈升级路线
└── requirements.txt
```

## 核心流程

### 请求生命周期

```
用户输入
  → IntentClassifier.classify()   (LLM 分类: medical / chat / system)
    ├─ medical_inquiry / unknown
    │     → ChromaDB 检索医学文档
    │     → LLM 生成回答 (对话历史 + 医学文档 + 当前问题)
    │     → Redis 保存本轮问答
    │
    ├─ chat_general + 天气
    │     → 高德天气 API → LLM 润色输出
    │     → Redis 保存
    │
    └─ chat_general / system_query
          → LLM 直接回复 (对话历史 + 当前问题)
          → Redis 保存
```

### 流式 SSE 协议

```javascript
// 前端接收格式
data: {"intent":"medical_inquiry"}   // 意图事件（首个）
data: "头痛"                          // 逐 token，JSON 字符串
data: "可能"
...
data: [DONE]                          // 终止
```

## API 接口

### `POST /api/chat/stream` — 流式对话

```bash
curl -X POST http://localhost:8000/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "头痛怎么办", "session_id": "user-001"}'
```

### `POST /api/chat` — 普通对话

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "头痛怎么办", "session_id": "user-001"}'

# 响应: {"intent": "medical_inquiry", "answer": "根据医学资料..."}
```

### `POST /api/chat/daily_news` — 新闻

```bash
curl -X POST http://localhost:8000/api/chat/daily_news \
  -H "Content-Type: application/json" \
  -d '{"news_type": "top"}'
```

## 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 高性能异步框架 |
| LLM | Qwen2.5:7B (Ollama) | 本地部署，完全离线 |
| 嵌入模型 | nomic-embed-text (Ollama) | 768 维 |
| 向量数据库 | ChromaDB | 持久化到磁盘 |
| 对话记忆 | Redis | List 结构，TTL 24h |
| Agent 框架 | LangGraph | StateGraph 多节点编排 |
| 文档分块 | RecursiveCharacterTextSplitter | 中文标点递归分割 |
| 前端 | 原生 HTML + Fetch ReadableStream | 无框架依赖 |
| 外部 API | 高德天气 / 聚合数据新闻 | 可选 |

## 配置

见 `config/app_config.py`，关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_model_name` | `qwen2.5:7b` | 生成模型 |
| `llm_temperature` | `0.1` | 回复一致性 |
| `retrieval_k` | `6` | 检索文档数 |
| `retrieval_score_threshold` | `0.3` | 相似度阈值 |
| `chunk_size` | `500` | 文档分块大小 |
| `redis_ttl` | `86400` | 记忆过期时间(秒) |

---

> **⚠️ 免责声明**: 本系统提供的医疗信息仅供参考，不能替代专业医疗建议。
> 如有健康问题，请咨询专业医生。
