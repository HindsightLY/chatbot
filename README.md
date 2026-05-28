# Medical Chatbot - 智能医疗健康咨询系统

基于 RAG + LangGraph Agent 的医疗咨询系统，支持 API 和 CLI 两种交互模式。

## 快速开始

### 前置条件
- Python 3.9+
- [Ollama](https://ollama.ai) 服务已启动
- Redis 服务已启动（本地默认端口 6379）

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
pip install -r requirements.txt
```

### 启动
```bash
python src/main.py            # CLI 模式（默认）
python src/main.py --api      # API 模式
```

### API 接口

| 端点 | 方法 | 说明 |
|---|---|---|
| `/api/chat` | POST | 普通问答 |
| `/api/chat/stream` | POST | SSE 流式问答 |
| `/api/chat/daily_news` | POST | 获取新闻 |

**请求体**:
```json
{"query": "感冒了怎么办", "session_id": "user-001"}
```

**响应**:
```json
{"intent": "medical_inquiry", "answer": "根据医学资料..."}
```

## 项目结构

```
medical_chatbot/
├── config/app_config.py         # 全局配置（LLM、Redis、ChromaDB、API Key 等）
├── data/
│   ├── disease/                 # 医疗文档（.txt）
│   └── chroma_db/               # ChromaDB 向量索引持久化
├── src/
│   ├── main.py                  # 入口：API / CLI 模式
│   ├── api/routers/
│   │   ├── chat_router.py       # FastAPI 路由 & 意图分发
│   │   └── cli_router.py        # CLI 交互循环
│   ├── service/
│   │   ├── agent.py             # LangGraph Agent（状态图）
│   │   ├── chatbot.py           # 聊天机器人（Agent 包装器）
│   │   ├── vector_store.py      # ChromaDB 创建/加载/检索
│   │   ├── memory_store.py      # Redis 对话记忆存储
│   │   ├── document_loader.py   # 文档加载与分块
│   │   ├── intent_classifier.py # LLM 意图分类
│   │   ├── tool_manager.py      # 天气/闲聊处理器
│   │   └── system_initializer.py# 组件组装
│   ├── tools/
│   │   ├── weather_tool.py      # 高德天气 API
│   │   └── news_tool.py         # 聚合数据新闻 API
│   └── utils/
│       ├── text_utils.py        # 城市提取、文本清理
│       └── logger_config.py     # 日志 & 性能监控
└── requirements.txt
```

## 核心流程

```
用户输入
  → IntentClassifier.classify()
    ├─ medical_inquiry → MedicalAgent (LangGraph: 分类→检索→生成→记忆)
    ├─ chat_general    → ToolManager (天气 → 高德API / 闲聊 → LLM)
    ├─ system_query    → ToolManager (通用 LLM)
    └─ unknown         → 兜底走 Agent(RAG)
```

### Agent 内部流转 (LangGraph StateGraph)
```
classify_intent
  → conditional edge
    ├─ medical_inquiry → retrieve_docs (ChromaDB) → generate_answer → save_memory (Redis)
    ├─ chat_general+天气 → weather_query → save_memory
    └─ chat_general/系统 → general_chat → save_memory
```

### 数据准备
- 医疗文档放入 `data/disease/`
- 首次启动自动加载 → 分块(500字符, 重叠100) → 向量化 → 存入 ChromaDB
- 后续启动直接加载已持久化的集合

## 技术栈

| 组件 | 选型 |
|---|---|
| Web 框架 | FastAPI |
| LLM | Qwen2.5:7B (Ollama) |
| 嵌入模型 | nomic-embed-text (Ollama) |
| 向量库 | ChromaDB |
| 对话记忆 | Redis |
| Agent 框架 | LangGraph |
| 文档分块 | RecursiveCharacterTextSplitter |

---

> **注意**：本系统提供的医疗信息仅供参考，不能替代专业医疗建议。如有健康问题，请咨询专业医生。
