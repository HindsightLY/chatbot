# Medical Chatbot - 智能医疗健康咨询系统

基于 RAG（检索增强生成）的医疗咨询系统，支持 API 和 CLI 两种交互模式。

## 快速开始

### 前置条件
- Python 3.8+
- [Ollama](https://ollama.ai) 服务已启动

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
├── config/app_config.py      # 全局配置（LLM、API Key、检索参数等）
├── data/
│   ├── disease/              # 医疗文档（.txt/.md）
│   ├── faiss_index/          # FAISS 向量索引持久化
│   └── chat_memory/          # 对话历史向量存储
├── src/
│   ├── main.py               # 入口：API / CLI 模式
│   ├── api/routers/
│   │   ├── chat_router.py    # FastAPI 路由 & 意图分发
│   │   └── cli_router.py     # CLI 交互循环
│   ├── service/
│   │   ├── chatbot.py        # RAG 链 + 混合记忆
│   │   ├── vector_store.py   # FAISS 创建/加载/检索
│   │   ├── document_loader.py# 文档加载与分块
│   │   ├── intent_classifier.py # LLM 意图分类
│   │   ├── tool_manager.py   # 天气/闲聊处理器
│   │   └── system_initializer.py # 组件组装
│   ├── tools/
│   │   ├── weather_tool.py   # 高德天气 API
│   │   └── news_tool.py      # 聚合数据新闻 API
│   └── utils/
│       ├── text_utils.py     # 城市提取、文本清理
│       └── logger_config.py  # 日志 & 性能监控
└── requirements.txt
```

## 核心流程

```
用户输入
  → IntentClassifier.classify()
    ├─ medical_inquiry → MedicalChatbot (RAG: 向量检索 → LLM 生成)
    ├─ chat_general    → ToolManager (天气 → 高德API / 闲聊 → LLM)
    ├─ system_query    → ToolManager (通用 LLM)
    └─ unknown         → 兜底走 RAG
```

### 数据准备
- 医疗文档放入 `data/disease/`
- 首次启动自动加载 → 分块(500字符, 重叠100) → 向量化 → 存入 FAISS
- 后续启动直接加载已持久化的索引

## 技术栈

| 组件 | 选型 |
|---|---|
| Web 框架 | FastAPI |
| LLM | Qwen2.5:7B (Ollama) |
| 嵌入模型 | nomic-embed-text (Ollama) |
| 向量库 | FAISS |
| 文档分块 | RecursiveCharacterTextSplitter |

---

> **注意**：本系统提供的医疗信息仅供参考，不能替代专业医疗建议。如有健康问题，请咨询专业医生。



