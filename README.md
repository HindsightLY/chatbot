# 医疗聊天机器人项目文档

## 项目概述

本项目是一个基于大语言模型的医疗健康咨询系统，能够处理医疗咨询、天气查询、新闻获取等多种用户需求。系统采用模块化设计，集成了文档加载、向量化存储、意图识别、对话管理等核心功能。

## 项目架构

### 技术栈

- Python 3.8+
- FastAPI - Web框架
- LangChain - LLM应用开发框架
- FAISS - 向量数据库
- Ollama - 本地LLM服务
- Nomic Embed Text - 嵌入模型
- Qwen2.5 - 主要LLM模型

### 项目结构

```
medical_chatbot/
├── config/                 # 配置文件目录 (API Keys, 系统参数)
├── data/                   # 数据存储目录
│   ├── disease/            # 疾病相关原始数据 (PDF, TXT, MD等)
│   └── faiss_index/        # 构建好的 FAISS 向量索引文件
├── src/                    # 源代码核心目录
│   ├── api/                # API 接口层
│   │   └── routers/        # 路由定义
│   │       ├── chat_router.py  # 聊天接口路由 (/api/chat)
│   │       └── cli_router.py   # 命令行交互路由 (可选)
│   ├── service/            # 业务逻辑层
│   │   ├── system_initializer.py # 系统启动初始化 (加载模型、索引)
│   │   └── tool_manager.py     # 工具调度管理器
│   ├── tools/              # 外部工具集
│   │   ├── news_tool.py    # 新闻查询工具
│   │   └── weather_tool.py # 天气查询工具
│   ├── utils/              # 通用工具类
│   │   └── text_utils.py   # 文本处理工具
│   ├── chatbot.py          # 核心聊天机器人逻辑 (Brain)
│   ├── document_loader.py  # 文档加载与预处理
│   ├── intent_classifier.py# 意图识别分类器
│   ├── main.py             # 程序入口 (FastAPI 启动)
│   └── vector_store.py     # 向量数据库封装 (FAISS)
├── .venv/                  # Python 虚拟环境
├── requirements.txt        # 依赖库列表
└── README.md               # 项目说明文档
```

## 核心功能

### 1. 文档加载与处理

- 支持从指定目录加载TXT格式的医疗文档
- 自动提取疾病名称作为元数据
- 支持文档内容清洗和格式化

### 2. 向量存储管理

- 使用FAISS进行向量存储
- 支持文档分块和嵌入
- 提供持久化存储和加载功能
- 自动检测现有索引并重新创建

### 3. 意图分类

- 基于LLM的意图识别
- 支持医疗咨询、通用聊天、系统查询等意图
- 支持关键词匹配和语义分析
- 可扩展的意图分类 schema

### 4. 天气查询功能

- 支持城市天气信息查询
- 集成高德天气API
- 智能城市名称提取
- 自然语言回复生成

### 5. 对话管理

- 支持会话状态管理
- 基于意图的路由处理
- 性能监控和日志记录
- 错误处理和异常捕获

## 安装与配置

### 环境准备

1. 安装Python 3.8+
2. 安装Ollama服务
3. 拉取所需模型：

```
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

### 依赖安装

```
pip install -r requirements.txt
```

### 配置文件

创建 `.env` 文件：

```
AMAP_API_KEY=your_amap_api_key
AMAP_WEATHER_URL=http://restapi.amap.com/v3/weather/weatherInfo
```

## 使用说明

### 启动服务

```
cd medical_chatbot
python -m src.main
```

### API接口

#### 对话接口

- **URL**: `/api/chat`
- **Method**: POST
- **Request Body**:

```
{
  "query": "用户输入内容",
  "session_id": "会话ID"
}
```

- **Response**:

```
{
  "intent": "意图类型",
  "answer": "回复内容"
}
```

### 支持的查询类型

1. **医疗咨询**：疾病、症状、药物、治疗方法等
2. **天气查询**：城市天气、气温、湿度等
3. **通用聊天**：问候、闲聊等
4. **系统查询**：功能咨询、帮助等

## 优化特性

### 性能优化

- 意图分类缓存
- 向量库持久化
- 异步处理支持
- 性能监控装饰器

### 错误处理

- 完善的异常捕获
- 日志记录系统
- 错误恢复机制
- 用户友好的错误提示

### 扩展性

- 模块化设计
- 易于添加新功能
- 支持多种数据源
- 可配置的意图分类

## 开发指南

### 添加新意图

1. 在 `intent_classifier.py` 中更新 intent schema
2. 在 `main.py` 中添加对应处理逻辑
3. 测试意图识别准确性

### 添加新工具

1. 在 `tools.py` 中实现工具函数
2. 在 `main.py` 中添加调用逻辑
3. 更新意图分类关键词

## 未来改进方向

- 增加更多医疗数据源
- 支持多轮对话管理
- 添加用户认证系统
- 实现更精准的意图识别
- 优化性能和响应速度

## 贡献指南

欢迎贡献代码和改进！请遵循以下步骤：

1. Fork 项目仓库
2. 创建新分支
3. 实现功能或修复问题
4. 提交 Pull Request
5. 通过代码审查后合并

## 许可证

本项目采用 MIT 许可证，详情请参阅 LICENSE 文件。


