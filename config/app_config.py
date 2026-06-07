"""
应用配置模块 — 基于 Pydantic 集中管理所有可调参数

全局单例 APP_CONFIG 被以下模块引用:
  - agent.py            (LLM / 检索 / 分层记忆)
  - vector_store.py     (ChromaDB 路径 / 混合检索 / 重排序)
  - memory_store.py     (Redis 连接)
  - document_loader.py  (分块 / 语义分块参数)
  - intent_classifier.py (LLM / BERT 模型名)
  - tool_manager.py     (LLM)
  - bert_classifier.py  (模型名 / 阈值)
  - weather_tool.py     (高德 API Key / URL)
  - news_tool.py        (新闻类型)
  - chat_router.py      (新闻类型)
  - text_utils.py       (城市列表)
  - main.py             (API host/port)

敏感信息（API Key）应迁移至环境变量。
"""
import os
from pydantic import BaseModel
from src.utils.logger_config import logger


class AppConfig(BaseModel):
    """
    应用配置类。

    使用 Pydantic BaseModel 提供类型校验和默认值。
    所有路径类字段在类定义时自动计算，不支持运行时修改。
    """

    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── 数据目录 ──
    data_dir: str = os.path.join(project_root, "data")
    disease_dir: str = os.path.join(data_dir, "disease")
    chroma_persist_dir: str = os.path.join(data_dir, "chroma_db")

    # ── LLM（生成 / 分类 / 闲聊） ──
    llm_model_name: str = "qwen2.5:7b"
    llm_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_temperature: float = 0.1

    # ── 嵌入模型 ──
    embedding_model_name: str = "nomic-embed-text"

    # ── API 服务 ──
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── 文档分块 ──
    chunk_size: int = 500
    chunk_overlap: int = 100
    use_semantic_chunking: bool = True                     # 是否使用语义分块
    semantic_chunk_min_size: int = 200                     # 语义块最小字符数
    semantic_chunk_max_size: int = 800                     # 语义块最大字符数
    semantic_chunk_breakpoint_percentile: int = 80         # 断点百分位阈值

    # ── Hugging Face 镜像 ──
    hf_mirror: str = "https://hf-mirror.com"

    # ── BERT 分类器 ──
    use_bert_classifier: bool = True
    bert_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    bert_classifier_threshold: float = 0.45                # 余弦相似度阈值，低于此值返回 unknown

    # ── HyDE 查询转换 ──
    use_hyde: bool = True
    hyde_temperature: float = 0.1

    # ── 检索 ──
    retrieval_k: int = 6
    retrieval_score_threshold: float = 0.3
    use_hybrid_search: bool = True                         # 是否启用混合检索（向量 + BM25）
    use_reranking: bool = True                             # 是否启用 Cross-Encoder 重排序
    hybrid_prefetch_k: int = 20                            # 混合检索预取数
    rrf_k: int = 60                                        # RRF 融合常数
    rerank_top_k: int = 6                                  # 重排序后返回数

    # ── Redis 对话记忆 ──
    redis_host: str = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_db: int = 0
    redis_ttl: int = 86400                                 # Redis key 过期时间（秒）

    # ── 分层记忆 ──
    use_hierarchical_memory: bool = True
    memory_summary_turns: int = 20                         # 超过多少轮触发摘要
    memory_summary_max_age: int = 30                       # 保留的最大摘要数
    memory_retrieval_k: int = 3                            # 检索返回的相关摘要数

    # ── ChromaDB ──
    chroma_collection_name: str = "medical_docs"

    # ── 城市列表（天气查询正则匹配用） ──
    common_cities: list = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "天津",
        "重庆", "成都", "武汉", "西安", "青岛", "大连", "厦门", "宁波",
        "长沙", "郑州", "济南", "福州", "合肥", "太原", "石家庄", "沈阳",
        "长春", "哈尔滨", "昆明", "南宁", "海口", "兰州", "银川", "西宁",
        "乌鲁木齐", "拉萨", "呼和浩特", "香港", "澳门", "台北"
    ]

    # ── 高德天气 API ──
    amap_api_key: str = "442835220e8faecaf0dc626b52e3f143"
    amap_weather_url: str = "https://restapi.amap.com/v3/weather/weatherInfo"

    # ── 新闻分类 ──
    valid_news_types: list = [
        "top", "shehui", "guonei", "guoji",
        "yule", "tiyu", "junshi", "keji",
        "caijing", "shishang"
    ]

    def ensure_data_dirs(self):
        """创建 data/disease/chroma_db 目录（首次运行必需）"""
        for attr in ["data_dir", "disease_dir", "chroma_persist_dir"]:
            path = getattr(self, attr)
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"✅ 创建目录: {path}")


APP_CONFIG = AppConfig()


def get_app_config() -> AppConfig:
    """获取全局配置单例"""
    return APP_CONFIG
