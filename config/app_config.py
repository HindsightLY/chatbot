"""
应用配置模块
基于 Pydantic BaseModel，集中管理所有可调参数

全局单例 APP_CONFIG 被以下模块引用:
  - chatbot.py      (LLM / 嵌入模型 / 检索参数)
  - vector_store.py (持久化路径 / 嵌入模型)
  - document_loader.py (分块参数)
  - intent_classifier.py (LLM)
  - tool_manager.py (LLM)
  - weather_tool.py (高德 API)
  - news_tool.py    (新闻类型)
  - chat_router.py  (新闻类型)
  - text_utils.py   (城市列表)

所有敏感信息（API Key 等）应迁移至环境变量。
"""
import os
from pydantic import BaseModel
from src.utils.logger_config import logger


class AppConfig(BaseModel):
    """应用配置，使用 Pydantic 提供类型校验和默认值"""

    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据目录
    data_dir: str = os.path.join(project_root, "data")
    disease_dir: str = os.path.join(data_dir, "disease")
    vector_persist_dir: str = os.path.join(data_dir, "faiss_index")

    # LLM — 用于 RAG 生成 & 意图分类 & 闲聊
    llm_model_name: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1

    # 嵌入模型 — 用于向量检索
    embedding_model_name: str = "nomic-embed-text"

    # API 服务
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 检索
    retrieval_k: int = 6
    retrieval_score_threshold: float = 0.3

    # 文档分块
    chunk_size: int = 500
    chunk_overlap: int = 100

    # 城市列表（天气查询用）
    common_cities: list = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "天津",
        "重庆", "成都", "武汉", "西安", "青岛", "大连", "厦门", "宁波",
        "长沙", "郑州", "济南", "福州", "合肥", "太原", "石家庄", "沈阳",
        "长春", "哈尔滨", "昆明", "南宁", "海口", "兰州", "银川", "西宁",
        "乌鲁木齐", "拉萨", "呼和浩特", "香港", "澳门", "台北"
    ]

    # 高德天气 API
    amap_api_key: str = "442835220e8faecaf0dc626b52e3f143"
    amap_weather_url: str = "https://restapi.amap.com/v3/weather/weatherInfo"

    # 新闻分类
    valid_news_types: list = [
        "top", "shehui", "guonei", "guoji",
        "yule", "tiyu", "junshi", "keji",
        "caijing", "shishang"
    ]

    def ensure_data_dirs(self):
        """创建 data/disease/faiss_index 目录（首次运行必需）"""
        for attr in ["data_dir", "disease_dir", "vector_persist_dir"]:
            path = getattr(self, attr)
            if not os.path.exists(path):
                os.makedirs(path)
                logger.info(f"✅ 创建目录: {path}")


APP_CONFIG = AppConfig()


def get_app_config() -> AppConfig:
    """获取全局配置实例"""
    return APP_CONFIG