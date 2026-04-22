"""
应用配置模块
统一管理系统配置和参数
"""
import os
from pydantic import BaseModel


class AppConfig(BaseModel):
    """
    应用配置类
    """
    # 项目路径配置
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    disease_dir: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "disease")

    # 向量存储配置
    vector_persist_dir: str = "faiss_index"

    # LLM模型配置
    llm_model_name: str = "qwen2.5:7b"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.1

    # 嵌入模型配置
    embedding_model_name: str = "nomic-embed-text"

    # API配置
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # 检索参数配置
    retrieval_k: int = 6
    retrieval_score_threshold: float = 0.3

    # 分块参数配置
    chunk_size: int = 500
    chunk_overlap: int = 100

    # 城市列表
    common_cities: list = [
        "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "天津",
        "重庆", "成都", "武汉", "西安", "青岛", "大连", "厦门", "宁波",
        "长沙", "郑州", "济南", "福州", "合肥", "太原", "石家庄", "沈阳",
        "长春", "哈尔滨", "昆明", "南宁", "海口", "兰州", "银川", "西宁",
        "乌鲁木齐", "拉萨", "呼和浩特", "香港", "澳门", "台北"
    ]

    # 新闻类型验证
    valid_news_types: list = [
        "top", "shehui", "guonei", "guoji",
        "yule", "tiyu", "junshi", "keji",
        "caijing", "shishang"
    ]


# 全局配置实例
APP_CONFIG = AppConfig()


def get_app_config() -> AppConfig:
    """
    获取应用配置实例

    Returns:
        AppConfig: 应用配置实例
    """
    return APP_CONFIG