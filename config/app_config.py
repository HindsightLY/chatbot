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
    # 项目根目录 - 从配置文件位置向上两级
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据目录配置
    data_dir: str = os.path.join(project_root, "data")
    disease_dir: str = os.path.join(data_dir, "disease")
    vector_persist_dir: str = os.path.join(data_dir, "faiss_index")

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

    def ensure_data_dirs(self):
        """
        确保数据目录存在，如果不存在则创建
        """
        import os

        # 确保data目录存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"✅ 创建数据目录: {self.data_dir}")

        # 确保disease目录存在
        if not os.path.exists(self.disease_dir):
            os.makedirs(self.disease_dir)
            print(f"✅ 创建疾病文档目录: {self.disease_dir}")

        # 确保faiss_index目录存在
        if not os.path.exists(self.vector_persist_dir):
            os.makedirs(self.vector_persist_dir)
            print(f"✅ 创建向量索引目录: {self.vector_persist_dir}")


# 全局配置实例
APP_CONFIG = AppConfig()

# 确保数据目录存在
APP_CONFIG.ensure_data_dirs()


def get_app_config() -> AppConfig:
    """
    获取应用配置实例

    Returns:
        AppConfig: 应用配置实例
    """
    return APP_CONFIG