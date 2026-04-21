import os
from dotenv import load_dotenv

load_dotenv()

# Ollama配置
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 文档路径配置
DATA_DIR = "data/product_docs"
VECTOR_STORE_PATH = "data/vector_store"

# 模型配置
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"

# 检索配置
TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

AMAP_API_KEY = "442835220e8faecaf0dc626b52e3f143"
AMAP_WEATHER_URL = "https://restapi.amap.com/v3/weather/weatherInfo"