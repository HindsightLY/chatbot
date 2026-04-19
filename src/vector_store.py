import os
from .logger_config import logger
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStoreManager:
    def __init__(self, persist_dir: str = "faiss_index"):
        self.persist_dir = persist_dir
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )
        # ✅ 定义索引文件的完整路径（用于检查）
        self.index_file = os.path.join(persist_dir, "index.faiss")

    def create_vector_store(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            length_function=len,
        )
        split_docs = text_splitter.split_documents(documents)
        logger.info(f"📊 分块数: {len(split_docs)}")

        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        # ✅ 保存时指定文件名
        vector_store.save_local(self.persist_dir, index_name="index")
        return vector_store

    def load_vector_store(self):
        # ✅ 改进：使用 try-except 捕获加载异常，而不是仅判断路径
        try:
            # 检查索引文件是否存在（双重保险）
            if os.path.exists(self.index_file):
                return FAISS.load_local(
                    self.persist_dir,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                    index_name="index"  # ✅ 对应保存时的文件名
                )
            else:
                logger.info("🆕 未检测到现有索引文件，将创建新的向量库。")
                return None
        except Exception as e:
            logger.info(f"⚠️ 加载向量库失败 (可能文件损坏或格式不匹配): {e}")
            logger.info("🆕 将重新创建向量库。")
            return None
