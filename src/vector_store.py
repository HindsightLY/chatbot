"""
向量存储管理模块
负责FAISS向量库的创建、加载和管理
"""
import os
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document
from src.logger_config import logger
from config.app_config import APP_CONFIG


class VectorStoreManager:
    """
    向量存储管理器
    负责向量库的创建、加载和持久化
    """

    def __init__(self, persist_dir: str = None):
        """
        初始化向量存储管理器

        Args:
            persist_dir: 持久化目录路径，如果为None则使用配置文件中的默认路径
        """
        # 使用配置中的路径，如果传入了路径则覆盖
        self.persist_dir = persist_dir if persist_dir is not None else APP_CONFIG.vector_persist_dir
        self.embedding = self._create_embedding_model()

        # 确保向量存储目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        logger.info(f"📁 向量存储目录: {self.persist_dir}")

    def _create_embedding_model(self) -> OllamaEmbeddings:
        """
        创建嵌入模型实例

        Returns:
            OllamaEmbeddings: 嵌入模型实例
        """
        return OllamaEmbeddings(
            model=APP_CONFIG.embedding_model_name,
            base_url="http://localhost:11434"
        )

    def load_vector_store(self) -> Optional[FAISS]:
        """
        加载现有的FAISS向量库

        Returns:
            Optional[FAISS]: FAISS向量库实例，如果加载失败则返回None
        """
        try:
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                logger.info(f"🔄 尝试加载向量库从: {self.persist_dir}")
                vector_store = FAISS.load_local(
                    self.persist_dir,
                    self.embedding,
                    allow_dangerous_deserialization=True  # 允许反序列化
                )
                logger.info(f"✅ 成功加载向量库，包含 {vector_store.index.ntotal} 个向量")
                return vector_store
            else:
                logger.info(f"📁 向量库目录不存在或为空: {self.persist_dir}")
                return None
        except Exception as e:
            logger.error(f"❌ 加载向量库失败: {e}")
            return None

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        创建新的FAISS向量库

        Args:
            documents: 文档列表

        Returns:
            FAISS: 创建的FAISS向量库实例
        """
        if not documents:
            raise ValueError("文档列表为空，无法创建向量库")

        logger.info(f"🔄 开始创建向量库，文档数量: {len(documents)}")

        try:
            # 创建FAISS向量库
            vector_store = FAISS.from_documents(
                documents,
                self.embedding,
                normalize_L2=True
            )

            # 保存到磁盘
            vector_store.save_local(self.persist_dir)
            logger.info(f"✅ 向量库创建成功并保存到: {self.persist_dir}")
            logger.info(f"📊 向量库统计: 总向量数 = {vector_store.index.ntotal}")

            return vector_store

        except Exception as e:
            logger.error(f"❌ 创建向量库失败: {e}")
            raise

    def similarity_search(self, query: str, k: int = None,
                         score_threshold: float = None) -> List[Document]:
        """
        执行相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量，默认使用配置中的值
            score_threshold: 相似度阈值，默认使用配置中的值

        Returns:
            List[Document]: 匹配的文档列表
        """
        if k is None:
            k = APP_CONFIG.retrieval_k
        if score_threshold is None:
            score_threshold = APP_CONFIG.retrieval_score_threshold

        vector_store = self.load_vector_store()
        if not vector_store:
            logger.warning("⚠️ 向量库未加载，无法执行搜索")
            return []

        try:
            # 执行相似度搜索
            results = vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )

            # 过滤低于阈值的结果
            filtered_results = [
                doc for doc, score in results
                if score >= score_threshold
            ]

            logger.info(f"🔍 搜索结果: 找到 {len(results)} 个结果，过滤后 {len(filtered_results)} 个")
            return filtered_results

        except Exception as e:
            logger.error(f"❌ 搜索失败: {e}")
            return []

    def get_retriever(self, k: int = None, score_threshold: float = None):
        """
        获取检索器对象

        Args:
            k: 返回结果数量
            score_threshold: 相似度阈值

        Returns:
            检索器对象
        """
        vector_store = self.load_vector_store()
        if not vector_store:
            return None

        return vector_store.as_retriever(
            search_kwargs={
                "k": k or APP_CONFIG.retrieval_k,
                "score_threshold": score_threshold or APP_CONFIG.retrieval_score_threshold
            }
        )