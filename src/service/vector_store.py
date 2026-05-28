"""
向量存储管理模块
负责 ChromaDB 向量库的创建、加载、持久化和检索

使用 nomic-embed-text 通过 Ollama 生成文本嵌入，存入 ChromaDB。
"""
import os
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from src.utils.logger_config import logger
from config.app_config import APP_CONFIG


class VectorStoreManager:
    """
    向量存储管理器

    职责:
      - 创建/加载 ChromaDB 集合
      - 将文档向量化并持久化到磁盘
      - 提供相似度搜索接口

    被 system_initializer.py 调用，结果注入 MedicalChatbot。
    """

    def __init__(self, persist_dir: str = None):
        """
        Args:
            persist_dir: ChromaDB 持久化目录，默认使用 APP_CONFIG.chroma_persist_dir
        """
        self.persist_dir = persist_dir if persist_dir is not None else APP_CONFIG.chroma_persist_dir
        self.embedding = self._create_embedding_model()

        os.makedirs(self.persist_dir, exist_ok=True)

        logger.info(f"📁 ChromaDB 存储目录: {self.persist_dir}")

        self._vector_store = None

    @property
    def vector_store(self):
        """懒加载: 首次访问时自动从磁盘加载"""
        if self._vector_store is None:
            self._vector_store = self.load_vector_store()
        return self._vector_store

    def _create_embedding_model(self) -> OllamaEmbeddings:
        """
        嵌入式模型与 agent.py 中使用的是同一模型，
        确保检索和记忆的向量空间一致。
        """
        return OllamaEmbeddings(
            model=APP_CONFIG.embedding_model_name,
            base_url=APP_CONFIG.llm_base_url
        )

    def load_vector_store(self) -> Optional[Chroma]:
        """
        从 persist_dir 加载已有的 ChromaDB 集合

        Returns:
            Chroma 实例，目录不存在或加载失败时返回 None
        """
        try:
            if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
                logger.info(f"🔄 尝试加载 ChromaDB 从: {self.persist_dir}")
                vector_store = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embedding,
                    collection_name=APP_CONFIG.chroma_collection_name
                )
                count = vector_store._collection.count()
                logger.info(f"✅ 成功加载 ChromaDB，包含 {count} 个文档")
                self._vector_store = vector_store
                return vector_store
            else:
                logger.info(f"📁 ChromaDB 目录不存在或为空: {self.persist_dir}")
                return None
        except Exception as e:
            logger.exception(f"❌ 加载 ChromaDB 失败")
            return None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        用文档列表创建 ChromaDB 集合并持久化

        Args:
            documents: DocumentLoader 分块后的 Document 列表

        Returns:
            Chroma 实例

        Raises:
            ValueError: documents 为空
        """
        if not documents:
            raise ValueError("文档列表为空，无法创建向量库")

        logger.info(f"🔄 开始创建 ChromaDB，文档数量: {len(documents)}")

        try:
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_dir,
                collection_name=APP_CONFIG.chroma_collection_name
            )

            count = vector_store._collection.count()
            logger.info(f"✅ ChromaDB 创建成功并保存到: {self.persist_dir}")
            logger.info(f"📊 ChromaDB 统计: 总文档数 = {count}")

            self._vector_store = vector_store
            return vector_store

        except Exception as e:
            logger.exception(f"❌ 创建 ChromaDB 失败")
            raise

    def similarity_search(self, query: str, k: int = None,
                          score_threshold: float = None) -> List[Document]:
        """
        执行相似度搜索并过滤低分结果

        Args:
            query: 用户查询
            k: 返回数量，默认 APP_CONFIG.retrieval_k
            score_threshold: 最低相似度，默认 APP_CONFIG.retrieval_score_threshold

        Returns:
            过滤后的文档列表
        """
        if k is None:
            k = APP_CONFIG.retrieval_k
        if score_threshold is None:
            score_threshold = APP_CONFIG.retrieval_score_threshold

        vector_store = self._vector_store
        if not vector_store:
            logger.warning("⚠️ ChromaDB 未加载，无法执行搜索")
            return []

        try:
            results = vector_store.similarity_search_with_relevance_scores(
                query,
                k=k
            )

            filtered_results = [
                doc for doc, score in results
                if score >= score_threshold
            ]

            logger.info(f"🔍 搜索结果: 找到 {len(results)} 个结果，过滤后 {len(filtered_results)} 个")
            return filtered_results

        except Exception as e:
            logger.exception(f"❌ 搜索失败（query前50字: {query[:50]}）")
            return []

    def get_retriever(self, k: int = None, score_threshold: float = None):
        """
        获取 LangChain Retriever 对象，可嵌入 LCEL 管道

        目前未被直接使用，MedicalChatbot 自行从 vector_store 创建 retriever。
        """
        vector_store = self._vector_store
        if not vector_store:
            return None

        return vector_store.as_retriever(
            search_kwargs={
                "k": k or APP_CONFIG.retrieval_k,
                "score_threshold": score_threshold or APP_CONFIG.retrieval_score_threshold
            }
        )
