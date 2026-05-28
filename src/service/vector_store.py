"""
向量存储管理模块
负责 ChromaDB 向量库的创建、加载、持久化和检索

检索流水线（由 APP_CONFIG 控制各阶段开关）:
  1. 混合检索 (Hybrid Search)
     ├─ 稠密检索: ChromaDB 余弦相似度 (Top-N)
     └─ 稀疏检索: BM25 关键词匹配 (Top-N)
     └─ RRF 融合: Reciprocal Rank Fusion 合并排序
  2. Cross-Encoder 重排序 (可选)
     └─ BGE-Reranker / MiniLM 对 (query, doc) 对逐一打分
"""
import os
import jieba
import numpy as np
from typing import List, Optional, Tuple
from rank_bm25 import BM25Okapi
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
      - 提供混合检索 (稠密+稀疏+RRF)
      - 提供 Cross-Encoder 重排序

    被 system_initializer.py 调用，结果注入 MedicalChatbot。
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir if persist_dir is not None else APP_CONFIG.chroma_persist_dir
        self.embedding = self._create_embedding_model()

        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(f"📁 ChromaDB 存储目录: {self.persist_dir}")

        self._vector_store = None
        self._bm25_index = None
        self._bm25_docs: List[Tuple[str, str, dict]] = []
        self._reranker = None

    # ────────── ChromaDB 懒加载 ──────────

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = self.load_vector_store()
        return self._vector_store

    def _create_embedding_model(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=APP_CONFIG.embedding_model_name,
            base_url=APP_CONFIG.llm_base_url
        )

    # ────────── ChromaDB 加载/创建 ──────────

    def load_vector_store(self) -> Optional[Chroma]:
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
                self._ensure_bm25()
                return vector_store
            else:
                logger.info(f"📁 ChromaDB 目录不存在或为空: {self.persist_dir}")
                return None
        except Exception as e:
            logger.exception(f"❌ 加载 ChromaDB 失败")
            return None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
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
            self._build_bm25_index(documents)
            return vector_store

        except Exception as e:
            logger.exception(f"❌ 创建 ChromaDB 失败")
            raise

    # ────────── BM25 索引 ──────────

    def _build_bm25_index(self, documents: List[Document] = None):
        """
        构建 BM25 倒排索引

        优先使用传入的 Document 列表（create 时），
        否则从已加载的 ChromaDB 集合中读取全部文档。
        """
        try:
            if documents is None:
                if not self._vector_store:
                    logger.warning("⚠️ ChromaDB 未加载，无法构建 BM25 索引")
                    return
                collection = self._vector_store._collection
                all_data = collection.get(include=["documents", "metadatas"])
                if not all_data or not all_data.get("documents"):
                    return
                raw_texts = all_data["documents"]
                metadatas = all_data["metadatas"]
            else:
                raw_texts = [d.page_content for d in documents]
                metadatas = [d.metadata for d in documents]

            tokenized_corpus = []
            self._bm25_docs = []

            for i, text in enumerate(raw_texts):
                tokens = list(jieba.cut(text))
                tokenized_corpus.append(tokens)
                meta = metadatas[i] if i < len(metadatas) else {}
                doc_id = f"{meta.get('source', '')}_{meta.get('chunk_index', i)}"
                self._bm25_docs.append((doc_id, text, meta))

            self._bm25_index = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ BM25 索引构建完成，共 {len(tokenized_corpus)} 个文档")

        except Exception as e:
            logger.warning(f"⚠️ BM25 索引构建失败，降级为纯稠密检索: {e}")
            self._bm25_index = None

    def _ensure_bm25(self):
        """确保 BM25 索引已构建（懒加载）"""
        if self._bm25_index is None and self._vector_store is not None:
            self._build_bm25_index()

    # ────────── Cross-Encoder 重排序 ──────────

    @property
    def reranker(self):
        """懒加载 Cross-Encoder 重排序模型"""
        if self._reranker is None and APP_CONFIG.use_reranking:
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(
                    'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    device='cpu'
                )
                logger.info("✅ Cross-Encoder 重排序模型加载完成")
            except ImportError:
                logger.warning("⚠️ sentence-transformers 未安装，跳过 Cross-Encoder 重排序")
            except Exception as e:
                logger.warning(f"⚠️ Cross-Encoder 加载失败: {e}")
        return self._reranker

    # ────────── 相似度搜索（稠密） ──────────

    def similarity_search(self, query: str, k: int = None,
                          score_threshold: float = None) -> List[Document]:
        """纯稠密向量检索（兼容旧接口）"""
        if k is None:
            k = APP_CONFIG.retrieval_k
        if score_threshold is None:
            score_threshold = APP_CONFIG.retrieval_score_threshold

        vector_store = self._vector_store
        if not vector_store:
            logger.warning("⚠️ ChromaDB 未加载，无法执行搜索")
            return []

        try:
            results = vector_store.similarity_search_with_relevance_scores(query, k=k)
            filtered = [doc for doc, score in results if score >= score_threshold]
            logger.info(f"🔍 稠密检索: 共 {len(results)} 个，过滤后 {len(filtered)} 个")
            return filtered
        except Exception as e:
            logger.exception(f"❌ 搜索失败（query前50字: {query[:50]}）")
            return []

    # ────────── 混合检索 + 重排序（主入口） ──────────

    def hybrid_search(self, query: str, k: int = None) -> List[Document]:
        """
        混合检索 (BM25 + Dense + RRF) → 可选 Cross-Encoder 重排序 → 返回 Top-K

        流水线:
          Step 1: ChromaDB 稠密检索 Top-N (N = hybrid_prefetch_k)
          Step 2: BM25 稀疏检索 Top-N
          Step 3: RRF 融合 (reciprocal_rank = 1 / (rrf_k + rank))
          Step 4: Cross-Encoder 重排序（若启用且模型可用）
          Step 5: 返回 Top-K
        """
        if k is None:
            k = APP_CONFIG.retrieval_k

        vector_store = self._vector_store
        if not vector_store:
            logger.warning("⚠️ ChromaDB 未加载，无法执行检索")
            return []

        prefetch_k = APP_CONFIG.hybrid_prefetch_k
        all_docs_dict = {}
        seen_ids = set()

        # ── Step 1: 稠密检索 ──
        try:
            dense_results = vector_store.similarity_search_with_relevance_scores(
                query, k=prefetch_k
            )
            for rank, (doc, score) in enumerate(dense_results):
                doc_id = self._doc_id(doc)
                all_docs_dict[doc_id] = (doc, score, 0.0, rank)
                seen_ids.add(doc_id)
            logger.info(f"🔍 稠密检索完成: {len(dense_results)} 个结果")
        except Exception as e:
            logger.warning(f"⚠️ 稠密检索失败: {e}")
            dense_results = []

        # ── Step 2: BM25 稀疏检索 ──
        self._ensure_bm25()
        if self._bm25_index and self._bm25_docs:
            try:
                query_tokens = list(jieba.cut(query))
                bm25_scores = self._bm25_index.get_scores(query_tokens)
                ranked_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:prefetch_k]

                for rank, idx in enumerate(ranked_indices):
                    doc_id, text, meta = self._bm25_docs[idx]
                    if doc_id not in seen_ids:
                        bm25_doc = Document(page_content=text, metadata=meta)
                        all_docs_dict[doc_id] = (bm25_doc, 0.0, float(bm25_scores[idx]), rank)
                        seen_ids.add(doc_id)
                    else:
                        existing = all_docs_dict[doc_id]
                        all_docs_dict[doc_id] = (
                            existing[0], existing[1], float(bm25_scores[idx]), existing[3]
                        )

                logger.info(f"🔍 BM25 检索完成: {len(ranked_indices)} 个结果")
            except Exception as e:
                logger.warning(f"⚠️ BM25 检索失败: {e}")

        if not all_docs_dict:
            logger.warning("⚠️ 检索结果为空")
            return []

        # ── Step 3: RRF 融合 ──
        rrf_k_const = APP_CONFIG.rrf_k
        scored_docs = []
        for doc_id, (doc, dense_score, bm25_score, dense_rank) in all_docs_dict.items():
            rrf = 0.0
            if dense_rank is not None:
                rrf += 1.0 / (rrf_k_const + dense_rank + 1)
            bm25_rank = next(
                (i for i, (did, _, _) in enumerate(self._bm25_docs) if did == doc_id),
                None
            ) if self._bm25_docs else None
            if bm25_rank is not None:
                rrf += 1.0 / (rrf_k_const + bm25_rank + 1)
            scored_docs.append((doc, rrf))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"🔀 RRF 融合完成: {len(scored_docs)} 个结果")

        # ── Step 4: Cross-Encoder 重排序 ──
        reranker_model = self.reranker
        if reranker_model and APP_CONFIG.use_reranking and len(scored_docs) > 1:
            try:
                pairs = [[query, doc.page_content] for doc, _ in scored_docs]
                ce_scores = reranker_model.predict(pairs)
                scored_docs = list(zip([d for d, _ in scored_docs], ce_scores))
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                logger.info(f"🎯 Cross-Encoder 重排序完成")
            except Exception as e:
                logger.warning(f"⚠️ Cross-Encoder 重排序失败: {e}")

        # ── Step 5: 返回 Top-K ──
        final_docs = [doc for doc, _ in scored_docs[:k]]
        logger.info(f"📚 最终返回 {len(final_docs)} 个文档")
        return final_docs

    # ────────── 工具方法 ──────────

    @staticmethod
    def _doc_id(doc: Document) -> str:
        return f"{doc.metadata.get('source', '')}_{doc.metadata.get('chunk_index', '')}"

    def get_retriever(self, k: int = None, score_threshold: float = None):
        """获取 LangChain Retriever 对象（兼容旧接口）"""
        vector_store = self._vector_store
        if not vector_store:
            return None
        return vector_store.as_retriever(
            search_kwargs={
                "k": k or APP_CONFIG.retrieval_k,
                "score_threshold": score_threshold or APP_CONFIG.retrieval_score_threshold
            }
        )
