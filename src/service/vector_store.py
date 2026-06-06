"""
向量存储管理模块 — ChromaDB 创建、加载、混合检索、重排序

检索流水线（由 APP_CONFIG 控制各阶段开关）:
  1. 混合检索 (Hybrid Search)
     ├─ 稠密检索: ChromaDB 余弦相似度 (Top-N, N = hybrid_prefetch_k)
     └─ 稀疏检索: BM25 关键词匹配 (jieba 分词 + BM25Okapi, Top-N)
     └─ RRF 融合: Reciprocal Rank Fusion 合并排序（分别计算稠密和稀疏的贡献）
  2. Cross-Encoder 重排序 (可选)
     └─ cross-encoder/ms-marco-MiniLM-L-6-v2 对 (query, doc) 对逐一打分

被 system_initializer.py 调用，结果通过 system_initializer.vector_store 暴露给 agent.py。
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
      - 创建/加载 ChromaDB 集合（懒加载，磁盘持久化）
      - 构建 BM25 倒排索引（用于稀疏检索）
      - 提供混合检索（稠密 + 稀疏 + RRF 融合 + 可选重排序）
      - 提供纯稠密检索接口（兼容旧接口）

    RRF 计算说明:
      每个文档的最终得分 = Σ 1/(rrf_k + rank)
      稠密和稀疏分别在 rank 位置上给分。
      对于只在 BM25 中命中的文档，稠密 rank 为 None，不贡献稠密分数。
    """

    def __init__(self, persist_dir: str = None):
        self.persist_dir = persist_dir if persist_dir is not None else APP_CONFIG.chroma_persist_dir
        self.embedding = self._create_embedding_model()

        os.makedirs(self.persist_dir, exist_ok=True)
        logger.info(f"📁 ChromaDB 存储目录: {self.persist_dir}")

        self._vector_store = None                    # Chroma 实例（懒加载）
        self._bm25_index = None                      # BM25Okapi 实例
        self._bm25_docs: List[Tuple[str, str, dict]] = []  # BM25 文档列表: [(doc_id, text, metadata)]
        self._reranker = None                        # Cross-Encoder 模型实例

    # ══════════════════════════════════════════
    #  ChromaDB 懒加载
    # ══════════════════════════════════════════

    @property
    def vector_store(self):
        """懒加载 ChromaDB：首次访问时从磁盘加载"""
        if self._vector_store is None:
            self._vector_store = self.load_vector_store()
        return self._vector_store

    def _create_embedding_model(self) -> OllamaEmbeddings:
        return OllamaEmbeddings(
            model=APP_CONFIG.embedding_model_name,
            base_url=APP_CONFIG.llm_base_url
        )

    # ══════════════════════════════════════════
    #  ChromaDB 加载/创建
    # ══════════════════════════════════════════

    def load_vector_store(self) -> Optional[Chroma]:
        """
        从磁盘加载已持久化的 ChromaDB 集合

        若 persist_dir 存在且非空，尝试加载已有 ChromaDB；
        加载成功后自动确保 BM25 索引已构建。
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
                self._ensure_bm25()
                return vector_store
            else:
                logger.info(f"📁 ChromaDB 目录不存在或为空: {self.persist_dir}")
                return None
        except Exception as e:
            logger.exception(f"❌ 加载 ChromaDB 失败")
            return None

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """
        创建新的 ChromaDB 集合并持久化到磁盘

        Args:
            documents: Document 列表（已分块）

        Returns:
            Chroma 实例（同时构建了 BM25 索引）
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
            self._build_bm25_index(documents)
            return vector_store

        except Exception as e:
            logger.exception(f"❌ 创建 ChromaDB 失败")
            raise

    # ══════════════════════════════════════════
    #  BM25 索引
    # ══════════════════════════════════════════

    def _build_bm25_index(self, documents: List[Document] = None):
        """
        构建 BM25 倒排索引

        使用 jieba 中文分词对每个文档做 tokenization，
        然后创建 BM25Okapi 实例。

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
        """确保 BM25 索引已构建（懒加载触发器）"""
        if self._bm25_index is None and self._vector_store is not None:
            self._build_bm25_index()

    # ══════════════════════════════════════════
    #  Cross-Encoder 重排序
    # ══════════════════════════════════════════

    @property
    def reranker(self):
        """
        懒加载 Cross-Encoder 重排序模型

        模型: cross-encoder/ms-marco-MiniLM-L-6-v2
        依赖: sentence-transformers 库（可选，未安装时静默跳过）
        """
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

    # ══════════════════════════════════════════
    #  相似度搜索（纯稠密）
    # ══════════════════════════════════════════

    def similarity_search(self, query: str, k: int = None,
                          score_threshold: float = None) -> List[Document]:
        """
        纯稠密向量检索（兼容旧接口）

        内部调用 ChromaDB 的 similarity_search_with_relevance_scores()，
        然后按 score_threshold 过滤。

        Args:
            query: 查询文本
            k: 返回文档数（默认 APP_CONFIG.retrieval_k）
            score_threshold: 相似度阈值（默认 APP_CONFIG.retrieval_score_threshold）

        Returns:
            过滤后的 Document 列表
        """
        if k is None:
            k = APP_CONFIG.retrieval_k
        if score_threshold is None:
            score_threshold = APP_CONFIG.retrieval_score_threshold

        if not self._vector_store:
            logger.warning("⚠️ ChromaDB 未加载，无法执行搜索")
            return []

        try:
            results = self._vector_store.similarity_search_with_relevance_scores(query, k=k)
            filtered = [doc for doc, score in results if score >= score_threshold]
            logger.info(f"🔍 稠密检索: 共 {len(results)} 个，过滤后 {len(filtered)} 个")
            return filtered
        except Exception as e:
            logger.exception(f"❌ 搜索失败（query前50字: {query[:50]}）")
            return []

    # ══════════════════════════════════════════
    #  混合检索 + 重排序（主入口）
    # ══════════════════════════════════════════

    def hybrid_search(self, query: str, k: int = None) -> List[Document]:
        """
        混合检索 (BM25 + Dense + RRF) → 可选 Cross-Encoder 重排序 → 返回 Top-K

        流水线:
          Step 1: ChromaDB 稠密检索 Top-N（N = hybrid_prefetch_k）
                  → 存入 all_docs_dict，格式: (doc, dense_score, 0.0, dense_rank)
          Step 2: BM25 稀疏检索 Top-N
                  → BM25 独有的文档存入 all_docs_dict: (doc, 0.0, bm25_score, None)
                  → 稠密也命中的文档: 更新 bm25_score，保持 dense_rank
          Step 3: RRF 融合 — 每个文档得分 = 1/(rrf_k + dense_rank+1) + 1/(rrf_k + bm25_rank+1)
          Step 4: Cross-Encoder 重排序（若启用）
          Step 5: 返回 Top-K

        RRF 注意事项:
          - 稠密 rank 由 similarity_search_with_relevance_scores 返回的排序位置决定
          - BM25 rank 由 BM25Okapi.get_scores 得分排序决定
          - BM25 独有文档的 dense_rank = None，RRF 不贡献稠密分数
          - _bm25_rank_map 缓存了 doc_id → bm25_rank 的映射，避免 O(n) 遍历查找
        """
        if k is None:
            k = APP_CONFIG.retrieval_k

        if not self._vector_store:
            logger.warning("⚠️ ChromaDB 未加载，无法执行检索")
            return []

        prefetch_k = APP_CONFIG.hybrid_prefetch_k
        all_docs_dict = {}  # doc_id → (Document, dense_score, bm25_score, dense_rank)
        seen_ids = set()

        # ── Step 1: 稠密检索 ──
        try:
            dense_results = self._vector_store.similarity_search_with_relevance_scores(query, k=prefetch_k)
            for rank, (doc, score) in enumerate(dense_results):
                doc_id = self._doc_id(doc)
                all_docs_dict[doc_id] = (doc, score, 0.0, rank)
                seen_ids.add(doc_id)
            logger.info(f"🔍 稠密检索完成: {len(dense_results)} 个结果")
        except Exception as e:
            logger.warning(f"⚠️ 稠密检索失败: {e}")

        # ── Step 2: BM25 稀疏检索 ──
        self._ensure_bm25()
        if self._bm25_index and self._bm25_docs:
            try:
                query_tokens = list(jieba.cut(query))
                bm25_scores = self._bm25_index.get_scores(query_tokens)
                # 取 Top-N 的 BM25 结果索引
                bm25_ranked = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True
                )[:prefetch_k]

                for rank, idx in enumerate(bm25_ranked):
                    doc_id, text, meta = self._bm25_docs[idx]
                    if doc_id not in seen_ids:
                        # BM25 独有文档：dense_rank = None 避免 RRF 重复计分
                        bm25_doc = Document(page_content=text, metadata=meta)
                        all_docs_dict[doc_id] = (bm25_doc, 0.0, float(bm25_scores[idx]), None)
                        seen_ids.add(doc_id)
                    else:
                        existing = all_docs_dict[doc_id]
                        all_docs_dict[doc_id] = (existing[0], existing[1], float(bm25_scores[idx]), existing[3])

                logger.info(f"🔍 BM25 检索完成: {len(bm25_ranked)} 个结果")
            except Exception as e:
                logger.warning(f"⚠️ BM25 检索失败: {e}")

        if not all_docs_dict:
            logger.warning("⚠️ 检索结果为空")
            return []

        # ── Step 3: RRF 融合（带 bm25_rank 缓存，优化 O(n²) 查找） ──
        rrf_k_const = APP_CONFIG.rrf_k
        # 构建 doc_id → bm25_rank 的哈希表缓存，避免逐文档遍历 _bm25_docs
        bm25_rank_map = {did: i for i, (did, _, _) in enumerate(self._bm25_docs)} if self._bm25_docs else {}

        scored_docs = []
        for doc_id, (doc, dense_score, bm25_score, dense_rank) in all_docs_dict.items():
            rrf = 0.0
            if dense_rank is not None:
                rrf += 1.0 / (rrf_k_const + dense_rank + 1)
            bm25_rank = bm25_rank_map.get(doc_id)
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
                scored_docs = sorted(zip([d for d, _ in scored_docs], ce_scores), key=lambda x: x[1], reverse=True)
                logger.info(f"🎯 Cross-Encoder 重排序完成")
            except Exception as e:
                logger.warning(f"⚠️ Cross-Encoder 重排序失败: {e}")

        # ── Step 5: 返回 Top-K ──
        final_docs = [doc for doc, _ in scored_docs[:k]]
        logger.info(f"📚 最终返回 {len(final_docs)} 个文档")
        return final_docs

    # ══════════════════════════════════════════
    #  工具方法
    # ══════════════════════════════════════════

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """生成文档的唯一标识（source + chunk_index）"""
        return f"{doc.metadata.get('source', '')}_{doc.metadata.get('chunk_index', '')}"

    def get_retriever(self, k: int = None, score_threshold: float = None):
        """
        获取 LangChain Retriever 对象（兼容旧接口）

        返回 ChromaDB 的 as_retriever() 包装，支持 LangChain 原生链调用。
        """
        if not self._vector_store:
            return None
        return self._vector_store.as_retriever(
            search_kwargs={
                "k": k or APP_CONFIG.retrieval_k,
                "score_threshold": score_threshold or APP_CONFIG.retrieval_score_threshold
            }
        )
