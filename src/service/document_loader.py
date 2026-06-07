"""
文档加载与分块模块 — 从 data/disease/ 读取医学文档并分块

分块策略（二选一，由 APP_CONFIG.use_semantic_chunking 控制）:
  - 语义分块（默认）: 句子级拆分 → 向量化 → 按余弦距离间隙合并，自动保持话题内聚
  - 固定分块（回退）: RecursiveCharacterTextSplitter (chunk_size=500, overlap=100)

调用链:
  system_initializer → DocumentLoader.load_and_split_documents()
    → SemanticChunker.split_documents() 或 RecursiveCharacterTextSplitter
    → VectorStoreManager.create_vector_store()
"""
import os
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


class SemanticChunker:
    """
    语义分块器 — 基于文本嵌入余弦距离的话题边界检测

    原理:
      1. 先用 RecursiveCharacterTextSplitter 将文本拆为极小单元（句子级，chunk_size=50）
      2. 对每个单元计算嵌入向量（OllamaEmbeddings）
      3. 计算相邻单元的余弦距离（1 - cosine_similarity）
      4. 在距离超过百分位阈值的位置断开，合并为语义块

    回退:
      嵌入模型不可用时自动降级为固定分块（RecursiveCharacterTextSplitter）
    """

    def __init__(self, embeddings: OllamaEmbeddings = None):
        self.min_size = APP_CONFIG.semantic_chunk_min_size
        self.max_size = APP_CONFIG.semantic_chunk_max_size
        self.breakpoint_percentile = APP_CONFIG.semantic_chunk_breakpoint_percentile

        self.pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        try:
            self.embeddings = embeddings or OllamaEmbeddings(
                model=APP_CONFIG.embedding_model_name,
                base_url=APP_CONFIG.llm_base_url
            )
        except Exception as e:
            logger.warning(f"⚠️ 语义分块嵌入模型加载失败，降级为固定分块: {e}")
            self.embeddings = None

    def split_text(self, text: str) -> List[str]:
        """
        对一段文本执行语义分块。

        流程:
          1. 预分块为句子级单元
          2. 编码各单元为嵌入向量
          3. 计算相邻向量余弦距离 → 百分位阈值 → 确定断点
          4. 按断点合并单元，再按 min_size / max_size 二次合并

        Args:
            text: 原始文本

        Returns:
            语义块列表
        """
        if not self.embeddings:
            return self._fixed_size_fallback(text)

        units = self.pre_splitter.split_text(text)
        units = [u.strip() for u in units if len(u.strip()) >= 5]

        if len(units) <= 1:
            return units if units else [text]

        try:
            unit_embeddings = self.embeddings.embed_documents(units)
        except Exception as e:
            logger.warning(f"⚠️ 语义分块嵌入计算失败，降级为固定分块: {e}")
            return self._fixed_size_fallback(text)

        distances = []
        for i in range(len(unit_embeddings) - 1):
            sim = self._cosine_similarity(unit_embeddings[i], unit_embeddings[i + 1])
            distances.append(1.0 - sim)

        if not distances:
            return self._merge_units(units, []) if units else [text]

        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        breakpoints = [i for i, d in enumerate(distances) if d > threshold]

        merged = self._merge_units(units, breakpoints)
        if not merged:
            return [text]
        return merged

    def _merge_units(self, units: List[str], breakpoints: List[int]) -> List[str]:
        """
        按断点将句子单元合并为块，再按 min_size / max_size 做二次合并。

        Args:
            units:       句子单元列表
            breakpoints: 话题断点索引列表

        Returns:
            合并后的文本块列表
        """
        chunks = []
        start = 0
        for bp in breakpoints:
            chunk = "".join(units[start:bp + 1]).strip()
            if chunk:
                chunks.append(chunk)
            start = bp + 1
        if start < len(units):
            remaining = "".join(units[start:]).strip()
            if remaining:
                chunks.append(remaining)

        merged = []
        current = ""
        for chunk in chunks:
            if len(current) + len(chunk) <= self.max_size:
                current += chunk
            else:
                if current and len(current) >= self.min_size:
                    merged.append(current)
                elif current:
                    merged.append(current)
                current = chunk
        if current:
            if merged and len(current) < self.min_size:
                merged[-1] += current
            else:
                merged.append(current)

        return [c.strip() for c in merged if c.strip()]

    def _fixed_size_fallback(self, text: str) -> List[str]:
        """嵌入模型不可用时降级为固定大小分块"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG.chunk_size,
            chunk_overlap=APP_CONFIG.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        return splitter.split_text(text)

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """计算两个向量的余弦相似度，避免除零"""
        a_arr = np.array(a, dtype=np.float64)
        b_arr = np.array(b, dtype=np.float64)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对 Document 列表执行语义分块，保留并增强元数据。

        Args:
            documents: 原始 Document 列表

        Returns:
            分块后的 Document 列表，每块含 chunk_index/total_chunks/chunk_size/chunk_method
        """
        split_docs = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                        'chunk_method': 'semantic'
                    }
                )
                split_docs.append(split_doc)

        logger.info(f"✂️ 语义分块完成，总共 {len(split_docs)} 个块")
        return split_docs


class DocumentLoader:
    """
    文档加载 & 分块器

    职责:
      - 扫描 data_dir 下 .txt/.md/.mdx 文件，构建 Document 对象
      - 按 APP_CONFIG.use_semantic_chunking 选择分块策略
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir if data_dir is not None else APP_CONFIG.disease_dir
        logger.info(f"📁 文档目录: {self.data_dir}")

    def load_documents(self) -> List[Document]:
        """
        加载 data_dir 下所有支持格式的文档。

        支持的格式: .txt, .md, .mdx
        每个文档附加元数据: source, file_path, file_type

        Returns:
            Document 列表（空目录或异常时返回空列表）
        """
        if not os.path.exists(self.data_dir):
            logger.warning(f"⚠️ 文档目录不存在: {self.data_dir}")
            return []

        documents = []
        supported_extensions = {'.txt', '.md', '.mdx'}

        for filename in os.listdir(self.data_dir):
            if any(filename.endswith(ext) for ext in supported_extensions):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': filename,
                            'file_path': file_path,
                            'file_type': os.path.splitext(filename)[1]
                        }
                    )
                    documents.append(doc)
                    logger.info(f"✅ 加载文档: {filename}")

                except Exception as e:
                    logger.error(f"❌ 加载文档失败 {filename}: {e}")

        logger.info(f"📊 总共加载 {len(documents)} 个文档")
        return documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        对 Document 列表执行分块。

        策略选择:
          - APP_CONFIG.use_semantic_chunking = True → SemanticChunker（语义分块）
          - False → RecursiveCharacterTextSplitter（固定分块）

        Args:
            documents: 原始 Document 列表

        Returns:
            分块后的 Document 列表，每块含 chunk_index/total_chunks/chunk_size/chunk_method
        """
        if not documents:
            return []

        if APP_CONFIG.use_semantic_chunking:
            return SemanticChunker().split_documents(documents)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG.chunk_size,
            chunk_overlap=APP_CONFIG.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

        split_docs = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_docs.append(Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk),
                        'chunk_method': 'fixed'
                    }
                ))

        logger.info(f"✂️ 固定分块完成，总共 {len(split_docs)} 个块")
        return split_docs

    def load_and_split_documents(self) -> List[Document]:
        """加载 + 分块一步到位"""
        documents = self.load_documents()
        return self.split_documents(documents)
