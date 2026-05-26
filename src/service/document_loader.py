"""
文档加载模块
从 data/disease/ 目录读取 .txt/.md/.mdx 文件并分块

被 system_initializer.py 调用:
  DocumentLoader → load_and_split_documents()
    → VectorStoreManager.create_vector_store()
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


class DocumentLoader:
    """
    文档加载 & 分块器

    职责:
      - 扫描目录下支持的文档格式
      - 读取内容并构建 LangChain Document 对象
      - 按配置的 chunk_size / chunk_overlap 分块
    """

    def __init__(self, data_dir: str = None):
        """
        Args:
            data_dir: 文档目录，默认 APP_CONFIG.disease_dir
        """
        self.data_dir = data_dir if data_dir is not None else APP_CONFIG.disease_dir
        self.text_splitter = self._create_text_splitter()

        logger.info(f"📁 文档目录: {self.data_dir}")

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        分块策略: 按段落 → 换行 → 句号 → 标点，逐级递归拆分，
        保证语义单元的完整性。
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG.chunk_size,
            chunk_overlap=APP_CONFIG.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """
        加载 data_dir 下所有支持格式的文档

        Returns:
            原始 Document 列表（未分块）
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
        对每个 Document 执行分块，保留原始元数据并附加块索引

        Args:
            documents: 已加载的原始文档列表

        Returns:
            分块后的 Document 列表
        """
        if not documents:
            return []

        split_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                split_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        'chunk_size': len(chunk)
                    }
                )
                split_docs.append(split_doc)

        logger.info(f"✂️ 文档分块完成，总共 {len(split_docs)} 个块")
        return split_docs

    def load_and_split_documents(self) -> List[Document]:
        """加载 + 分块一步到位"""
        documents = self.load_documents()
        return self.split_documents(documents)
