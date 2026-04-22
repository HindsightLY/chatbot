"""
文档加载模块
负责从指定目录加载文档文件
"""
import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.app_config import APP_CONFIG
from src.logger_config import logger


class DocumentLoader:
    """
    文档加载器
    负责加载和分块文档
    """

    def __init__(self, data_dir: str = None):
        """
        初始化文档加载器

        Args:
            data_dir: 文档目录路径，如果为None则使用配置文件中的默认路径
        """
        # 使用配置中的路径，如果传入了路径则覆盖
        self.data_dir = data_dir if data_dir is not None else APP_CONFIG.disease_dir
        self.text_splitter = self._create_text_splitter()

        logger.info(f"📁 文档目录: {self.data_dir}")

    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        创建文本分块器

        Returns:
            RecursiveCharacterTextSplitter: 文本分块器实例
        """
        return RecursiveCharacterTextSplitter(
            chunk_size=APP_CONFIG.chunk_size,
            chunk_overlap=APP_CONFIG.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )

    def load_documents(self) -> List[Document]:
        """
        加载所有文档

        Returns:
            List[Document]: 文档列表
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

                    # 创建文档对象
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
        将文档分块

        Args:
            documents: 文档列表

        Returns:
            List[Document]: 分块后的文档列表
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
        """
        加载并分块所有文档

        Returns:
            List[Document]: 分块后的文档列表
        """
        documents = self.load_documents()
        return self.split_documents(documents)


# 使用示例
if __name__ == "__main__":
    # 初始化文档加载器
    doc_loader = DocumentLoader()

    # 加载并分块文档
    split_docs = doc_loader.load_and_split_documents()

    print(f"📊 总共加载 {len(split_docs)} 个文档块")
    if split_docs:
        print(f"🔍 第一个块的内容: {split_docs[0].page_content[:200]}...")