"""
文档加载器模块
负责从指定目录加载医疗文档
"""
import os
from typing import List
from langchain_core.documents import Document
from .logger_config import logger


class DocumentLoader:
    """
    文档加载器类
    用于从指定目录加载TXT格式的医疗文档
    """

    def __init__(self, data_dir: str = None):
        """
        初始化文档加载器

        Args:
            data_dir: 文档目录路径，如果为None则自动定位到项目根目录下的data/disease
        """
        if data_dir is None:
            # 获取当前脚本所在目录 → 上两级是项目根目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # src → 上一级是根目录
            self.data_dir = os.path.join(project_root, "data", "disease")
        else:
            self.data_dir = data_dir

    def load_documents(self) -> List[Document]:
        """
        加载文档目录中的所有TXT文件

        Returns:
            List[Document]: 加载的文档列表

        Raises:
            FileNotFoundError: 当文档目录不存在时抛出异常
        """
        documents = []

        # 打印实际路径用于调试
        logger.info(f"🔍 正在加载文档目录: {self.data_dir}")

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"❌ 文档目录不存在！请检查路径：{self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if filename.lower().endswith('.txt'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    if content:
                        disease_name = os.path.splitext(filename)[0]
                        doc = Document(
                            page_content=content,
                            metadata={
                                "filename": filename,
                                "disease": disease_name,
                                "source": file_path,
                                "category": "medical_disease"
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✅ 加载: {disease_name}")
                except Exception as e:
                    logger.error(f"❌ 加载文件 {file_path} 时出错: {e}")

        logger.info(f"📊 共加载 {len(documents)} 个疾病文档")
        return documents