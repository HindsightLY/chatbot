import os
from typing import List
from langchain_core.documents import Document


class DocumentLoader:
    def __init__(self, data_dir: str = None):
        """
        初始化文档加载器
        如果 data_dir 为 None，则自动定位到项目根目录下的 data/disease
        """
        if data_dir is None:
            # 获取当前脚本所在目录 → 上两级是项目根目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)  # src → 上一级是根目录
            self.data_dir = os.path.join(project_root, "data", "disease")
        else:
            self.data_dir = data_dir

    def load_documents(self) -> List[Document]:
        documents = []

        # ✅ 关键：打印实际路径用于调试
        print(f"🔍 正在加载文档目录: {self.data_dir}")

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
                        print(f"✅ 加载: {disease_name}")
                except Exception as e:
                    print(f"⚠️ 跳过文件 {filename}: {e}")

        print(f"📊 共加载 {len(documents)} 个疾病文档")
        return documents
