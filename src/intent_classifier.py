"""
意图分类器模块
使用大语言模型对用户输入进行意图分类
"""
import json
from langchain_ollama import OllamaLLM
from .logger_config import logger


class IntentClassifier:
    """
    意图分类器
    使用大语言模型对用户输入进行意图分类
    """

    def __init__(self, model_name: str = "qwen2.5:7b"):
        """
        初始化意图分类器

        Args:
            model_name: 使用的LLM模型名称
        """
        self.llm = OllamaLLM(
            model=model_name,
            base_url="http://localhost:11434",
            # temperature=0.0 用于分类任务，追求确定性
        )

        # 定义意图类别（可根据项目需求修改）
        self.intent_schema = {
            "intents": [
                {
                    "name": "medical_inquiry",
                    "description": "用户询问疾病、症状、药物、治疗方法、病因等医疗相关问题",
                    "keywords": ["病", "症状", "药", "怎么治", "原因", "医生"]
                },
                {
                    "name": "chat_general",
                    "description": "用户进行闲聊、打招呼、询问AI是谁、或者无关医疗的日常对话",
                    "keywords": ["你好", "你是谁", "聊天", "名字", "干嘛"]
                },
                {
                    "name": "system_query",
                    "description": "用户询问关于本系统、程序、代码、功能相关的问题",
                    "keywords": ["功能", "系统", "怎么用", "帮助"]
                }
            ]
        }

    def classify(self, query: str) -> str:
        """
        对用户输入进行意图分类

        Args:
            query: 用户输入的查询文本

        Returns:
            str: 分类结果，可能的值包括 'medical_inquiry', 'chat_general', 'system_query', 'unknown'
        """
        # 设计一个结构化的 Prompt，要求模型返回 JSON
        prompt = f"""
        你是一个意图分类器。请严格分析用户的输入，并从预定义的 Schema 中选择最匹配的一个意图。
        请只返回 JSON 对象，不要包含任何其他解释文字。

        <Schema>
        {json.dumps(self.intent_schema, ensure_ascii=False, indent=2)}
        </Schema>

        <Rules>
        1. 仔细分析用户输入的语义，而不仅仅是关键词匹配。
        2. 如果用户同时提到了医疗和闲聊（例如："你好，我头疼"），优先识别为医疗意图。
        3. 必须返回一个标准的 JSON 对象，包含 "intent" 字段。
        </Rules>

        <Example>
        用户输入: "扁桃体发炎怎么办？"
        返回: {{"intent": "medical_inquiry"}}
        </Example>

        <User_Input>
        {query}
        </User_Input>

        请开始分类:
        """

        try:
            result = self.llm.invoke(prompt)
            # 解析 JSON (这里简单处理，实际项目中建议用更健壮的 JSON 解析)
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
                return data.get("intent", "unknown")
            else:
                return "unknown"
        except Exception as e:
            logger.info(f"意图识别错误: {e}")
            return "unknown"