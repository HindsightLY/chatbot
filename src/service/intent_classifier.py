"""
意图分类器模块
双引擎分类器：BERT 优先（~50ms），LLM 兜底（~2s）

分类结果:
  medical_inquiry → MedicalChatbot (RAG)
  chat_general   → ToolManager (天气/闲聊)
  system_query   → ToolManager (通用 LLM)
  unknown        → 兜底走 RAG

引擎选择:
  1. BertIntentClassifier  — sentence-transformers 多头匹配（快速，可离线）
  2. IntentClassifier      — OllamaLLM JSON prompt（稳重，依赖 Ollama）
  3. 两者均不可用 → keyword 规则兜底
"""
import json
from langchain_ollama import OllamaLLM
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


class IntentClassifier:
    """
    双引擎意图分类器

    使用 BERT(sentence-transformers) 优先，OllamaLLM 兜底，
    两者均不可用时使用关键词规则匹配。
    """

    def __init__(self, model_name: str = "qwen2.5:7b"):
        self.llm = OllamaLLM(
            model=model_name,
            base_url=APP_CONFIG.llm_base_url,
        )

        self.bert = None
        if APP_CONFIG.use_bert_classifier:
            try:
                from src.service.bert_classifier import BertIntentClassifier
                self.bert = BertIntentClassifier()
            except Exception as e:
                logger.warning(f"⚠️ BERT 分类器实例化失败: {e}")

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
                    "keywords": ["你好", "你是谁", "聊天", "名字", "干嘛", "天气", "日期", "时间"]
                },
                {
                    "name": "system_query",
                    "description": "用户询问关于本系统、程序、代码、功能相关的问题",
                    "keywords": ["功能", "系统", "怎么用", "帮助"]
                }
            ]
        }

    # ── 关键词规则兜底 ──

    def _keyword_classify(self, query: str) -> str:
        """关键词规则分类（最轻量兜底）"""
        q = query.lower()
        medical_kw = ["病", "症状", "药", "怎么治", "原因", "医生", "医院",
                       "痛", "疼", "发烧", "咳嗽", "感冒", "过敏", "血压",
                       "糖", "炎", "肿", "痒", "吐", "泻"]
        weather_kw = ["天气", "温度", "下雨", "下雪", "刮风", "气温", "多少度"]
        system_kw = ["功能", "系统", "怎么用", "帮助", "命令", "操作"]

        weather_score = sum(1 for kw in weather_kw if kw in q)
        medical_score = sum(1 for kw in medical_kw if kw in q)
        system_score = sum(1 for kw in system_kw if kw in q)

        if medical_score >= 2:
            return "medical_inquiry"
        if weather_score >= 1:
            return "chat_general"
        if system_score >= 2:
            return "system_query"
        return "unknown"

    def classify(self, query: str) -> str:
        """
        意图分类主入口（双引擎）

        优先级:
          1. BERT (sentence-transformers) — 最快 ~50ms
          2. LLM (Ollama)                  — 稳重 ~2s
          3. 关键词规则                    — 兜底

        Returns:
            'medical_inquiry' | 'chat_general' | 'system_query' | 'unknown'
        """
        # ── 引擎 1: BERT ──
        if self.bert is not None:
            try:
                result = self.bert.classify(query)
                if result is not None:
                    logger.debug(f"BERT 分类: {result}")
                    return result
            except Exception as e:
                logger.debug(f"BERT 分类失败，降级到 LLM: {e}")

        # ── 引擎 2: LLM ──
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
            json_start = result.find("{")
            json_end = result.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                json_str = result[json_start:json_end]
                data = json.loads(json_str)
                intent = data.get("intent", "unknown")
                logger.debug(f"LLM 分类: {intent}")
                return intent
        except Exception as e:
            logger.info(f"LLM 分类失败: {e}")

        # ── 引擎 3: 关键词兜底 ──
        result = self._keyword_classify(query)
        logger.debug(f"关键词兜底分类: {result}")
        return result
