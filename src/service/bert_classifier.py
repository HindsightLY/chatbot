"""
BERT 意图分类器 — 基于 sentence-transformers 的多头语义匹配

原理:
  1. 为每个意图预定义若干示例问句（INTENT_EXAMPLES）
  2. 使用多语言 sentence-transformer 模型将所有示例编码为嵌入向量
  3. 用户查询编码后与所有意图的示例向量计算余弦相似度
  4. 取最相似的意图作为分类结果（得分低于阈值时返回 unknown）

优势: 相比 LLM 分类（~2s），BERT 分类仅 ~50ms，且可完全离线运行
回退: sentence-transformers 未安装或模型加载失败 → 返回 None，调用方降级至 LLM
"""
import os
import numpy as np
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger

# 强制 HuggingFace 从国内镜像加载，且仅使用本地缓存（不联网）
_HF_MIRROR = APP_CONFIG.hf_mirror or os.getenv("HF_ENDPOINT") or os.getenv("HF_MIRROR")
if _HF_MIRROR:
    os.environ["HF_ENDPOINT"] = _HF_MIRROR
os.environ["HF_HUB_OFFLINE"] = "1"

# 每个意图的示例问句（覆盖常见表述，新增意图只需在此添加示例）
INTENT_EXAMPLES = {
    "medical_inquiry": [
        "感冒了怎么办", "头疼是什么原因", "发烧吃什么药",
        "高血压怎么治疗", "咳嗽怎么办", "胃痛是什么病",
        "糖尿病症状有哪些", "扁桃体发炎怎么治", "皮肤过敏怎么办",
        "失眠怎么调理", "腰痛是什么原因", "拉肚子吃什么药",
        "眼睛发红怎么办", "关节炎怎么治疗", "嗓子疼怎么办",
    ],
    "chat_general": [
        "你好", "今天天气怎么样", "北京天气", "上海今天气温",
        "你叫什么名字", "你是做什么的", "你会什么", "讲个笑话",
        "今天星期几", "现在几点", "再见", "谢谢", "你真聪明", "你能帮我什么",
    ],
    "system_query": [
        "这个系统有什么用", "你怎么工作的", "你能做什么",
        "你们的功能有哪些", "怎么使用这个系统", "这是什么软件",
        "有哪些命令", "怎么用", "帮助", "功能有哪些",
    ],
}

# 各意图的示例问句向量已预编码，推理时直接做矩阵乘法，无需重复编码示例


class BertIntentClassifier:
    """
    基于 sentence-transformers 多头匹配的意图分类器

    将每个意图的示例问句预编码为嵌入向量，用户查询进入后计算与所有示例的余弦相似度，
    取最高分意图。低于 threshold 时返回 unknown，由上层降级至 LLM 分类。

    使用示例:
        classifier = BertIntentClassifier()
        intent = classifier.classify("头痛怎么办")  # → "medical_inquiry"
    """

    def __init__(self):
        self.model = None                      # SentenceTransformer 模型实例
        self.intent_embeddings = {}            # intent_name → ndarray [n_examples, 384]
        self.intent_labels = []                # 意图名称列表
        self._ready = False                    # 模型加载成功标记
        self._load_model()

    def _load_model(self):
        """
        加载 sentence-transformer 模型并预编码所有意图示例向量。

        预编码后的嵌入存入 self.intent_embeddings，推理时只需计算
        query_emb @ examples_emb.T 矩阵乘法即可获得所有意图得分，无需重复编码示例。
        """
        try:
            from sentence_transformers import SentenceTransformer
            model_name = APP_CONFIG.bert_model_name
            logger.info(f"📥 加载 BERT 分类模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            logger.info(f"✅ BERT 分类模型加载完成: {model_name}")

            for intent, examples in INTENT_EXAMPLES.items():
                if examples:
                    emb = self.model.encode(examples, normalize_embeddings=True)
                    self.intent_embeddings[intent] = emb
                else:
                    self.intent_embeddings[intent] = np.zeros((1, 384))

            self.intent_labels = list(INTENT_EXAMPLES.keys())
            self._ready = True
            logger.info(f"✅ BERT 分类器就绪，支持 {len(self.intent_labels)} 个意图")
        except ImportError:
            logger.warning("⚠️ sentence-transformers 未安装，BERT 分类器不可用")
            logger.warning("   如需启用: pip install sentence-transformers")
        except Exception as e:
            logger.warning(f"⚠️ BERT 分类器加载失败: {e}")

    def classify(self, query: str) -> str:
        """
        意图分类主入口

        计算逻辑:
          1. query_emb = model.encode(query)                    # [1, 384]
          2. for each intent: similarities = query_emb @ examples_emb.T  # [1, n_examples]
          3. best_score = max of all max_sim across intents
          4. if best_score < threshold → return "unknown"

        Args:
            query: 用户输入文本

        Returns:
            "medical_inquiry" | "chat_general" | "system_query" | "unknown" | None（模型未就绪）
        """
        if not self._ready or self.model is None:
            return None

        if not query or not query.strip():
            return "unknown"

        try:
            query_emb = self.model.encode(query, normalize_embeddings=True)

            best_intent = "unknown"
            best_score = -1.0

            for intent, examples_emb in self.intent_embeddings.items():
                similarities = query_emb @ examples_emb.T
                max_sim = float(similarities.max())
                if max_sim > best_score:
                    best_score = max_sim
                    best_intent = intent

            if best_score < APP_CONFIG.bert_classifier_threshold:
                logger.debug(f"BERT 分类得分 {best_score:.3f} < 阈值 {APP_CONFIG.bert_classifier_threshold}，返回 unknown")
                return "unknown"

            logger.debug(f"BERT 分类: {best_intent} (score={best_score:.3f})")
            return best_intent

        except Exception as e:
            logger.warning(f"BERT 分类失败: {e}")
            return None
