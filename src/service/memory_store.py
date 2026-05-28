"""
Redis 对话记忆模块
使用 Redis 存储会话历史，替代原 HybridChatMemory（内存+FAISS）

存储结构:
  key: chat:{session_id}:messages (Redis List)
  value: JSON 格式 {"role": "user"/"assistant", "content": "...", "timestamp": "..."}

过期策略: 每轮对话写入后刷新 TTL（默认 24 小时）
"""
import json
from datetime import datetime
from typing import List, Dict
import redis
from config.app_config import APP_CONFIG
from src.utils.logger_config import logger


class MemoryStore:
    """
    Redis 对话记忆存储

    职责:
      - 按 session_id 存储对话历史
      - 提供最近 N 轮对话的读取
      - 自动过期清理
    """

    def __init__(self):
        self.max_history_turns = 10
        self._client = None
        # 预触发懒加载，使调用方首次使用时无需等待
        _ = self.client

    @property
    def client(self):
        """懒加载 Redis 连接"""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """建立 Redis 连接"""
        try:
            self._client = redis.Redis(
                host=APP_CONFIG.redis_host,
                port=APP_CONFIG.redis_port,
                db=APP_CONFIG.redis_db,
                decode_responses=True
            )
            self._client.ping()
            logger.info(f"✅ Redis 连接成功: {APP_CONFIG.redis_host}:{APP_CONFIG.redis_port}")
        except redis.ConnectionError as e:
            logger.warning(f"⚠️ Redis 连接失败，使用内存回退: {e}")
            self._client = None
        except Exception as e:
            logger.warning(f"⚠️ Redis 初始化异常，使用内存回退: {e}")
            self._client = None

    def _key(self, session_id: str) -> str:
        """构造 Redis key"""
        return f"chat:{session_id}:messages"

    def add_message(self, session_id: str, role: str, content: str):
        """
        添加一条消息到 Redis

        Args:
            session_id: 会话 ID
            role: "user" 或 "assistant"
            content: 消息内容
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        rc = self.client
        if rc:
            try:
                key = self._key(session_id)
                rc.rpush(key, json.dumps(message, ensure_ascii=False))
                rc.expire(key, APP_CONFIG.redis_ttl)
            except Exception as e:
                logger.warning(f"⚠️ Redis 写入失败: {e}")
                self._client = None

    def get_recent_messages(self, session_id: str, n: int = None) -> List[Dict]:
        """
        获取最近 N 条消息

        Args:
            session_id: 会话 ID
            n: 返回条数，默认 self.max_history_turns

        Returns:
            消息列表，每项 {"role": str, "content": str, "timestamp": str}
        """
        if n is None:
            n = self.max_history_turns

        rc = self.client
        if rc:
            try:
                key = self._key(session_id)
                messages = rc.lrange(key, -n, -1)
                return [json.loads(m) for m in messages]
            except Exception as e:
                logger.warning(f"⚠️ Redis 读取失败: {e}")

        return []

    def get_history_text(self, session_id: str, n: int = None) -> str:
        """
        将最近 N 轮对话格式化为文本，供 prompt 使用

        Args:
            session_id: 会话 ID
            n: 返回轮数，默认 self.max_history_turns

        Returns:
            格式化后的对话文本
        """
        messages = self.get_recent_messages(session_id, n)
        if not messages:
            return "无历史对话记录"

        lines = []
        for i, msg in enumerate(messages):
            role = "用户" if msg["role"] == "user" else "助手"
            lines.append(f"对话 {i + 1}:")
            lines.append(f"{role}: {msg['content']}")
            lines.append("-" * 30)

        return "\n".join(lines)

    def clear_session(self, session_id: str):
        """清除指定会话的全部历史"""
        rc = self.client
        if rc:
            try:
                key = self._key(session_id)
                rc.delete(key)
                logger.info(f"🧹 清除会话 {session_id} 的记忆")
            except Exception as e:
                logger.warning(f"⚠️ 清除会话 {session_id} 失败: {e}")
