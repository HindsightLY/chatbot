"""
Redis 对话记忆模块 — 按 session_id 存储/读取对话历史

存储结构:
  key:  chat:{session_id}:messages (Redis List)
  value: {"role": "user"|"assistant", "content": str, "timestamp": str}  (JSON)

过期策略: 每次 rpush 后重新 expire，TTL 来自 APP_CONFIG.redis_ttl（默认 86400s）
回退策略: Redis 不可用时静默降级（client 返回 None），不影响主流程
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
      - 按 session_id 存储/追加对话历史（add_message）
      - 读取最近 N 轮对话（get_recent_messages / get_history_text）
      - 清除指定会话（clear_session）
      - Redis 不可用时静默降级，不阻塞业务流程
    """

    def __init__(self):
        self.max_history_turns = 10
        self._client = None
        _ = self.client

    @property
    def client(self):
        """Redis 连接的懒加载属性，首次访问时调用 _connect"""
        if self._client is None:
            self._connect()
        return self._client

    def _connect(self):
        """
        建立 Redis 连接。

        连接失败时不抛出异常，client 保持 None，后续写入/读取均静默跳过。
        """
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
        return f"chat:{session_id}:messages"

    def add_message(self, session_id: str, role: str, content: str):
        """
        添加一条消息到 Redis List（尾部 rpush），刷新 TTL。

        Args:
            session_id: 会话 ID
            role:       "user" 或 "assistant"
            content:    消息文本
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
        获取最近 N 条消息（从尾部倒序取 N 条）。

        Args:
            session_id: 会话 ID
            n:          返回条数，默认 self.max_history_turns

        Returns:
            [{"role": str, "content": str, "timestamp": str}, ...]
            列表顺序与存储顺序一致（最早 → 最晚）
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
        将最近 N 轮对话格式化为文本，供 Agent prompt 使用。

        Args:
            session_id: 会话 ID
            n:          返回轮数，默认 self.max_history_turns

        Returns:
            格式化后的对话文本，每轮包含 role + content + 分隔线
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
        """
        清除指定会话的全部历史消息。

        Args:
            session_id: 会话 ID
        """
        rc = self.client
        if rc:
            try:
                key = self._key(session_id)
                rc.delete(key)
                logger.info(f"🧹 清除会话 {session_id} 的记忆")
            except Exception as e:
                logger.warning(f"⚠️ 清除会话 {session_id} 失败: {e}")
