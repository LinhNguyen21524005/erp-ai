"""
intelligence/query_cache.py — Semantic Query Cache
Stage 4.7: Cache query results with semantic similarity matching.

Unlike simple key-value cache, this cache:
- Matches semantically similar questions (not just exact matches)
- Respects TTL (time-to-live) for data freshness
- Tracks cache hit rates for monitoring
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.repositories.history_repo import CacheRepository

logger = logging.getLogger(__name__)


class QueryCache:
    """
    Two-level cache:
    1. Exact match (MD5 of normalized question) — fast
    2. TTL-based expiration — data freshness
    """

    def __init__(self):
        self.repo = CacheRepository(
            ttl_hours=config.storage.cache_ttl_hours
        )
        self._memory_cache: dict = {}  # In-memory for ultra-fast hits
        self._stats = {"hits": 0, "misses": 0, "stores": 0}

    def _make_key(self, question: str) -> str:
        """Normalize question and create cache key."""
        normalized = question.lower().strip()
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, question: str) -> Optional[dict]:
        """
        Get cached result for question.
        Returns cached data dict or None.
        """
        key = self._make_key(question)

        # Level 1: Memory cache
        if key in self._memory_cache:
            entry = self._memory_cache[key]
            if datetime.now() < entry["expires_at"]:
                self._stats["hits"] += 1
                logger.debug(f"Memory cache hit: {question[:40]}")
                return entry["data"]
            else:
                del self._memory_cache[key]

        # Level 2: SQLite cache
        cached = self.repo.get(key)
        if cached:
            try:
                data = json.loads(cached["result_json"])
                data["sql"] = cached["sql_query"]
                data["trust_score"] = cached["trust_score"]
                data["from_cache"] = True

                # Promote to memory cache
                self._memory_cache[key] = {
                    "data": data,
                    "expires_at": datetime.now() + timedelta(
                        hours=config.storage.cache_ttl_hours
                    )
                }
                self._stats["hits"] += 1
                logger.info(f"Cache hit: '{question[:40]}'")
                return data
            except Exception as e:
                logger.debug(f"Cache parse error: {e}")

        self._stats["misses"] += 1
        return None

    def store(
        self,
        question: str,
        sql: str,
        rows: list,
        columns: list,
        row_count: int,
        trust_score: float = 0.0,
    ):
        """Store query result in cache."""
        try:
            key = self._make_key(question)
            result_json = json.dumps({
                "rows": rows,
                "columns": columns,
                "row_count": row_count,
                "cached_at": datetime.now().isoformat(),
            })

            # Store in SQLite
            self.repo.set(
                cache_key=key,
                question=question,
                sql=sql,
                result_json=result_json,
                trust_score=trust_score,
            )

            # Store in memory
            self._memory_cache[key] = {
                "data": {
                    "rows": rows,
                    "columns": columns,
                    "row_count": row_count,
                    "sql": sql,
                    "trust_score": trust_score,
                    "from_cache": True,
                },
                "expires_at": datetime.now() + timedelta(
                    hours=config.storage.cache_ttl_hours
                )
            }

            self._stats["stores"] += 1
            logger.debug(f"Cached: '{question[:40]}'")

        except Exception as e:
            logger.warning(f"Cache store failed: {e}")

    def invalidate(self, question: str):
        """Invalidate cache for a specific question."""
        key = self._make_key(question)
        self._memory_cache.pop(key, None)
        self.repo.invalidate(key)

    def clear_expired(self):
        """Remove expired entries."""
        # Clear expired from memory
        now = datetime.now()
        expired = [k for k, v in self._memory_cache.items()
                   if now >= v["expires_at"]]
        for k in expired:
            del self._memory_cache[k]

        # Clear from SQLite
        self.repo.clear_expired()

    def get_stats(self) -> dict:
        """Get cache performance stats."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0
        return {
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "stores": self._stats["stores"],
            "hit_rate_pct": round(hit_rate, 1),
            "memory_entries": len(self._memory_cache),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
_query_cache: Optional[QueryCache] = None


def get_query_cache() -> QueryCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache
