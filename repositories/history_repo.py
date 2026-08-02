"""
repositories/history_repo.py — Query History & Audit Repository
Replaces: storage.py (history functions)
Manages query history, audit logs, instructions, cache.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional

from erp42_analytics.core.database import get_sqlite
from erp42_analytics.core.models import AuditEntry, QueryStatus

logger = logging.getLogger(__name__)


class HistoryRepository:
    """Manages query history and business instructions."""

    def __init__(self):
        self.db = get_sqlite()

    # ─── Query History ────────────────────────────────────────────────────────

    def save_query(self, question: str, sql: str, success: bool,
                   exec_time: float = 0.0, row_count: int = 0,
                   confidence_score: float = 0.0,
                   tables_used: list = None):
        """Save query to history."""
        self.db.execute_write(
            """INSERT INTO query_history
               (question, sql_query, success, exec_time, row_count,
                confidence_score, tables_used)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (question, sql, int(success), exec_time, row_count,
             confidence_score, json.dumps(tables_used or []))
        )

    def get_recent(self, limit: int = 10) -> list:
        """Get recent query history."""
        return self.db.execute(
            """SELECT * FROM query_history
               ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        )

    def get_stats(self) -> dict:
        """Get system statistics."""
        total = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM query_history"
        )
        success = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM query_history WHERE success = 1"
        )
        tables = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM table_descriptions WHERE status = 'SCANNED'"
        )
        golden = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM golden_sql"
        )

        total_cnt = total["cnt"] if total else 0
        success_cnt = success["cnt"] if success else 0

        return {
            "total_queries": total_cnt,
            "successful_queries": success_cnt,
            "success_rate": round(success_cnt / total_cnt * 100, 1) if total_cnt > 0 else 0,
            "tables_scanned": tables["cnt"] if tables else 0,
            "golden_sql_count": golden["cnt"] if golden else 0,
        }

    # ─── Instructions ─────────────────────────────────────────────────────────

    def save_instruction(self, instruction: str):
        """Add business instruction."""
        self.db.execute_write(
            "INSERT INTO instructions (instruction) VALUES (?)",
            (instruction.strip(),)
        )

    def get_instructions(self) -> list:
        """Get all active instructions."""
        rows = self.db.execute(
            "SELECT instruction FROM instructions WHERE active = 1 ORDER BY id"
        )
        return [row["instruction"] for row in rows]

    # ─── Synonyms ─────────────────────────────────────────────────────────────

    def save_synonym(self, term: str, synonym: str,
                     domain: str = "general", confidence: float = 1.0):
        """Add synonym mapping."""
        self.db.execute_write(
            """INSERT OR IGNORE INTO synonyms
               (term, synonym, domain, confidence)
               VALUES (?, ?, ?, ?)""",
            (term.lower(), synonym.lower(), domain, confidence)
        )

    def get_synonyms(self, domain: str = None) -> dict:
        """Get synonym map: {term: [synonyms]}."""
        if domain:
            rows = self.db.execute(
                "SELECT term, synonym FROM synonyms WHERE domain = ?",
                (domain,)
            )
        else:
            rows = self.db.execute("SELECT term, synonym FROM synonyms")

        result = {}
        for row in rows:
            term = row["term"]
            if term not in result:
                result[term] = []
            result[term].append(row["synonym"])
        return result


class AuditRepository:
    """Manages audit logs for enterprise compliance."""

    def __init__(self):
        self.db = get_sqlite()

    def log(self, entry: AuditEntry):
        """Write audit entry."""
        self.db.execute_write(
            """INSERT INTO audit_log
               (timestamp, user_id, session_id, question, sql_generated,
                tables_accessed, status, trust_score, exec_time,
                row_count, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.timestamp.isoformat(),
                entry.user_id,
                entry.session_id,
                entry.question,
                entry.sql_generated,
                json.dumps(entry.tables_accessed),
                entry.status.value,
                entry.trust_score,
                entry.exec_time,
                entry.row_count,
                entry.error,
            )
        )

    def get_recent(self, limit: int = 50, user_id: str = None) -> list:
        """Get recent audit entries."""
        if user_id:
            return self.db.execute(
                """SELECT * FROM audit_log WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit)
            )
        return self.db.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )


class CacheRepository:
    """Query result cache for repeated queries."""

    def __init__(self, ttl_hours: int = 24):
        self.db = get_sqlite()
        self.ttl_hours = ttl_hours

    def get(self, cache_key: str) -> Optional[dict]:
        """Get cached result if not expired."""
        row = self.db.execute_one(
            """SELECT * FROM query_cache
               WHERE cache_key = ?
               AND expires_at > datetime('now')""",
            (cache_key,)
        )
        if row:
            # Increment hit count
            self.db.execute_write(
                "UPDATE query_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                (cache_key,)
            )
            logger.info(f"Cache hit: {cache_key[:30]}")
            return row
        return None

    def set(self, cache_key: str, question: str, sql: str,
            result_json: str, trust_score: float = 0.0):
        """Store result in cache."""
        expires = (datetime.now() + timedelta(hours=self.ttl_hours)).isoformat()
        self.db.execute_write(
            """INSERT OR REPLACE INTO query_cache
               (cache_key, question, sql_query, result_json,
                trust_score, expires_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cache_key, question, sql, result_json, trust_score, expires)
        )

    def invalidate(self, cache_key: str):
        """Remove specific cache entry."""
        self.db.execute_write(
            "DELETE FROM query_cache WHERE cache_key = ?",
            (cache_key,)
        )

    def clear_expired(self):
        """Remove expired cache entries."""
        deleted = self.db.execute_write(
            "DELETE FROM query_cache WHERE expires_at <= datetime('now')"
        )
        if deleted:
            logger.info(f"Cleared {deleted} expired cache entries")
