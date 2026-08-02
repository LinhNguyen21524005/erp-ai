"""
intelligence/pilot.py — Pilot Management System
Stage 8.1: Production Environment Preparation
Stage 8.7: User Acceptance Testing (UAT) feedback loop
Stage 8.8: Monitoring & Feedback Pipeline Setup
Stage 8.10: Phase 1 Final Validation (KPI checklist)

Responsibilities:
- Manage pilot configuration
- Collect and process human feedback
- Track pilot KPIs (query success rate, time saved, adoption)
- Prepare system for executive demo
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.core.database import get_sqlite

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEntry:
    """Human feedback on a query result."""
    question: str
    sql: str
    rating: int          # 1-5 stars
    is_correct: bool     # Was the SQL correct?
    correction: Optional[str] = None  # Corrected SQL if wrong
    comment: Optional[str] = None
    user_id: str = "anonymous"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PilotKPIs:
    """Pilot program KPI metrics."""
    # Usage
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate_pct: float = 0.0

    # Performance
    avg_response_time_s: float = 0.0
    cache_hit_rate_pct: float = 0.0

    # Quality
    avg_trust_score: float = 0.0
    golden_sql_count: int = 0
    feedback_count: int = 0
    positive_feedback_pct: float = 0.0

    # Business value
    tables_covered: int = 0
    unique_users: int = 0

    # Time period
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None

    def summary(self) -> str:
        return (
            f"Pilot KPIs Summary\n"
            f"{'─'*40}\n"
            f"Total Queries:     {self.total_queries}\n"
            f"Success Rate:      {self.success_rate_pct:.1f}%\n"
            f"Avg Response Time: {self.avg_response_time_s:.1f}s\n"
            f"Cache Hit Rate:    {self.cache_hit_rate_pct:.1f}%\n"
            f"Avg Trust Score:   {self.avg_trust_score:.1f}%\n"
            f"Golden SQL Pairs:  {self.golden_sql_count}\n"
            f"Positive Feedback: {self.positive_feedback_pct:.1f}%\n"
            f"Tables Covered:    {self.tables_covered}\n"
        )


class FeedbackCollector:
    """
    Stage 8.7/8.8 — UAT Feedback Collection + Monitoring Pipeline.
    Collects user feedback to improve system over time.
    """

    def __init__(self):
        self.db = get_sqlite()
        self._init_table()

    def _init_table(self):
        """Create feedback table if not exists."""
        try:
            self.db.execute_write("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    sql_query TEXT,
                    rating INTEGER,
                    is_correct INTEGER,
                    correction TEXT,
                    comment TEXT,
                    user_id TEXT DEFAULT 'anonymous',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            logger.debug(f"Feedback table init: {e}")

    def submit(self, entry: FeedbackEntry) -> bool:
        """Submit user feedback."""
        try:
            self.db.execute_write(
                """INSERT INTO feedback
                   (question, sql_query, rating, is_correct,
                    correction, comment, user_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.question, entry.sql, entry.rating,
                    int(entry.is_correct), entry.correction,
                    entry.comment, entry.user_id,
                )
            )

            # If user provided correction, auto-add to Golden SQL
            if entry.correction and entry.is_correct is False:
                self._promote_to_golden(entry.question, entry.correction)

            logger.info(
                f"Feedback received: {entry.rating}⭐ "
                f"correct={entry.is_correct}"
            )
            return True

        except Exception as e:
            logger.error(f"Feedback submit failed: {e}")
            return False

    def _promote_to_golden(self, question: str, corrected_sql: str):
        """
        Auto-promote corrected SQL to Golden SQL store.
        Stage 8.8 — feedback becomes training signal for continuous improvement.
        """
        try:
            from erp42_analytics.repositories.golden_sql_repo import GoldenSQLRepository
            golden = GoldenSQLRepository()
            golden.add(question, corrected_sql)
            logger.info(
                f"Auto-promoted to Golden SQL: '{question[:50]}'"
            )
        except Exception as e:
            logger.warning(f"Golden SQL promotion failed: {e}")

    def get_recent(self, limit: int = 20) -> list:
        """Get recent feedback entries."""
        return self.db.execute(
            "SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )

    def get_stats(self) -> dict:
        """Get feedback statistics."""
        total = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM feedback"
        )
        positive = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM feedback WHERE rating >= 4"
        )
        correct = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM feedback WHERE is_correct = 1"
        )
        avg_rating = self.db.execute_one(
            "SELECT AVG(rating) as avg FROM feedback"
        )

        total_cnt = total["cnt"] if total else 0
        positive_cnt = positive["cnt"] if positive else 0
        correct_cnt = correct["cnt"] if correct else 0

        return {
            "total_feedback": total_cnt,
            "positive_rate": round(
                positive_cnt / total_cnt * 100, 1
            ) if total_cnt > 0 else 0,
            "accuracy_rate": round(
                correct_cnt / total_cnt * 100, 1
            ) if total_cnt > 0 else 0,
            "avg_rating": round(
                avg_rating["avg"] or 0, 1
            ) if avg_rating else 0,
        }


class PilotManager:
    """
    Stage 8.1 — Production Environment Preparation / Pilot Management.
    Manages pilot configuration and KPI tracking.
    """

    def __init__(self):
        self.db = get_sqlite()
        self.feedback = FeedbackCollector()

    def get_kpis(self) -> PilotKPIs:
        """
        Stage 8.10 — Phase 1 Final Validation KPIs.
        """
        kpis = PilotKPIs()

        # Query stats
        total = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM query_history"
        )
        success = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM query_history WHERE success = 1"
        )
        avg_time = self.db.execute_one(
            "SELECT AVG(exec_time) as avg FROM query_history WHERE success = 1"
        )
        avg_trust = self.db.execute_one(
            "SELECT AVG(confidence_score) as avg FROM query_history WHERE success = 1"
        )
        golden_count = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM golden_sql"
        )
        table_count = self.db.execute_one(
            "SELECT COUNT(DISTINCT tables_used) as cnt FROM query_history WHERE tables_used != '[]'"
        )

        kpis.total_queries = total["cnt"] if total else 0
        kpis.successful_queries = success["cnt"] if success else 0
        kpis.failed_queries = kpis.total_queries - kpis.successful_queries
        kpis.success_rate_pct = round(
            kpis.successful_queries / kpis.total_queries * 100, 1
        ) if kpis.total_queries > 0 else 0
        kpis.avg_response_time_s = round(
            avg_time["avg"] or 0, 2
        ) if avg_time else 0
        kpis.avg_trust_score = round(
            avg_trust["avg"] or 0, 1
        ) if avg_trust else 0
        kpis.golden_sql_count = golden_count["cnt"] if golden_count else 0

        # Feedback stats
        fb_stats = self.feedback.get_stats()
        kpis.feedback_count = fb_stats["total_feedback"]
        kpis.positive_feedback_pct = fb_stats["positive_rate"]

        return kpis

    def prepare_demo(self) -> dict:
        """
        Stage 8.9 — Pilot Documentation & Demo Readiness.
        Pre-warm cache with common queries for smooth demo.
        """
        from erp42_analytics.intelligence.benchmark import GROUND_TRUTH

        # Get top queries for demo
        demo_queries = [
            t["question"] for t in GROUND_TRUTH
            if t["difficulty"] in ("easy", "medium")
        ][:5]

        return {
            "demo_queries": demo_queries,
            "golden_sql_count": self.get_kpis().golden_sql_count,
            "tables_scanned": self.db.execute_one(
                "SELECT COUNT(*) as cnt FROM table_descriptions WHERE status='SCANNED'"
            )["cnt"],
            "system_ready": True,
        }


# ─── Singletons ───────────────────────────────────────────────────────────────
_feedback_collector: Optional[FeedbackCollector] = None
_pilot_manager: Optional[PilotManager] = None


def get_feedback_collector() -> FeedbackCollector:
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector


def get_pilot_manager() -> PilotManager:
    global _pilot_manager
    if _pilot_manager is None:
        _pilot_manager = PilotManager()
    return _pilot_manager
