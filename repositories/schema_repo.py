"""
repositories/schema_repo.py — Schema Metadata Repository
Replaces: storage.py (schema-related functions)
Abstracts all schema metadata persistence.
"""

import json
import logging
from typing import Optional
from datetime import datetime

from erp42_analytics.core.database import get_sqlite
from erp42_analytics.core.models import TableInfo, ColumnInfo, JoinRelationship

logger = logging.getLogger(__name__)


class SchemaRepository:
    """
    Handles all schema metadata storage and retrieval.
    Single source of truth for table/column/relationship metadata.
    """

    def __init__(self):
        self.db = get_sqlite()

    # ─── Table Operations ─────────────────────────────────────────────────────

    def save_table(self, table_name: str, columns: list,
                   foreign_keys: list = None, row_count: int = 0,
                   examples: list = None):
        """Save or update table metadata including cardinality and sample rows."""

        self.db.execute_write(
            """INSERT OR REPLACE INTO table_descriptions
               (table_name, columns_json, foreign_keys_json, row_count,
                examples_json, scanned_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                table_name.lower(),
                json.dumps(columns),
                json.dumps(foreign_keys or []),
                row_count,
                json.dumps(examples or []),
                datetime.now().isoformat(),
            )
        )

    def get_table(self, table_name: str) -> Optional[dict]:
        """Get single table metadata."""
        row = self.db.execute_one(
            "SELECT * FROM table_descriptions WHERE table_name = ?",
            (table_name.lower(),)
        )
        if not row:
            return None
        return self._deserialize_table(row)

    def get_all_tables(self) -> dict:
        """
        Get all scanned tables.
        Returns dict: {table_name: {columns, foreign_keys, ...}}
        Compatible with existing code.
        """
        rows = self.db.execute(
            "SELECT * FROM table_descriptions WHERE status = 'SCANNED'"
        )
        return {row["table_name"]: self._deserialize_table(row) for row in rows}

    def get_table_count(self) -> int:
        """Get total number of scanned tables."""
        result = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM table_descriptions WHERE status = 'SCANNED'"
        )
        return result["cnt"] if result else 0

    def save_business_description(self, table_name: str, description: str):
        """Save AI-generated business description."""
        self.db.execute_write(
            """UPDATE table_descriptions 
               SET business_description = ? WHERE table_name = ?""",
            (description, table_name.lower())
        )

    def get_tables_without_description(self) -> list:
        """Get tables that haven't been enriched yet."""
        rows = self.db.execute(
            """SELECT table_name FROM table_descriptions
               WHERE business_description IS NULL AND status = 'SCANNED'
               ORDER BY table_name"""
        )
        return [row["table_name"] for row in rows]

    def is_schema_stale(self, max_age_hours: int = 24) -> bool:
        """Check if schema needs rescanning."""
        result = self.db.execute_one(
            """SELECT MIN(scanned_at) as oldest 
               FROM table_descriptions WHERE status = 'SCANNED'"""
        )
        if not result or not result["oldest"]:
            return True
        try:
            oldest = datetime.fromisoformat(result["oldest"])
            age_hours = (datetime.now() - oldest).total_seconds() / 3600
            return age_hours > max_age_hours
        except Exception:
            return True

    # ─── Join Relationships ───────────────────────────────────────────────────

    def save_join(self, from_table: str, from_col: str,
                  to_table: str, to_col: str,
                  confidence: float = 1.0, is_verified: bool = False,
                  rel_type: str = "many_to_one"):
        """Save detected join relationship."""
        self.db.execute_write(
            """INSERT OR REPLACE INTO join_relationships
               (from_table, from_column, to_table, to_column,
                confidence, is_verified, relationship_type)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (from_table.lower(), from_col, to_table.lower(), to_col,
             confidence, int(is_verified), rel_type)
        )

    def get_joins_for_table(self, table_name: str) -> list:
        """Get all join relationships for a table."""
        rows = self.db.execute(
            """SELECT * FROM join_relationships
               WHERE from_table = ? OR to_table = ?
               ORDER BY confidence DESC""",
            (table_name.lower(), table_name.lower())
        )
        return [JoinRelationship(
            from_table=r["from_table"],
            from_column=r["from_column"],
            to_table=r["to_table"],
            to_column=r["to_column"],
            confidence=r["confidence"],
            is_verified=bool(r["is_verified"]),
            relationship_type=r["relationship_type"]
        ) for r in rows]

    def get_all_joins(self) -> list:
        """Get all join relationships."""
        rows = self.db.execute(
            "SELECT * FROM join_relationships ORDER BY confidence DESC"
        )
        return rows

    def verify_join(self, from_table: str, from_col: str,
                    to_table: str, to_col: str):
        """Mark a join as human-verified."""
        self.db.execute_write(
            """UPDATE join_relationships SET is_verified = 1
               WHERE from_table = ? AND from_column = ?
               AND to_table = ? AND to_column = ?""",
            (from_table.lower(), from_col, to_table.lower(), to_col)
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _deserialize_table(self, row: dict) -> dict:
        """Convert DB row to dict with parsed JSON."""
        try:
            columns = json.loads(row.get("columns_json") or "[]")
        except Exception:
            columns = []
        try:
            foreign_keys = json.loads(row.get("foreign_keys_json") or "[]")
        except Exception:
            foreign_keys = []
        try:
            examples = json.loads(row.get("examples_json") or "[]")
        except Exception:
            examples = []

        return {
            "columns": columns,
            "foreign_keys": foreign_keys,
            "row_count": row.get("row_count", 0),
            "business_description": row.get("business_description"),
            "examples": examples,
            "scanned_at": row.get("scanned_at"),
        }