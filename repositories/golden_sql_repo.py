"""
repositories/golden_sql_repo.py — Golden SQL Repository
Replaces: golden_sql_store.py
Manages verified Q&A pairs for few-shot learning.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional

import chromadb

from erp42_analytics.core.config import config
from erp42_analytics.core.database import get_sqlite
from erp42_analytics.core.models import GoldenSQL

logger = logging.getLogger(__name__)

COLLECTION_NAME = "golden_sql"


class GoldenSQLRepository:
    """
    Stores and retrieves verified SQL examples.
    Uses ChromaDB for semantic similarity search.
    SQLite for persistence and metadata.
    """

    def __init__(self):
        self.db = get_sqlite()
        self.chroma_client = chromadb.PersistentClient(
            path=config.storage.chroma_path
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        self._sync_from_sqlite()

    # ─── Write Operations ─────────────────────────────────────────────────────

    def add(self, question: str, sql: str,
            tables_used: list = None) -> bool:
        """Add verified Q&A pair to both SQLite and ChromaDB."""
        try:
            entry_id = str(uuid.uuid4())[:8]
            tables_json = json.dumps(tables_used or [])

            # Save to SQLite
            self.db.execute_write(
                """INSERT OR REPLACE INTO golden_sql
                   (id, question, sql_query, tables_used, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (entry_id, question.strip(), sql.strip(),
                 tables_json, datetime.now().isoformat())
            )

            # Save to ChromaDB for semantic search
            existing = self.collection.get(ids=[entry_id])
            if existing["ids"]:
                self.collection.update(
                    ids=[entry_id],
                    documents=[question],
                    metadatas=[{"sql": sql, "tables": tables_json}]
                )
            else:
                self.collection.add(
                    ids=[entry_id],
                    documents=[question],
                    metadatas=[{"sql": sql, "tables": tables_json}]
                )

            logger.info(f"Golden SQL added: '{question[:50]}'")
            return True

        except Exception as e:
            logger.error(f"Failed to add Golden SQL: {e}")
            return False

    # ─── Read Operations ──────────────────────────────────────────────────────

    def find_similar(self, question: str, n_results: int = 3,
                     min_score: float = 0.3) -> list:
        """Find similar Golden SQL examples using semantic search."""
        if self.collection.count() == 0:
            return []

        try:
            n = min(n_results, self.collection.count())
            results = self.collection.query(
                query_texts=[question],
                n_results=n,
            )

            examples = []
            if results and results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    score = 1 - distance

                    if score < min_score:
                        continue

                    metadata = results["metadatas"][0][i]
                    examples.append(GoldenSQL(
                        id=doc_id,
                        question=results["documents"][0][i],
                        sql=metadata.get("sql", ""),
                        tables_used=json.loads(metadata.get("tables", "[]")),
                        score=round(score, 3),
                    ))

            logger.info(
                f"Found {len(examples)} Golden SQL examples "
                f"for: '{question[:40]}'"
            )
            return examples

        except Exception as e:
            logger.error(f"Golden SQL search failed: {e}")
            return []

    def get_all(self) -> list:
        """Get all Golden SQL entries."""
        rows = self.db.execute(
            "SELECT * FROM golden_sql ORDER BY created_at DESC"
        )
        return [GoldenSQL(
            id=r["id"],
            question=r["question"],
            sql=r["sql_query"],
            tables_used=json.loads(r.get("tables_used") or "[]"),
            created_at=r.get("created_at"),
        ) for r in rows]

    def count(self) -> int:
        """Get total number of Golden SQL entries."""
        result = self.db.execute_one(
            "SELECT COUNT(*) as cnt FROM golden_sql"
        )
        return result["cnt"] if result else 0

    # ─── Sync ─────────────────────────────────────────────────────────────────

    def _sync_from_sqlite(self):
        """Sync SQLite → ChromaDB on startup."""
        try:
            sqlite_count = self.count()
            chroma_count = self.collection.count()

            if sqlite_count > chroma_count:
                rows = self.db.execute(
                    "SELECT * FROM golden_sql ORDER BY created_at"
                )
                for row in rows:
                    existing = self.collection.get(ids=[row["id"]])
                    if not existing["ids"]:
                        self.collection.add(
                            ids=[row["id"]],
                            documents=[row["question"]],
                            metadatas=[{
                                "sql": row["sql_query"],
                                "tables": row.get("tables_used", "[]")
                            }]
                        )
                logger.info(
                    f"Synced {sqlite_count - chroma_count} "
                    f"Golden SQL entries to ChromaDB"
                )
        except Exception as e:
            logger.warning(f"Golden SQL sync failed: {e}")
