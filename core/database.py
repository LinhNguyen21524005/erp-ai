"""
core/database.py — Database Connection Manager
Manages SQLite metadata DB and MSSQL ERP connections.
Single connection point for entire system.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── SQLite Manager ───────────────────────────────────────────────────────────

class SQLiteManager:
    """
    Manages SQLite connection for metadata storage.
    Replaces direct sqlite3 calls scattered across old storage.py.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        """Initialize all tables on first run."""
        with self.connect() as conn:
            conn.executescript("""
                -- Table metadata
                CREATE TABLE IF NOT EXISTS table_descriptions (
                    table_name TEXT PRIMARY KEY,
                    columns_json TEXT,
                    foreign_keys_json TEXT,
                    row_count INTEGER DEFAULT 0,
                    business_description TEXT,
                    examples_json TEXT,
                    status TEXT DEFAULT 'SCANNED',
                    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Golden SQL store
                CREATE TABLE IF NOT EXISTS golden_sql (
                    id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    sql_query TEXT NOT NULL,
                    tables_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verified INTEGER DEFAULT 1
                );

                -- Query history
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    sql_query TEXT,
                    success INTEGER,
                    exec_time REAL,
                    row_count INTEGER,
                    confidence_score REAL,
                    tables_used TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Business instructions
                CREATE TABLE IF NOT EXISTS instructions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instruction TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                -- Synonym knowledge base (NEW - Stage 3.4)
                CREATE TABLE IF NOT EXISTS synonyms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    term TEXT NOT NULL,
                    synonym TEXT NOT NULL,
                    domain TEXT DEFAULT 'general',
                    confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(term, synonym)
                );

                -- Join relationships (NEW - Stage 2.6)
                CREATE TABLE IF NOT EXISTS join_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_table TEXT NOT NULL,
                    from_column TEXT NOT NULL,
                    to_table TEXT NOT NULL,
                    to_column TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    is_verified INTEGER DEFAULT 0,
                    relationship_type TEXT DEFAULT 'many_to_one',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(from_table, from_column, to_table, to_column)
                );

                -- Audit log (NEW - Stage 7.4)
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id TEXT DEFAULT 'anonymous',
                    session_id TEXT,
                    question TEXT,
                    sql_generated TEXT,
                    tables_accessed TEXT,
                    status TEXT,
                    trust_score REAL,
                    exec_time REAL,
                    row_count INTEGER,
                    error TEXT
                );

                -- Query cache (NEW - Stage 4.7)
                CREATE TABLE IF NOT EXISTS query_cache (
                    cache_key TEXT PRIMARY KEY,
                    question TEXT,
                    sql_query TEXT,
                    result_json TEXT,
                    trust_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    hit_count INTEGER DEFAULT 0
                );
            """)
        logger.info(f"SQLite initialized: {self.db_path}")

    @contextmanager
    def connect(self):
        """Context manager for SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"SQLite error: {e}")
            raise
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> list:
        """Execute query and return results."""
        with self.connect() as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def execute_one(self, query: str, params: tuple = ()) -> Optional[dict]:
        """Execute query and return first result."""
        results = self.execute(query, params)
        return results[0] if results else None

    def execute_write(self, query: str, params: tuple = ()) -> int:
        """Execute write query and return affected rows."""
        with self.connect() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount


# ─── MSSQL Manager ────────────────────────────────────────────────────────────

class MSSQLManager:
    """
    Manages MSSQL ERP database connections.
    Uses SQLAlchemy for connection pooling.
    """

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self._engine = None

    @property
    def engine(self):
        """Lazy-load SQLAlchemy engine."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                self._engine = create_engine(
                    self.connection_string,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                )
                logger.info("MSSQL engine created")
            except Exception as e:
                logger.error(f"MSSQL connection failed: {e}")
                raise
        return self._engine

    def test_connection(self) -> bool:
        """Test if ERP database is reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"MSSQL connection test failed: {e}")
            return False

    def execute(self, sql: str, timeout: int = 30) -> tuple:
        """
        Execute SQL and return (columns, rows).
        Read-only — no writes allowed.
        """
        from sqlalchemy import text
        with self.engine.connect() as conn:
            result = conn.execute(
                text(sql).execution_options(timeout=timeout)
            )
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
            return columns, rows


# ─── Database Registry ────────────────────────────────────────────────────────

_sqlite: Optional[SQLiteManager] = None
_mssql: Optional[MSSQLManager] = None


def get_sqlite(db_path: str = None) -> SQLiteManager:
    """Get SQLite singleton."""
    global _sqlite
    if _sqlite is None:
        from erp42_analytics.core.config import config
        path = db_path or config.storage.sqlite_path
        _sqlite = SQLiteManager(path)
    return _sqlite


def get_mssql(connection_string: str = None) -> MSSQLManager:
    """Get MSSQL singleton."""
    global _mssql
    if _mssql is None:
        from erp42_analytics.core.config import config
        conn_str = connection_string or config.db.connection_string
        _mssql = MSSQLManager(conn_str)
    return _mssql
