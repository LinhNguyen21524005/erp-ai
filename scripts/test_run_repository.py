"""
scripts/test_run_repository.py — Test Run Repository (Step 1.11)

Persists every batch-test run into SQLite so runs are versioned and
comparable week-over-week:

    test_runs     — one row per run (run_id, timestamp, label, env info)
    test_results  — one row per question result, linked by run_id

This is what powers the Baseline → Fix → Re-test loop:
    Week 1 on-site: run full UAT set  → run_id=baseline_w1
    Week 2:         fix, re-run        → run_id=after_fixes_w2
    Compare the two → fixed / regressed / unchanged (run_comparison.py)
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Store next to the existing metadata DB
_DEFAULT_DB = (
    Path(__file__).resolve().parents[1] / "data" / "test_runs.db"
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS test_runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    label         TEXT DEFAULT '',
    runtime_mode  TEXT DEFAULT '',
    llm_model     TEXT DEFAULT '',
    database_name TEXT DEFAULT '',
    n_questions   INTEGER DEFAULT 0,
    n_success     INTEGER DEFAULT 0,
    n_failed      INTEGER DEFAULT 0,
    success_rate  REAL DEFAULT 0.0,
    avg_time_s    REAL DEFAULT 0.0,
    notes         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS test_results (
    run_id            TEXT NOT NULL,
    question_id       TEXT NOT NULL,
    question          TEXT DEFAULT '',
    language          TEXT DEFAULT '',
    category          TEXT DEFAULT '',
    success           INTEGER DEFAULT 0,
    error             TEXT DEFAULT '',
    failure_stage     TEXT DEFAULT '',
    error_category    TEXT DEFAULT '',
    sql               TEXT DEFAULT '',
    row_count         INTEGER DEFAULT 0,
    trust_score       REAL DEFAULT 0.0,
    num_attempts      INTEGER DEFAULT 1,
    total_time_s      REAL DEFAULT 0.0,
    raw_json          TEXT DEFAULT '',
    PRIMARY KEY (run_id, question_id),
    FOREIGN KEY (run_id) REFERENCES test_runs(run_id)
);

CREATE INDEX IF NOT EXISTS ix_results_run
    ON test_results (run_id);
"""


class TestRunRepository:
    def __init__(self, db_path: Optional[str] = None):
        self._path = str(db_path or _DEFAULT_DB)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        con = sqlite3.connect(self._path)
        con.row_factory = sqlite3.Row
        return con

    # ── write ─────────────────────────────────────────────────────────
    def save_run(
        self,
        run_id: str,
        results: list,          # list[BatchResult]
        label: str = "",
        runtime_mode: str = "",
        llm_model: str = "",
        database_name: str = "",
        notes: str = "",
    ) -> str:
        """
        Persist one completed batch run. `results` are BatchResult
        objects from batch_test_runner (uses duck-typing — any object
        with the same attribute names works).
        """
        n = len(results)
        n_ok = sum(1 for r in results if r.success)
        times = [r.total_time_s for r in results if r.success]
        avg_t = round(sum(times) / len(times), 1) if times else 0.0

        with self._conn() as con:
            con.execute(
                """INSERT OR REPLACE INTO test_runs
                   (run_id, created_at, label, runtime_mode, llm_model,
                    database_name, n_questions, n_success, n_failed,
                    success_rate, avg_time_s, notes)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id, datetime.now().isoformat(), label,
                    runtime_mode, llm_model, database_name,
                    n, n_ok, n - n_ok,
                    round(100.0 * n_ok / n, 1) if n else 0.0,
                    avg_t, notes,
                ),
            )
            for r in results:
                con.execute(
                    """INSERT OR REPLACE INTO test_results
                       (run_id, question_id, question, language, category,
                        success, error, failure_stage, error_category,
                        sql, row_count, trust_score, num_attempts,
                        total_time_s, raw_json)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        run_id, r.row_id, r.question, r.language,
                        r.category, int(r.success), r.error,
                        getattr(r, "failure_stage", ""),
                        getattr(r, "error_category", ""),
                        r.sql, r.row_count, r.trust_score,
                        getattr(r, "num_attempts", 1),
                        r.total_time_s,
                        json.dumps(
                            r.to_csv_row(), ensure_ascii=False,
                            default=str,
                        ),
                    ),
                )
        logger.info(
            "TestRunRepository: saved run '%s' (%d results, %.1f%% ok)",
            run_id, n, 100.0 * n_ok / n if n else 0.0,
        )
        return run_id

    # ── read ──────────────────────────────────────────────────────────
    def list_runs(self, limit: int = 20) -> list:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM test_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as con:
            row = con.execute(
                "SELECT * FROM test_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_results(self, run_id: str) -> list:
        with self._conn() as con:
            rows = con.execute(
                "SELECT * FROM test_results WHERE run_id = ? "
                "ORDER BY question_id",
                (run_id,),
            ).fetchall()
        return [dict(r) for r in rows]


def make_run_id(prefix: str = "run") -> str:
    """e.g. run_20260712_1430"""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"
