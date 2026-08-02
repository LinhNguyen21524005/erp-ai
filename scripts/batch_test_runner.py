"""
scripts/batch_test_runner.py — Automated Batch Question Tester

Runs a large list of natural language questions against the query
pipeline (without going through Gradio/HTTP) and logs detailed
results to CSV + JSON for offline analysis.

Why this exists:
- Manually copy-pasting 100 questions into Gradio is impractical
  (each query takes 200-700s currently).
- This script lets you queue up a question list, walk away, and
  come back to a structured log of what passed/failed and why.

Usage (from D:/SEPT/AskEase, with the venv/conda env active):

    python -m erp42_analytics.scripts.batch_test_runner --input questions.csv

    # Or run the built-in GROUND_TRUTH set:
    python -m erp42_analytics.scripts.batch_test_runner --use-ground-truth

    # Repeat each question N times to measure stability:
    python -m erp42_analytics.scripts.batch_test_runner --input questions.csv --repeat 3

Input CSV format (UTF-8, header required):
    id,question,language,category
    Q001,"Wie viele Aufträge haben wir?",de,simple_count
    Q002,"How many customers do we have?",en,simple_count

Only the 'question' column is required; the others are optional
metadata carried through into the results log.

Output:
    logs/batch_results_<timestamp>.csv   — one row per attempt
    logs/batch_results_<timestamp>.json  — full detail incl. SQL + rows
    logs/batch_summary_<timestamp>.txt   — human-readable summary
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# Allow running as `python -m erp42_analytics.scripts.batch_test_runner`
# from the parent folder (D:/SEPT/AskEase)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """One row in the batch test log — one question attempt."""
    row_id: str
    question: str
    language: str = ""
    category: str = ""
    attempt_number: int = 1

    success: bool = False
    error: str = ""
    sql: str = ""
    row_count: int = 0
    tables_used: list = field(default_factory=list)
    trust_score: float = 0.0
    confidence_label: str = ""
    warnings: list = field(default_factory=list)
    retried: bool = False
    explanation: str = ""

    total_time_s: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    # Flags for quick filtering in the CSV afterwards
    flag_empty_result: bool = False
    flag_explanation_table_mismatch: bool = False
    flag_slow: bool = False  # > 60s
    flag_suspicious_comment: bool = False
    flag_unexpected_table: str = ""  # non-empty = list of unknown tables used
    flag_self_join: bool = False
    flag_unjoined_filter_table: str = ""  # non-empty = dangling qualifier/parse issue

    # Attempt history (Nhóm 1+2) — full retry-loop trace
    num_attempts: int = 1
    first_attempt_sql: str = ""
    first_attempt_stage_failed_at: str = ""
    first_attempt_error: str = ""
    first_attempt_error_category: str = ""
    final_attempt_sql: str = ""
    attempts_summary: str = ""  # compact "attempt 1: stage/category | attempt 2: ..."

    # Record-level failure classification (Step 5.6 Failure Taxonomy).
    # For a FAILED question: which pipeline stage it ultimately failed at
    # and the standardized error category. Empty when success=True.
    # These two feed the TestRunRepository → Run Comparison Engine.
    failure_stage: str = ""      # retrieval|sql_generation|validation|execution|response
    error_category: str = ""     # hallucinated_table|syntax_error|empty_result|timeout|...

    # Hallucination severity of the FINAL accepted attempt (Nhóm "còn sót")
    hallucination_severity: str = ""
    max_hallucination_severity: str = ""  # worst across ALL attempts

    # Trust score breakdown (Nhóm 6) — final attempt's component scores
    trust_retrieval_confidence: float = 0.0
    trust_sql_reliability: float = 0.0
    trust_validation_score: float = 0.0
    trust_data_freshness: float = 0.0
    trust_join_confidence: float = 0.0

    # Latency breakdown of the FINAL (successful) attempt (Nhóm 5)
    final_attempt_sql_generation_ms: float = 0.0
    final_attempt_db_execution_ms: float = 0.0
    # Sum across ALL attempts — shows total LLM time burned by retries
    total_sql_generation_ms: float = 0.0
    total_db_execution_ms: float = 0.0

    def to_csv_row(self) -> dict:
        """Flat dict for CSV writing (no nested lists)."""
        d = asdict(self)
        d["tables_used"] = "|".join(self.tables_used)
        d["warnings"] = " || ".join(self.warnings)
        d["sql"] = self.sql.replace("\n", " ").strip()
        d["first_attempt_sql"] = self.first_attempt_sql.replace(
            "\n", " "
        ).strip()
        d["final_attempt_sql"] = self.final_attempt_sql.replace(
            "\n", " "
        ).strip()
        return d


CSV_FIELDS = [
    "row_id", "question", "language", "category", "attempt_number",
    "success", "error", "sql", "row_count", "tables_used",
    "trust_score", "confidence_label", "warnings", "retried",
    "explanation", "total_time_s", "timestamp",
    "flag_empty_result", "flag_explanation_table_mismatch", "flag_slow",
    "flag_suspicious_comment", "flag_unexpected_table",
    "flag_self_join", "flag_unjoined_filter_table",
    "num_attempts", "first_attempt_sql", "first_attempt_stage_failed_at",
    "first_attempt_error", "first_attempt_error_category",
    "final_attempt_sql", "attempts_summary",
    "failure_stage", "error_category",
    "hallucination_severity",
    "max_hallucination_severity",
    "trust_retrieval_confidence", "trust_sql_reliability",
    "trust_validation_score", "trust_data_freshness", "trust_join_confidence",
    "final_attempt_sql_generation_ms", "final_attempt_db_execution_ms",
    "total_sql_generation_ms", "total_db_execution_ms",
]


def load_questions_from_csv(path: str) -> list:
    """Load question rows from a CSV file."""
    questions = []
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            questions.append({
                "id": row.get("id") or f"Q{i:03d}",
                "question": row["question"].strip(),
                "language": row.get("language", "").strip(),
                "category": row.get("category", "").strip(),
            })
    return questions


def load_ground_truth_questions() -> list:
    """Load the built-in GROUND_TRUTH set from benchmark.py."""
    from erp42_analytics.intelligence.benchmark import GROUND_TRUTH
    return [
        {
            "id": t["id"],
            "question": t["question"],
            "language": "de" if any(
                c in t["question"] for c in "äöüÄÖÜß"
            ) else "en",
            "category": t.get("intent_type", ""),
        }
        for t in GROUND_TRUTH
    ]


def check_explanation_table_mismatch(
    sql: str, explanation: str
) -> bool:
    """
    Quick sanity check: does the explanation mention a table name
    that doesn't actually appear in the SQL? Helps catch the
    'explanation lists tables not in the SQL' bug pattern.
    """
    if not sql or not explanation:
        return False
    try:
        from sql_metadata import Parser
        sql_tables = {
            t.split(".")[-1].lower().strip("[]")
            for t in Parser(sql).tables
        }
    except Exception:
        return False

    # crude extraction: words in explanation matching "dbo.xxx" pattern
    import re
    mentioned = set(
        m.lower() for m in re.findall(r"dbo\.(\w+)", explanation)
    )
    extra = mentioned - sql_tables
    return len(extra) > 0


SUSPICIOUS_COMMENT_PATTERN = (
    r"(?i)--\s*(assuming|note:|i\s+assum|as\s+per\s+the\s+rules|"
    r"we\s+want\s+to\s+include|this\s+assumes)"
)


def check_suspicious_comment(sql: str) -> bool:
    """
    Detect SQL containing an inline comment where the model appears
    to be 'justifying' a filter/assumption decision — often a sign
    of a questionable or made-up business rule (e.g. the Q005 case:
    "-- Assuming we want to include all records with non-null sales
    values..."). Worth a human glance even if the SQL ran fine.
    """
    import re
    if not sql:
        return False
    return bool(re.search(SUSPICIOUS_COMMENT_PATTERN, sql))


def check_unexpected_table(sql: str, known_tables: set) -> str:
    """
    Returns a comma-separated list of tables used in the SQL that
    are NOT in the known core/scanned table set — empty string if
    all tables are recognized. Catches hallucinated table names
    like 'sh_order' that happen to exist in the DB but aren't part
    of the documented schema the model was given.
    """
    if not sql or not known_tables:
        return ""
    try:
        from sql_metadata import Parser
        sql_tables = {
            t.split(".")[-1].lower().strip("[]")
            for t in Parser(sql).tables
        }
    except Exception:
        return ""
    unexpected = sql_tables - known_tables
    return "|".join(sorted(unexpected))


def check_self_join(sql: str) -> bool:
    """
    Detect a redundant self-join: the same underlying table joined
    to itself under two different aliases, joined on a column that
    is identical in both (e.g. the same primary key). This doesn't
    break the query's correctness for 1:1 joins, but it's wasted
    work and usually a sign the model got confused about which
    alias to use (e.g. Q002's 'auftrag JOIN auftrag').

    Heuristic: sql_metadata's tables_aliases maps alias -> full
    table name. If 2+ aliases point to the same table, flag it.
    """
    if not sql:
        return False
    try:
        from sql_metadata import Parser
        aliases = Parser(sql).tables_aliases
    except Exception:
        return False

    seen_tables = {}
    for alias, table in aliases.items():
        normalized = table.split(".")[-1].lower().strip("[]")
        seen_tables.setdefault(normalized, []).append(alias)

    return any(len(aliases_list) > 1 for aliases_list in seen_tables.values())


def check_unjoined_filter_table(sql: str) -> str:
    """
    Detect a WHERE/ON clause referencing a table.column combination
    where that table alias/name doesn't appear in the FROM/JOIN
    clauses at all — a sign of a dangling reference that may or may
    not error out depending on SQL Server's resolution rules (e.g.
    Q008's case referencing a phantom 'ERP42test.dbo01_dbODBCDataBase'
    prefix that was never joined).

    Returns a short description of the issue found, or "" if none.
    Two detection paths:
    1. sql_metadata itself fails to parse the column list — this
       reliably happens for malformed double-dot qualifiers like
       "db.dbo..table.column", which is itself a strong signal of
       a hallucinated/garbled table reference.
    2. A column's table qualifier doesn't match any FROM/JOIN table
       or alias in the query.
    """
    if not sql:
        return ""
    try:
        from sql_metadata import Parser
        p = Parser(sql)
        from_join_tables = {
            t.split(".")[-1].lower().strip("[]") for t in p.tables
        }
        aliases = {a.lower() for a in p.tables_aliases.keys()}
        valid_qualifiers = from_join_tables | aliases

        dangling = set()
        for col in p.columns:
            if "." in col:
                qualifier = col.rsplit(".", 1)[0].lower()
                qualifier_table = qualifier.split(".")[-1].strip("[]")
                if (
                    qualifier_table
                    and qualifier_table not in valid_qualifiers
                    and qualifier_table not in ("dbo",)  # schema prefix, ignore
                ):
                    dangling.add(qualifier)
        return "|".join(sorted(dangling))
    except Exception as e:
        # sql_metadata choking on the column list is itself a strong
        # signal of a malformed/hallucinated table.column reference
        # (e.g. double-dot qualifiers, phantom database prefixes).
        return f"PARSE_FAILURE: {str(e)[:100]}"


def _infer_failure_stage(error: str) -> str:
    """
    Map a raw error string to a pipeline stage when no traced stage
    is available. Conservative — defaults to 'sql_generation' since
    that is where most uncaught failures originate.
    """
    e = (error or "").lower()
    if not e:
        return ""
    if any(k in e for k in ("no tables", "retrieval", "no relevant", "schema context")):
        return "retrieval"
    if any(k in e for k in ("timeout", "connection", "cannot execute", "database error", "execution")):
        return "execution"
    if any(k in e for k in ("invalid column", "invalid object", "syntax", "validation", "unknown table")):
        return "validation"
    if any(k in e for k in ("no sql", "could not generate", "empty response")):
        return "sql_generation"
    return "sql_generation"


def _infer_error_category(result) -> str:
    """
    Map the richest available failure signal to a standard error
    category. Order matters: most-specific flags first.
    """
    e = (result.error or "").lower()
    # Flag-derived (most reliable)
    if result.flag_unexpected_table:
        return "hallucinated_table"
    if result.flag_self_join:
        return "self_join"
    if result.flag_suspicious_comment:
        return "suspicious_comment"
    # Error-text derived
    if "timeout" in e:
        return "timeout"
    if any(k in e for k in ("permission", "not authorized", "access denied", "rbac")):
        return "permission_denied"
    if any(k in e for k in ("invalid column", "unknown column")):
        return "hallucinated_column"
    if any(k in e for k in ("invalid object", "unknown table")):
        return "hallucinated_table"
    if any(k in e for k in ("syntax", "incorrect syntax")):
        return "syntax_error"
    if any(k in e for k in ("no tables", "no relevant", "ambiguous")):
        return "ambiguous_query"
    if "exception" in e:
        return "runtime_exception"
    return "other"


def run_batch(
    questions: list,
    repeat: int = 1,
    role: str = "analyst",
    use_cache: bool = False,
    delay_between_s: float = 1.0,
) -> list:
    """
    Run all questions (optionally repeated) through the query pipeline.
    Returns a list of BatchResult.
    """
    from erp42_analytics.services.schema_service import SchemaService
    from erp42_analytics.services.semantic_service import (
        SemanticService, IMPORTANT_COLUMNS
    )

    known_tables = set(IMPORTANT_COLUMNS.keys())
    from erp42_analytics.services.query_service import QueryService

    logger.info("Initializing pipeline (schema + embeddings)...")
    schema_svc = SchemaService()
    all_tables = schema_svc.get_all_tables()
    if not all_tables:
        logger.warning(
            "No scanned tables found — run schema_svc.scan_all_tables() "
            "first if this is unexpected."
        )

    sem_svc = SemanticService()
    if not sem_svc.is_indexed():
        logger.info("Indexing schema embeddings (one-time)...")
        sem_svc.index_tables(all_tables)

    svc = QueryService(all_tables=all_tables)
    logger.info(f"Pipeline ready: {len(all_tables)} tables loaded.")

    total_runs = len(questions) * repeat
    results = []
    run_num = 0

    for q in questions:
        for attempt in range(1, repeat + 1):
            run_num += 1
            logger.info(
                f"[{run_num}/{total_runs}] {q['id']} "
                f"(attempt {attempt}/{repeat}): "
                f"'{q['question'][:60]}'"
            )

            result = BatchResult(
                row_id=q["id"],
                question=q["question"],
                language=q.get("language", ""),
                category=q.get("category", ""),
                attempt_number=attempt,
            )

            start = time.time()
            try:
                response = svc.ask(
                    question=q["question"],
                    role=role,
                    use_cache=use_cache,
                )
                result.total_time_s = round(time.time() - start, 2)
                result.success = (
                    response.result.status.value == "success"
                )
                result.error = response.result.error or ""
                result.sql = response.sql
                result.row_count = response.result.row_count
                result.tables_used = response.tables_used or []
                result.trust_score = response.trust_score.overall
                result.confidence_label = response.trust_score.label
                result.trust_retrieval_confidence = (
                    response.trust_score.retrieval_confidence
                )
                result.trust_sql_reliability = (
                    response.trust_score.sql_reliability
                )
                result.trust_validation_score = (
                    response.trust_score.validation_score
                )
                result.trust_data_freshness = (
                    response.trust_score.data_freshness
                )
                result.trust_join_confidence = (
                    response.trust_score.join_confidence
                )
                result.warnings = response.warnings or []
                result.retried = any(
                    "retry attempt" in w for w in result.warnings
                )
                result.explanation = response.explanation
                result.hallucination_severity = (
                    response.hallucination_severity or ""
                )
                result.max_hallucination_severity = (
                    getattr(response, "max_hallucination_severity", "") or ""
                )

                # ── Attempt history (Nhóm 1+2) ──────────────────────────────
                attempts = response.attempts or []
                result.num_attempts = len(attempts)
                if attempts:
                    first = attempts[0]
                    last = attempts[-1]
                    result.first_attempt_sql = first.sql
                    result.first_attempt_stage_failed_at = (
                        first.stage_failed_at or ""
                    )
                    result.first_attempt_error = first.error_message or ""
                    result.first_attempt_error_category = (
                        first.error_category or ""
                    )
                    result.final_attempt_sql = last.sql
                    result.attempts_summary = " | ".join(
                        f"#{a.attempt_number}: "
                        f"{'OK' if a.succeeded else (a.stage_failed_at or '?')}"
                        f"{'/' + a.error_category if a.error_category else ''}"
                        f"{'/hallu=' + a.hallucination_severity if a.hallucination_severity and a.hallucination_severity != 'none' else ''}"
                        f" ({a.latency_ms:.0f}ms)"
                        for a in attempts
                    )

                    # Latency breakdown (Nhóm 5)
                    result.final_attempt_sql_generation_ms = (
                        last.sql_generation_ms
                    )
                    result.final_attempt_db_execution_ms = (
                        last.db_execution_ms
                    )
                    result.total_sql_generation_ms = round(
                        sum(a.sql_generation_ms for a in attempts), 1
                    )
                    result.total_db_execution_ms = round(
                        sum(a.db_execution_ms for a in attempts), 1
                    )

                result.flag_empty_result = (
                    result.success and result.row_count == 0
                )
                result.flag_explanation_table_mismatch = (
                    check_explanation_table_mismatch(
                        result.sql, result.explanation
                    )
                )
                result.flag_slow = result.total_time_s > 60
                result.flag_suspicious_comment = check_suspicious_comment(
                    result.sql
                )
                result.flag_unexpected_table = check_unexpected_table(
                    result.sql, known_tables
                )
                result.flag_self_join = check_self_join(result.sql)
                result.flag_unjoined_filter_table = (
                    check_unjoined_filter_table(result.sql)
                )

            except Exception as e:
                result.total_time_s = round(time.time() - start, 2)
                result.success = False
                result.error = f"EXCEPTION: {e}"
                logger.error(f"  ❌ Exception: {e}", exc_info=True)

            status = "✅" if result.success else "❌"
            flags = []
            if result.flag_empty_result:
                flags.append("EMPTY")
            if result.flag_explanation_table_mismatch:
                flags.append("TABLE_MISMATCH")
            if result.flag_slow:
                flags.append("SLOW")
            if result.flag_suspicious_comment:
                flags.append("SUSPICIOUS_COMMENT")
            if result.flag_unexpected_table:
                flags.append(f"UNEXPECTED_TABLE={result.flag_unexpected_table}")
            if result.flag_self_join:
                flags.append("SELF_JOIN")
            if result.flag_unjoined_filter_table:
                flags.append(
                    f"UNJOINED_FILTER={result.flag_unjoined_filter_table[:40]}"
                )
            flag_str = f" [{', '.join(flags)}]" if flags else ""

            logger.info(
                f"  {status} {result.total_time_s}s | "
                f"rows={result.row_count} | "
                f"trust={result.trust_score:.0f}% | "
                f"attempts={result.num_attempts}{flag_str}"
            )

            # ── Record-level failure taxonomy (Step 5.6) ──────────────
            # Classify WHY a failed question failed, using the richest
            # signal available. For failures caught mid-pipeline we reuse
            # the first-attempt trace; for empty results / flags we map
            # the flag to a standard category.
            if not result.success:
                # stage: prefer the traced stage, else infer from error text
                result.failure_stage = (
                    result.first_attempt_stage_failed_at
                    or _infer_failure_stage(result.error)
                )
                result.error_category = (
                    result.first_attempt_error_category
                    or _infer_error_category(result)
                )
            elif result.flag_empty_result:
                # "succeeded" but 0 rows is a soft failure worth tracking
                result.failure_stage = "response"
                result.error_category = "empty_result"

            results.append(result)

            if delay_between_s > 0 and run_num < total_runs:
                time.sleep(delay_between_s)

    return results


def write_outputs(results: list, output_dir: Path) -> dict:
    """Write CSV, JSON, and summary text outputs. Returns file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_path = output_dir / f"batch_results_{ts}.csv"
    json_path = output_dir / f"batch_results_{ts}.json"
    summary_path = output_dir / f"batch_summary_{ts}.txt"

    # CSV
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow(r.to_csv_row())

    # JSON (full detail)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(r) for r in results], f,
            indent=2, ensure_ascii=False
        )

    # Summary
    n = len(results)
    n_success = sum(1 for r in results if r.success)
    n_empty = sum(1 for r in results if r.flag_empty_result)
    n_mismatch = sum(
        1 for r in results if r.flag_explanation_table_mismatch
    )
    n_slow = sum(1 for r in results if r.flag_slow)
    n_retried = sum(1 for r in results if r.retried)
    n_multi_attempt = sum(1 for r in results if r.num_attempts > 1)
    n_suspicious_comment = sum(
        1 for r in results if r.flag_suspicious_comment
    )
    n_unexpected_table = sum(
        1 for r in results if r.flag_unexpected_table
    )
    n_self_join = sum(1 for r in results if r.flag_self_join)
    n_unjoined_filter = sum(
        1 for r in results if r.flag_unjoined_filter_table
    )
    total_sql_gen = sum(r.total_sql_generation_ms for r in results)
    total_db_exec = sum(r.total_db_execution_ms for r in results)
    total_latency_tracked = total_sql_gen + total_db_exec
    n_hidden_severity = sum(
        1 for r in results
        if r.max_hallucination_severity
        and r.max_hallucination_severity != r.hallucination_severity
        and r.max_hallucination_severity != "none"
    )
    avg_time = sum(r.total_time_s for r in results) / n if n else 0
    avg_trust = (
        sum(r.trust_score for r in results if r.success) /
        max(n_success, 1)
    )

    # Aggregate error_category from first-attempt failures — shows
    # what kinds of errors the retry engine is actually fixing
    from collections import Counter
    first_attempt_categories = Counter(
        r.first_attempt_error_category for r in results
        if r.first_attempt_error_category
    )

    failed_questions = [
        f"  [{r.row_id}] {r.question[:60]} — {r.error[:80]}"
        for r in results if not r.success
    ]
    mismatch_questions = [
        f"  [{r.row_id}] {r.question[:60]}"
        for r in results if r.flag_explanation_table_mismatch
    ]
    suspicious_comment_questions = [
        f"  [{r.row_id}] {r.question[:60]}"
        for r in results if r.flag_suspicious_comment
    ]
    unexpected_table_questions = [
        f"  [{r.row_id}] {r.question[:60]} — tables: {r.flag_unexpected_table}"
        for r in results if r.flag_unexpected_table
    ]
    hidden_severity_questions = [
        f"  [{r.row_id}] {r.question[:60]} — final={r.hallucination_severity}, "
        f"worst_seen={r.max_hallucination_severity} ({r.attempts_summary})"
        for r in results
        if r.max_hallucination_severity
        and r.max_hallucination_severity != r.hallucination_severity
        and r.max_hallucination_severity != "none"
    ]
    self_join_questions = [
        f"  [{r.row_id}] {r.question[:60]}"
        for r in results if r.flag_self_join
    ]
    unjoined_filter_questions = [
        f"  [{r.row_id}] {r.question[:60]} — {r.flag_unjoined_filter_table[:80]}"
        for r in results if r.flag_unjoined_filter_table
    ]

    summary_lines = [
        "=" * 70,
        f"BATCH TEST SUMMARY — {ts}",
        "=" * 70,
        f"Total runs:              {n}",
        f"Successful:              {n_success} ({n_success/n*100:.0f}%)",
        f"Failed:                  {n - n_success}",
        f"Empty results:           {n_empty}",
        f"Explanation/table mismatch: {n_mismatch}",
        f"Slow (>60s):             {n_slow}",
        f"Required retry:          {n_retried}",
        f"Multi-attempt queries:   {n_multi_attempt}",
        f"Suspicious comments:     {n_suspicious_comment}",
        f"Unexpected tables used:  {n_unexpected_table}",
        f"Hidden severity (earlier attempt worse than final): {n_hidden_severity}",
        f"Redundant self-joins:    {n_self_join}",
        f"Unjoined/dangling filter refs: {n_unjoined_filter}",
        "",
        f"Avg response time:       {avg_time:.1f}s",
        f"Avg trust score (success): {avg_trust:.1f}%",
        "",
        f"--- Latency Breakdown (Nhóm 5, summed across all attempts) ---",
        f"Total LLM/SQL-gen time:  {total_sql_gen/1000:.1f}s "
        f"({total_sql_gen/max(total_latency_tracked,1)*100:.0f}% of tracked time)",
        f"Total DB execution time: {total_db_exec/1000:.1f}s "
        f"({total_db_exec/max(total_latency_tracked,1)*100:.0f}% of tracked time)",
        "",
    ]
    if first_attempt_categories:
        summary_lines.append("First-attempt error categories (what retry fixes):")
        for cat, count in first_attempt_categories.most_common():
            summary_lines.append(f"  {cat}: {count}")
        summary_lines.append("")
    if failed_questions:
        summary_lines.append("FAILED QUESTIONS:")
        summary_lines.extend(failed_questions)
        summary_lines.append("")
    if mismatch_questions:
        summary_lines.append("TABLE MISMATCH (explanation bug pattern):")
        summary_lines.extend(mismatch_questions)
        summary_lines.append("")
    if unexpected_table_questions:
        summary_lines.append(
            "UNEXPECTED TABLES (hallucinated table names, e.g. sh_order):"
        )
        summary_lines.extend(unexpected_table_questions)
        summary_lines.append("")
    if suspicious_comment_questions:
        summary_lines.append(
            "SUSPICIOUS COMMENTS (model 'justifying' a filter — review manually):"
        )
        summary_lines.extend(suspicious_comment_questions)
        summary_lines.append("")
    if hidden_severity_questions:
        summary_lines.append(
            "HIDDEN SEVERITY (an earlier attempt hallucinated worse "
            "than the final accepted SQL — retry masked it):"
        )
        summary_lines.extend(hidden_severity_questions)
        summary_lines.append("")
    if self_join_questions:
        summary_lines.append(
            "REDUNDANT SELF-JOINS (same table joined to itself "
            "under two aliases — wasted work, possible alias confusion):"
        )
        summary_lines.extend(self_join_questions)
        summary_lines.append("")
    if unjoined_filter_questions:
        summary_lines.append(
            "UNJOINED/DANGLING FILTER REFERENCES (column qualifier "
            "doesn't match any FROM/JOIN table — phantom prefix or "
            "malformed reference):"
        )
        summary_lines.extend(unjoined_filter_questions)
        summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("\n" + summary_text)

    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "summary": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run questions against the ERP42 query pipeline."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to CSV file with questions (columns: id,question,language,category)"
    )
    parser.add_argument(
        "--use-ground-truth", action="store_true",
        help="Use the built-in GROUND_TRUTH set from benchmark.py instead of --input"
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run each question N times to measure stability (default: 1)"
    )
    parser.add_argument(
        "--role", type=str, default="analyst",
        help="RBAC role to use for the test run (default: analyst)"
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Allow query cache (default: off, to force fresh generation each time)"
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Seconds to wait between queries (default: 1.0)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Where to write results (default: <project_root>/logs)"
    )
    parser.add_argument(
        "--run-id", type=str, default=None,
        help="Save this run to the Test Run Repository under this id "
             "(e.g. baseline_w1). Enables run-vs-run comparison. "
             "If omitted, an auto id (run_YYYYMMDD_HHMM) is used."
    )
    parser.add_argument(
        "--run-label", type=str, default="",
        help="Human-readable label for the run (e.g. 'Week 1 baseline, real schema')"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do NOT persist to the Test Run Repository (CSV/JSON only)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.input and not args.use_ground_truth:
        parser.error("Provide --input <csv> or --use-ground-truth")

    if args.use_ground_truth:
        questions = load_ground_truth_questions()
        logger.info(f"Loaded {len(questions)} questions from GROUND_TRUTH")
    else:
        questions = load_questions_from_csv(args.input)
        logger.info(f"Loaded {len(questions)} questions from {args.input}")

    if not questions:
        logger.error("No questions loaded — aborting.")
        return

    total_runs = len(questions) * args.repeat
    logger.info(
        f"Starting batch run: {len(questions)} questions × "
        f"{args.repeat} repeat(s) = {total_runs} total runs"
    )
    est_minutes = total_runs * 60 / 60  # rough placeholder, real time varies
    logger.info(
        "Note: at ~1-3 min/query observed in testing, this may take a "
        "while. The script logs progress as it goes — safe to leave running."
    )

    results = run_batch(
        questions=questions,
        repeat=args.repeat,
        role=args.role,
        use_cache=args.use_cache,
        delay_between_s=args.delay,
    )

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).resolve().parents[1] / "logs"
    )
    paths = write_outputs(results, output_dir)

    logger.info("Batch run complete. Output files:")
    for k, v in paths.items():
        logger.info(f"  {k}: {v}")

    # ── Persist to Test Run Repository (Step 1.11) ───────────────────
    # Enables Baseline → Fix → Re-test comparison across weeks.
    if not args.no_save:
        try:
            from erp42_analytics.scripts.test_run_repository import (
                TestRunRepository, make_run_id,
            )
            from erp42_analytics.core.config import config

            run_id = args.run_id or make_run_id()
            repo = TestRunRepository()
            repo.save_run(
                run_id=run_id,
                results=results,
                label=args.run_label,
                runtime_mode=getattr(config.runtime, "mode", ""),
                llm_model=getattr(config.model, "llm_model", ""),
                database_name=getattr(config.db, "database", ""),
            )
            logger.info(
                "Saved to Test Run Repository as run_id='%s'. Compare later with:\n"
                "  python -m erp42_analytics.scripts.run_comparison "
                "--baseline %s --candidate <next_run_id>",
                run_id, run_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not save to Test Run Repository (CSV/JSON still written): %s",
                e,
            )


if __name__ == "__main__":
    main()
