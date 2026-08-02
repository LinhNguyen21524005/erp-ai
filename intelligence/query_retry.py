"""
intelligence/query_retry.py — SQL Error Handling & Retry Engine
Stage 4.8: AI tự sửa query khi lỗi (self-healing SQL generation).

Responsibilities:
- Analyze SQL execution errors (column not found, table not found,
  timeout, datatype mismatch, syntax error)
- Classify the error into a known category
- Build a corrective retry prompt that tells the LLM exactly what
  went wrong and how to fix it
- Cap retries to avoid infinite loops / wasted LLM calls

This sits between Step 13 (Execute SQL) and the final response —
if execution fails, the pipeline gets one (configurable) chance to
retry with a simplified/corrected query before giving up.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    COLUMN_NOT_FOUND = "column_not_found"
    TABLE_NOT_FOUND = "table_not_found"
    SYNTAX_ERROR = "syntax_error"
    TIMEOUT = "timeout"
    DATATYPE_MISMATCH = "datatype_mismatch"
    JOIN_ERROR = "join_error"
    PERMISSION_DENIED = "permission_denied"
    UNKNOWN = "unknown"


@dataclass
class ErrorAnalysis:
    """Result of analyzing a failed SQL execution."""
    category: ErrorCategory = ErrorCategory.UNKNOWN
    raw_error: str = ""
    problem_identifier: Optional[str] = None  # e.g. the bad column/table name
    retry_recommended: bool = True
    retry_strategy: str = ""
    suggested_fix_hint: str = ""


@dataclass
class RetryAttempt:
    """A single retry attempt record."""
    attempt_number: int
    sql: str
    error: Optional[str] = None
    success: bool = False


@dataclass
class RetryResult:
    """Final result of the retry process."""
    final_sql: str = ""
    success: bool = False
    attempts: list = field(default_factory=list)
    gave_up_reason: Optional[str] = None


class ErrorAnalyzer:
    """
    Stage 4.8 — Error Analyzer.
    Classifies SQL/DB execution errors so the retry engine knows
    what kind of fix to attempt.
    """

    # MSSQL/ODBC error patterns → category
    PATTERNS = [
        (ErrorCategory.COLUMN_NOT_FOUND,
         r"(?i)invalid column name '?([^'\s]+)'?"),
        (ErrorCategory.TABLE_NOT_FOUND,
         r"(?i)invalid object name '?([^'\s]+)'?"),
        (ErrorCategory.SYNTAX_ERROR,
         r"(?i)incorrect syntax near '?([^'\s]+)'?"),
        (ErrorCategory.TIMEOUT,
         r"(?i)(timeout|timed out|query.*cancel)"),
        (ErrorCategory.DATATYPE_MISMATCH,
         r"(?i)(conversion failed|cannot convert|datatype mismatch)"),
        (ErrorCategory.JOIN_ERROR,
         r"(?i)(ambiguous column name|the multi-part identifier)"),
        (ErrorCategory.PERMISSION_DENIED,
         r"(?i)(permission denied|access.*denied|the select permission)"),
    ]

    def analyze(self, error_message: str) -> ErrorAnalysis:
        """Classify an error message and suggest a retry strategy."""
        if not error_message:
            return ErrorAnalysis(
                category=ErrorCategory.UNKNOWN,
                raw_error="",
                retry_recommended=False,
            )

        for category, pattern in self.PATTERNS:
            match = re.search(pattern, error_message)
            if match:
                identifier = match.group(1) if match.groups() else None
                return self._build_analysis(
                    category, error_message, identifier
                )

        return ErrorAnalysis(
            category=ErrorCategory.UNKNOWN,
            raw_error=error_message,
            retry_recommended=True,
            retry_strategy="generic_retry",
            suggested_fix_hint=(
                "The previous query failed. Re-examine the schema "
                "carefully and regenerate the SQL from scratch."
            ),
        )

    def _build_analysis(
        self,
        category: ErrorCategory,
        error_message: str,
        identifier: Optional[str],
    ) -> ErrorAnalysis:
        hints = {
            ErrorCategory.COLUMN_NOT_FOUND: (
                f"The column '{identifier}' does not exist. "
                f"Check the schema for the correct column name — it may "
                f"be spelled differently or belong to a different table. "
                f"Use ONLY columns listed in the schema."
            ),
            ErrorCategory.TABLE_NOT_FOUND: (
                f"The table '{identifier}' does not exist. "
                f"Use ONLY table names listed in the schema, with the "
                f"exact prefix shown (e.g. database.dbo.tablename)."
            ),
            ErrorCategory.SYNTAX_ERROR: (
                f"There is a SQL syntax error near '{identifier}'. "
                f"Check for missing commas, unmatched parentheses, or "
                f"incomplete clauses. Produce a single complete, valid "
                f"SQL Server query."
            ),
            ErrorCategory.TIMEOUT: (
                "The query took too long to execute. Simplify the query: "
                "reduce the number of joins, add more selective filters, "
                "or avoid scanning large tables without a WHERE clause."
            ),
            ErrorCategory.DATATYPE_MISMATCH: (
                "There is a datatype mismatch in a comparison or "
                "conversion. Check that filters compare compatible types "
                "(e.g. don't compare a date column to a plain string "
                "without proper formatting)."
            ),
            ErrorCategory.JOIN_ERROR: (
                f"There is an ambiguous or invalid join "
                f"('{identifier}'). Qualify all column references with "
                f"their table alias, and ensure every joined table has "
                f"a valid join condition."
            ),
            ErrorCategory.PERMISSION_DENIED: (
                "Access to this table/column is denied. Do not retry "
                "with the same table — choose an alternative table the "
                "user's role is permitted to access."
            ),
        }

        # Permission errors should not be retried — it's a policy issue
        retry_recommended = category != ErrorCategory.PERMISSION_DENIED

        return ErrorAnalysis(
            category=category,
            raw_error=error_message,
            problem_identifier=identifier,
            retry_recommended=retry_recommended,
            retry_strategy=category.value,
            suggested_fix_hint=hints.get(category, ""),
        )


class QueryRetryEngine:
    """
    Stage 4.8 — Query Failure Recovery.

    Orchestrates the retry loop: analyze error → build corrective
    prompt → re-call LLM → re-validate → re-execute, up to max_retries.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.analyzer = ErrorAnalyzer()

    def build_retry_prompt(
        self,
        original_prompt: str,
        failed_sql: str,
        error_analysis: ErrorAnalysis,
        attempt_number: int,
    ) -> str:
        """
        Build a corrective prompt that includes the failed SQL,
        the error, and a specific fix hint.
        """
        retry_section = f"""

─── PREVIOUS ATTEMPT FAILED (retry {attempt_number}/{self.max_retries}) ───
Your previous SQL was:
```sql
{failed_sql}
```

It failed with this error:
{error_analysis.raw_error}

Fix hint: {error_analysis.suggested_fix_hint}

Generate a NEW, corrected SQL query that fixes this issue. Re-check the
schema carefully. Output ONLY the corrected SQL query, nothing else.
─────────────────────────────────────────────────────────────────────
"""
        return original_prompt + retry_section

    def should_retry(
        self, error_analysis: ErrorAnalysis, attempt_number: int
    ) -> bool:
        """Decide whether another retry attempt is worthwhile."""
        if attempt_number >= self.max_retries:
            return False
        return error_analysis.retry_recommended

    def analyze_failure(self, error_message: str) -> ErrorAnalysis:
        """Public entry point to classify a failure."""
        analysis = self.analyzer.analyze(error_message)
        logger.info(
            f"SQL failure classified as '{analysis.category.value}': "
            f"{error_message[:80]}"
        )
        return analysis


# ─── Singleton ────────────────────────────────────────────────────────────────
_query_retry_engine: Optional[QueryRetryEngine] = None


def get_query_retry_engine() -> QueryRetryEngine:
    global _query_retry_engine
    if _query_retry_engine is None:
        from erp42_analytics.core.config import config
        max_retries = getattr(config.query, "max_sql_retries", 2)
        _query_retry_engine = QueryRetryEngine(max_retries=max_retries)
    return _query_retry_engine
