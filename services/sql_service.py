"""
services/sql_service.py — SQL Intelligence Service
Migrates: sql_guardrail.py + sql_executor.py + prompt_builder.py + evaluator.py
Covers: Stage 4 (SQL Intelligence Engine — Steps 4.1, 4.3, 4.5, 4.9)

Responsibilities:
- Build LLM prompts
- Auto-fix common SQL errors (guardrails)
- Execute SQL safely
- Evaluate result quality
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.core.models import SQLResult, QueryStatus, GoldenSQL

logger = logging.getLogger(__name__)


# ─── System Prompt ────────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = f"""You are a German/English business assistant that converts natural language questions into pure SQL code for Microsoft SQL Server (MSSQL).

Your task:
Generate ONE correct MSSQL SELECT query to answer the user question.

Rules:
- Output ONLY the SQL query, nothing else. No comments, no explanations.
- Start with SELECT or WITH.
- ALWAYS use {config.db.full_table_prefix}.tablename format — database is ALWAYS {config.db.database}.
- NEVER modify the database name — {config.db.database} is the ONLY valid name.
- Use proper MSSQL syntax and EXACT table/column names from the schema.
- For NULL values, add IS NOT NULL in WHERE clause.
- NEVER add date filters unless user explicitly mentions a specific date or period.
- NEVER use placeholder values like YYYY-MM-DD, START_DATE, END_DATE.
- If unsure about a table or column, DO NOT GUESS.
- Only use tables and columns that exist in the schema below.
- Column aliases: ASCII only — use ae/oe/ue/ss instead of ä/ö/ü/ß.
- When listing records, SELECT only 5-8 important columns, never SELECT *.
- TOP N rule: ONLY use TOP N when user explicitly asks for top/best/first/N records.
- NEVER add TOP N for: alle/all/every/jeder/each queries or aggregate queries.
- For COUNT/SUM/AVG queries: NEVER use TOP N.
- Use TOP N syntax — NEVER use OFFSET/FETCH or LIMIT.
"""


class PromptService:
    """Builds LLM prompts with schema, examples, and instructions."""

    def build_prompt(
        self,
        question: str,
        schema_text: str,
        golden_examples: list = None,
        instructions: list = None,
        current_time: bool = False,
        intent_hints: str = None,
    ) -> str:
        """Build complete prompt for LLM."""
        parts = [BASE_SYSTEM_PROMPT]

        if current_time:
            now = datetime.now()
            parts.append(
                f"\nCurrent date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )

        if instructions:
            parts.append("\nDatabase administrator instructions (MUST follow):")
            for i, inst in enumerate(instructions, 1):
                text = inst.get("instruction", inst) if isinstance(inst, dict) else inst
                parts.append(f"{i}) {text}")
            parts.append("")

        if intent_hints:
            parts.append(f"\n{intent_hints}\n")

        if golden_examples:
            parts.append(
                "Similar verified Question/SQL pairs — use as reference:"
            )
            for ex in golden_examples:
                q = ex.question if isinstance(ex, GoldenSQL) else ex.get("prompt_text", "")
                sql = ex.sql if isinstance(ex, GoldenSQL) else ex.get("sql", "")
                parts.append(f"Question: {q}")
                parts.append(f"SQL:\n```sql\n{sql}\n```")
                parts.append("")

        parts.append("Database schema (use EXACT table and column names):")
        parts.append(schema_text)
        parts.append("")
        parts.append(f"User question: {question}")

        return "\n".join(parts)


class SQLGuardrailService:
    """
    Auto-fixes common SQL errors before execution.
    Stage 4.3 — SQL Guardrail Engine.
    """

    BLOCKED = [
        "DROP", "DELETE", "UPDATE", "INSERT",
        "ALTER", "TRUNCATE", "EXEC", "EXECUTE",
        "XP_", "SP_EXECUTESQL",
    ]

    CORE_TABLES = [
        "auftrag", "kunde", "rechnung", "lieferan", "bestell",
        "lager", "fibubuch", "mitarbei", "projekt", "angebot"
    ]

    # Defense-in-depth: same list as semantic_service.py's SENSITIVE_TABLES.
    # Retrieval already hard-excludes these from candidates, but this is a
    # second, independent check at the execution boundary — if any future
    # change to keyword boosting, golden examples, or manual override ever
    # let a sensitive table slip through, this still blocks it before
    # execution. Per ibes classification: benutzer is Sensitive=YES with
    # masked column Passwort, access restricted to Geschäftsleitung only.
    SENSITIVE_TABLES = {
        "benutzer": ["Passwort"],
    }

    # Tokens that mean the raw LLM output stopped mid-expression — i.e.
    # the model hit its token budget before finishing the clause it was
    # writing. If the RAW sql (before any auto-fix) ends on one of these,
    # no amount of paren-balancing can reconstruct the missing logic —
    # the correct action is to fail fast and retry, not to cosmetically
    # "close" it (which is what _fix_incomplete_cte used to do, producing
    # syntactically-plausible-but-still-broken SQL like "NOT EXISTS ();").
    _TRUNCATION_TRAILERS = re.compile(
        r"(?i)("
        r"\(\s*$"                                      # dangling open paren
        r"|\b(AND|OR|WHERE|ON|IN|BETWEEN|LIKE|AS|FROM|SELECT|JOIN)\s*$"  # dangling keyword
        r"|,\s*$"                                       # dangling comma
        r"|(>=|<=|<>|!=|=|>|<)\s*$"                     # dangling comparison operator
        r")"
    )

    def validate(self, sql: str, allowed_tables: list = None) -> dict:
        """
        Validate and auto-fix SQL.
        Returns: {valid, cleaned_sql, warnings, error}
        """
        warnings = []

        # 1. Extract SQL from markdown
        sql = self._extract_sql(sql)

        # 1b. Detect truncation BEFORE any cosmetic fixing. This must run
        #     on the freshly-extracted raw body, before _fix_incomplete_cte
        #     gets a chance to paper over it with balanced parens.
        if self._looks_truncated(sql):
            return {
                "valid": False,
                "cleaned_sql": sql,
                "warnings": [],
                "error": (
                    "truncated_generation: model output stopped mid-clause "
                    "(likely hit num_predict token limit) — retry needs a "
                    "shorter query or higher token budget, not a syntax fix"
                ),
            }

        # 2. Check for dangerous statements
        sql_upper = sql.upper()
        for stmt in self.BLOCKED:
            if re.search(rf"\b{stmt}\b", sql_upper):
                return {
                    "valid": False,
                    "cleaned_sql": sql,
                    "warnings": [],
                    "error": f"Blocked statement: {stmt}"
                }

        # 2b. Block sensitive tables/columns outright — defense-in-depth
        #     alongside the retrieval-level exclusion in semantic_service.py.
        #     Checked by word-boundary regex so "benutzer" as a substring
        #     of some other identifier doesn't false-positive.
        for sensitive_table, masked_cols in self.SENSITIVE_TABLES.items():
            if re.search(rf"\b{sensitive_table}\b", sql, re.IGNORECASE):
                return {
                    "valid": False,
                    "cleaned_sql": sql,
                    "warnings": [],
                    "error": (
                        f"Blocked: query references sensitive table "
                        f"'{sensitive_table}' (masked columns: "
                        f"{masked_cols}) — access restricted per ibes "
                        f"table classification"
                    ),
                }

        # 3. Auto-fix syntax issues
        sql = self._fix_database_name(sql)
        sql = self._fix_missing_prefix(sql)
        sql = self._fix_date_placeholders(sql)
        sql = self._fix_top_zero(sql)
        sql = self._fix_aggregate_top(sql)
        sql = self._fix_incomplete_cte(sql)
        sql = self._fix_multiple_selects(sql)
        sql = self._fix_empty_where(sql)
        sql = self._fix_quoted_aliases(sql)

        # 4. Validate tables exist
        if allowed_tables:
            warnings.extend(self._check_tables(sql, allowed_tables))

        return {
            "valid": True,
            "cleaned_sql": sql.strip(),
            "warnings": warnings,
            "error": None,
        }

    def _looks_truncated(self, sql: str) -> bool:
        """
        True if the raw (pre-fix) SQL body ends on an incomplete
        expression — the signature of hitting the LLM's token limit
        mid-generation, as opposed to a genuine syntax mistake.
        """
        body = sql.strip().rstrip(";").rstrip()
        if not body:
            return False
        if self._TRUNCATION_TRAILERS.search(body):
            return True
        # Belt-and-suspenders: "(" immediately followed by ";" (with only
        # whitespace between) is never valid MSSQL syntax under any
        # circumstance — an empty parenthesized expression terminated by
        # a statement separator can't parse. Observed repeatedly in logs
        # as the smoking-gun signature of truncated generation, so treat
        # it as truncation regardless of where in the string it occurs
        # (not just at the very end) — some fixers may have already run
        # by the time this is checked in unexpected code paths.
        if re.search(r"\(\s*;", sql):
            return True
        return False

    def _extract_sql(self, sql: str) -> str:
        """
        Extract SQL from markdown code blocks.
        Handles CTE correctly — does NOT cut at first semicolon.
        """
        # Remove markdown code blocks
        for pattern in [r"```sql(.*?)```", r"```(.*?)```"]:
            matches = re.findall(pattern, sql, re.DOTALL | re.IGNORECASE)
            if matches:
                sql = matches[0].strip()
                break

        # Extract from first SELECT or WITH
        match = re.search(r"((?:WITH|SELECT)[\s\S]+)", sql, re.IGNORECASE)
        if match:
            sql = match.group(1).strip()

        # For CTEs: do NOT cut at semicolon inside CTE body
        # Only cut at semicolon if it's a proper statement terminator
        is_cte = bool(re.match(r"(?i)^\s*WITH\s+\w+\s+AS\s*\(", sql))
        if not is_cte and ";" in sql:
            sql = sql[:sql.index(";") + 1]

        return sql.strip()

    def _fix_database_name(self, sql: str) -> str:
        """Fix hallucinated database names."""
        db = config.db.database
        return re.sub(
            r'(?i)ERP42\w+\.dbo\.',
            f'{db}.dbo.',
            sql
        )

    def _fix_missing_prefix(self, sql: str) -> str:
        """Add ERP42test.dbo prefix to bare core table names."""
        prefix = config.db.full_table_prefix
        table_pattern = "|".join(self.CORE_TABLES)
        sql = re.sub(
            rf'(?i)\bFROM\s+({table_pattern})\b(?!\.)(?!\w)',
            lambda m: f'FROM {prefix}.{m.group(1).lower()}',
            sql
        )
        sql = re.sub(
            rf'(?i)\bJOIN\s+({table_pattern})\b(?!\.)(?!\w)',
            lambda m: f'JOIN {prefix}.{m.group(1).lower()}',
            sql
        )
        return sql

    def _fix_date_placeholders(self, sql: str) -> str:
        """Remove date placeholder conditions."""
        # Remove inline comments with placeholders
        sql = re.sub(
            r'(?i)--[^\n]*?(YYYY|Replace|if needed|adjust)[^\n]*\n',
            '\n', sql
        )
        sql = re.sub(
            r'(?i)--[^\n]*?(YYYY|Replace|if needed|adjust)[^\n]*$',
            '', sql
        )
        # Remove AND conditions with placeholders
        for pattern in [
            r'(?i)\s+AND\s+\w+\.?\w+\s*[><=!]+\s*\'YYYY-MM-DD\'',
            r'(?i)\s+AND\s+\w+\.?\w+\s*[><=!]+\s*\'START_DATE\'',
            r'(?i)\s+AND\s+\w+\.?\w+\s*[><=!]+\s*\'END_DATE\'',
            r'(?i)\s+AND\s+\w+\.?\w+\s+BETWEEN\s+\'YYYY[^\']*\'\s+AND\s+\'YYYY[^\']*\'',
        ]:
            sql = re.sub(pattern, '', sql)
        # Fix WHERE with placeholder → remove or keep rest
        sql = re.sub(
            r'(?i)\s+WHERE\s+\w+\.?\w+\s*[><=!]+\s*\'YYYY-MM-DD\'\s+AND\s+',
            ' WHERE ', sql
        )
        sql = re.sub(
            r'(?i)\s+WHERE\s+\w+\.?\w+\s*[><=!]+\s*\'YYYY-MM-DD\'',
            '', sql
        )
        # Fix dangling AND → WHERE
        sql = re.sub(
            r'(?i)(FROM\s+\S+(?:\s+\w+)?)\s+AND\s+',
            r'\1 WHERE ', sql
        )
        return sql

    def _fix_top_zero(self, sql: str) -> str:
        """Replace TOP 0 with TOP 100."""
        return re.sub(
            r'(?i)SELECT\s+TOP\s+\(?0\)?\s+',
            'SELECT TOP 100 ',
            sql, count=1
        )

    def _fix_aggregate_top(self, sql: str) -> str:
        """Remove TOP N from aggregate queries."""
        if re.search(
            r'(?i)SELECT\s+TOP\s+\d+\s+(COUNT|SUM|AVG|MIN|MAX)', sql
        ):
            sql = re.sub(
                r'(?i)SELECT\s+TOP\s+\d+\s+',
                'SELECT ', sql, count=1
            )
        return sql

    def _fix_multiple_selects(self, sql: str) -> str:
        """Take first complete SELECT if multiple found."""
        selects = list(re.finditer(r"\bSELECT\b", sql, re.IGNORECASE))
        if len(selects) > 1:
            sql = sql[:selects[1].start()].strip().rstrip(")").rstrip()
            if not sql.endswith(";"):
                sql += ";"
        return sql

    def _fix_empty_where(self, sql: str) -> str:
        """Remove empty WHERE clauses."""
        sql = re.sub(r'(?i)\s+WHERE\s*;', ';', sql)
        sql = re.sub(
            r'(?i)\s+WHERE\s+(GROUP BY|ORDER BY|HAVING|;)',
            r' \1', sql
        )
        sql = re.sub(r'(?i)\s+WHERE\s*$', '', sql)
        sql = re.sub(r'(?i)WHERE\s+WHERE', 'WHERE', sql)
        return sql

    def _fix_incomplete_cte(self, sql: str) -> str:
        """
        Fix incomplete CTE — model often generates CTE without outer SELECT.
        WITH foo AS (SELECT ... FROM table;  ← missing ) and outer SELECT

        Note: by the time this runs, validate()'s _looks_truncated() check
        has already rejected genuinely truncated generations (dangling
        paren/operator at the very end). What reaches this function is
        either a complete-but-malformed CTE (missing outer SELECT — fixable)
        or already-valid SQL — never a mid-expression cut, so this paren-
        balancing no longer masks truncation as a "successful" fix.
        """
        sql_stripped = sql.strip().rstrip(';').strip()

        # Detect CTE pattern
        if not re.match(r'(?i)^WITH\s+\w+\s+AS\s*\(', sql_stripped):
            return sql

        # Count parentheses balance
        open_count = sql_stripped.count('(')
        close_count = sql_stripped.count(')')

        if open_count > close_count:
            # Add missing closing parentheses
            sql_stripped += ')' * (open_count - close_count)

        # Check if there is an outer SELECT after the CTE
        # Pattern: WITH cte AS (...) SELECT ...
        cte_pattern = r'(?i)^WITH\s+(\w+)\s+AS\s*\(.*?\)\s*SELECT'
        if not re.search(cte_pattern, sql_stripped, re.DOTALL):
            # Extract CTE name and convert to simple SELECT
            cte_name_match = re.match(r'(?i)^WITH\s+(\w+)\s+AS\s*\(', sql_stripped)
            if cte_name_match:
                # Extract inner SELECT from CTE body
                inner_match = re.search(
                    r'(?i)WITH\s+\w+\s+AS\s*\((.*)\)',
                    sql_stripped, re.DOTALL
                )
                if inner_match:
                    inner_sql = inner_match.group(1).strip()
                    # Return the inner SELECT directly
                    return inner_sql + ';'

        return sql_stripped + ';'

    def _fix_quoted_aliases(self, sql: str) -> str:
        """
        Fix double-quoted aliases — MSSQL uses [] not double quotes.
        SELECT col AS "Name" → SELECT col AS [Name]
        """
        # Replace "alias" with [alias] in AS clauses
        sql = re.sub(
            r'(?i)AS\s+"([^"]+)"',
            lambda m: f'AS [{m.group(1)}]',
            sql
        )
        return sql

    def _check_tables(self, sql: str, allowed_tables: list) -> list:
        """Check if tables in SQL exist in schema."""
        warnings = []
        try:
            from sql_metadata import Parser
            tables = Parser(sql).tables
            allowed_lower = [t.lower() for t in allowed_tables]
            for table in tables:
                table_clean = table.split(".")[-1].lower().strip("[]")
                if table_clean and table_clean not in allowed_lower:
                    warnings.append(
                        f"Table '{table_clean}' not found in scanned schema. "
                        f"AI may have hallucinated this table name."
                    )
        except Exception as e:
            logger.debug(f"Table check failed: {e}")
        return warnings


class SQLExecutorService:
    """
    Executes SQL safely against ERP database.
    Stage 4.5 — SQL Execution Engine.
    """

    def __init__(self):
        from erp42_analytics.core.database import get_mssql
        self.mssql = get_mssql()

    def execute(
        self,
        sql: str,
        max_rows: int = None,
        timeout: int = None,
        allowed_tables: list = None,
    ) -> SQLResult:
        """Execute SQL and return structured result."""
        max_rows = max_rows or config.query.max_rows
        timeout = timeout or config.query.exec_timeout_seconds
        start = time.time()

        result = SQLResult(question="", sql=sql)

        try:
            columns, rows = self.mssql.execute(sql, timeout=timeout)

            if len(rows) > max_rows:
                result.warnings = [
                    f"Result truncated to {max_rows} rows "
                    f"(total: {len(rows)})"
                ]
                rows = rows[:max_rows]

            result.columns = columns
            result.rows = rows
            result.row_count = len(rows)
            result.exec_time = round(time.time() - start, 2)
            result.status = QueryStatus.SUCCESS

        except Exception as e:
            result.status = QueryStatus.FAILED
            result.exec_time = round(time.time() - start, 2)
            result.error = f"Query execution failed: {str(e)}"
            logger.error(f"SQL execution error: {e}")

        return result


class EvaluatorService:
    """
    Evaluates SQL quality and builds trust score.
    Stage 5 — Validation & Trust System (Step 5.6 Confidence Scoring, basic implementation).
    """

    def evaluate(self, sql: str, result: SQLResult, question: str) -> dict:
        """
        Rule-based SQL quality evaluation.
        Returns confidence score and metadata.
        """
        score = 100
        issues = []

        # Check execution success
        if result.status != QueryStatus.SUCCESS:
            return {
                "score": 0,
                "confidence": "Low",
                "tables_used": [],
                "issues": ["SQL execution failed"],
            }

        # Check empty results
        if result.row_count == 0:
            score -= 30
            issues.append("Query returned no results")

        # Check schema validation warnings
        schema_warnings = [
            w for w in result.warnings
            if "not found in scanned schema" in w
        ]
        if schema_warnings:
            score -= 40
            issues.extend(schema_warnings)

        # Check for placeholder values still in SQL
        if "YYYY" in sql or "START_DATE" in sql or "END_DATE" in sql:
            score -= 50
            issues.append("SQL contains placeholder values")

        # Check result size sanity
        if result.row_count > 10000:
            score -= 10
            issues.append(f"Large result set: {result.row_count} rows")

        # Extract tables used
        tables_used = []
        try:
            from sql_metadata import Parser
            tables_used = [
                t.split(".")[-1].lower().strip("[]")
                for t in Parser(sql).tables
            ]
        except Exception:
            pass

        score = max(0, score)
        confidence = (
            "High" if score >= 80
            else "Medium" if score >= 50
            else "Low"
        )

        return {
            "score": score,
            "confidence": confidence,
            "tables_used": tables_used,
            "issues": issues,
        }
