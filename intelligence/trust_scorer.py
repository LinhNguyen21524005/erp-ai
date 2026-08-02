"""
intelligence/trust_scorer.py — Enterprise Trust Score System
Stage 5.6: Compute multi-dimensional trust score for query results.

Trust Score Components:
1. Retrieval Confidence — how well tables were matched
2. SQL Reliability — SQL quality and validation
3. Result Consistency — result sanity checks
4. Data Freshness — how recent the data is
5. Join Confidence — confidence in JOIN relationships

Overall Trust Score = weighted average of components (0-100)
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from erp42_analytics.core.models import TrustScore, SQLResult, QueryStatus, QueryIntent

logger = logging.getLogger(__name__)


@dataclass
class TrustBreakdown:
    """Detailed breakdown of trust score components."""
    overall: float = 0.0
    retrieval_confidence: float = 0.0
    sql_reliability: float = 0.0
    result_consistency: float = 0.0
    data_freshness: float = 100.0
    join_confidence: float = 100.0

    # Explanations for each component
    retrieval_explanation: str = ""
    sql_explanation: str = ""
    result_explanation: str = ""
    issues: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    def to_trust_score(self) -> TrustScore:
        return TrustScore(
            overall=self.overall,
            retrieval_confidence=self.retrieval_confidence,
            sql_reliability=self.sql_reliability,
            validation_score=self.result_consistency,
            data_freshness=self.data_freshness,
            join_confidence=self.join_confidence,
        )

    @property
    def explanation_text(self) -> str:
        """Human-readable explanation of trust score."""
        parts = []
        if self.issues:
            parts.append("Issues: " + "; ".join(self.issues))
        if self.warnings:
            parts.append("Warnings: " + "; ".join(self.warnings))
        if not parts:
            parts.append("All checks passed.")
        return " | ".join(parts)


# ─── Weight Configuration ─────────────────────────────────────────────────────
WEIGHTS = {
    "retrieval_confidence": 0.25,
    "sql_reliability":      0.35,
    "result_consistency":   0.25,
    "data_freshness":       0.10,
    "join_confidence":      0.05,
}


class TrustScorer:
    """
    Computes enterprise-grade trust score for query results.
    Used to help users understand how much to trust AI-generated answers.
    """

    def compute(
        self,
        sql: str,
        result: SQLResult,
        intent: Optional[QueryIntent] = None,
        table_names: list = None,
        validation_warnings: list = None,
    ) -> TrustBreakdown:
        """
        Compute full trust breakdown.
        Main entry point.
        """
        breakdown = TrustBreakdown()

        # Component 1: Retrieval confidence
        breakdown.retrieval_confidence = self._score_retrieval(
            table_names or [], intent
        )
        breakdown.retrieval_explanation = self._explain_retrieval(
            table_names or []
        )

        # Component 2: SQL reliability
        sql_score, sql_issues = self._score_sql(
            sql, validation_warnings or []
        )
        breakdown.sql_reliability = sql_score
        breakdown.sql_explanation = "; ".join(sql_issues) if sql_issues else "SQL looks correct"
        breakdown.issues.extend(sql_issues)

        # Component 3: Result consistency
        result_score, result_warnings = self._score_result(result, intent)
        breakdown.result_consistency = result_score
        breakdown.warnings.extend(result_warnings)

        # Component 4: Data freshness (always 100 for live queries)
        breakdown.data_freshness = 100.0

        # Component 5: Join confidence
        breakdown.join_confidence = self._score_joins(sql, table_names or [])

        # Weighted average
        breakdown.overall = round(
            breakdown.retrieval_confidence * WEIGHTS["retrieval_confidence"] +
            breakdown.sql_reliability * WEIGHTS["sql_reliability"] +
            breakdown.result_consistency * WEIGHTS["result_consistency"] +
            breakdown.data_freshness * WEIGHTS["data_freshness"] +
            breakdown.join_confidence * WEIGHTS["join_confidence"],
            1
        )

        logger.debug(
            f"Trust score: {breakdown.overall:.1f} "
            f"(ret:{breakdown.retrieval_confidence:.0f} "
            f"sql:{breakdown.sql_reliability:.0f} "
            f"res:{breakdown.result_consistency:.0f})"
        )

        return breakdown

    # ─── Component Scorers ────────────────────────────────────────────────────

    def _score_retrieval(
        self, table_names: list, intent: Optional[QueryIntent]
    ) -> float:
        """Score how well relevant tables were found."""
        if not table_names:
            return 30.0  # No tables found — low confidence

        score = 80.0  # Base score for finding tables

        # Bonus if intent confirms tables
        if intent and intent.dimensions:
            overlap = set(table_names) & set(intent.dimensions)
            if overlap:
                score += min(20.0, len(overlap) * 10)

        return min(100.0, score)

    def _explain_retrieval(self, table_names: list) -> str:
        if not table_names:
            return "No relevant tables found"
        return f"Found tables: {', '.join(table_names)}"

    def _score_sql(self, sql: str, warnings: list) -> tuple:
        """Score SQL quality. Returns (score, issues)."""
        score = 100.0
        issues = []

        # Penalize for schema warnings (hallucinated tables)
        schema_warns = [
            w for w in warnings
            if "not found in scanned schema" in w.lower() or
               "hallucinated" in w.lower()
        ]
        if schema_warns:
            score -= 40.0
            issues.append(f"Possible hallucinated table(s)")

        # Penalize for placeholder values
        if re.search(r'YYYY|START_DATE|END_DATE', sql, re.IGNORECASE):
            score -= 50.0
            issues.append("SQL contains placeholder values")

        # Penalize for SELECT *
        if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
            score -= 10.0
            issues.append("Using SELECT * — consider selecting specific columns")

        # Penalize for missing ERP42test prefix
        table_refs = re.findall(r'FROM\s+(\w+)|JOIN\s+(\w+)', sql, re.IGNORECASE)
        for ref in table_refs:
            table = (ref[0] or ref[1]).lower()
            if table and not table.startswith('erp42'):
                # Check if it's likely a CTE alias
                cte_names = re.findall(r'WITH\s+(\w+)\s+AS', sql, re.IGNORECASE)
                if table.lower() not in [c.lower() for c in cte_names]:
                    score -= 5.0

        return max(0.0, score), issues

    def _score_result(
        self, result: SQLResult, intent: Optional[QueryIntent]
    ) -> tuple:
        """Score result quality. Returns (score, warnings)."""
        score = 100.0
        warnings = []

        # Failed execution
        if result.status != QueryStatus.SUCCESS:
            return 0.0, ["SQL execution failed"]

        # Empty results
        if result.row_count == 0:
            score -= 30.0
            warnings.append("Query returned no results")

        # Suspicious single-row result for listing queries
        if (intent and intent.intent_type == "listing"
                and result.row_count == 1):
            score -= 10.0
            warnings.append("Expected multiple rows for listing query")

        # Very large result set
        if result.row_count > 1000:
            score -= 5.0
            warnings.append(
                f"Large result set: {result.row_count} rows"
            )

        # Check for negative values in amount columns
        if result.rows and result.columns:
            amount_cols = [
                i for i, c in enumerate(result.columns)
                if any(kw in str(c).lower()
                       for kw in ["summ", "betrag", "umsatz", "amount"])
            ]
            for row in result.rows[:10]:  # Check first 10 rows
                for col_idx in amount_cols:
                    try:
                        val = float(row[col_idx] or 0)
                        if val < -1_000_000:
                            score -= 15.0
                            warnings.append(
                                "Suspicious negative values detected"
                            )
                            break
                    except (ValueError, TypeError, IndexError):
                        pass

        return max(0.0, score), warnings

    def _score_joins(self, sql: str, table_names: list) -> float:
        """Score confidence in JOIN relationships used."""
        # Count JOINs in SQL
        join_count = len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))

        if join_count == 0:
            return 100.0  # No joins — full confidence

        # Check if joins use verified columns
        # Simple heuristic: ERP42-style joins on Kundennumm, Lieferante, etc.
        erp_join_patterns = [
            r'ON\s+\w+\.Kundennumm\s*=\s*\w+\.Kundennumm',
            r'ON\s+\w+\.Lieferante\s*=\s*\w+\.Lieferante',
            r'ON\s+\w+\.Mitarbeite\s*=\s*\w+\.Mitarbeite',
            r'ON\s+\w+\.Projekt\s*=\s*\w+\.Projekt',
        ]
        verified = sum(
            1 for p in erp_join_patterns
            if re.search(p, sql, re.IGNORECASE)
        )

        # Score based on ratio of verified joins
        if join_count > 0:
            ratio = min(1.0, verified / join_count)
            return 60.0 + (ratio * 40.0)

        return 80.0


# ─── Singleton ────────────────────────────────────────────────────────────────
_trust_scorer: Optional[TrustScorer] = None


def get_trust_scorer() -> TrustScorer:
    global _trust_scorer
    if _trust_scorer is None:
        _trust_scorer = TrustScorer()
    return _trust_scorer
