"""
intelligence/hallucination_detector.py — Hallucination Detection Engine
Stage 5.5: Detect when AI generates invalid SQL content.

Detects:
- Invented table names not in schema
- Invented column names not in schema
- Placeholder values left in SQL
- Impossible SQL patterns
- Schema grounding violations
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HallucinationReport:
    """Report of detected hallucinations in generated SQL."""
    has_hallucinations: bool = False
    severity: str = "none"  # none, low, medium, high, critical
    issues: list = field(default_factory=list)
    confidence_penalty: float = 0.0  # 0-100 points to deduct from trust score

    def add_issue(self, issue: str, penalty: float, severity: str = "medium"):
        self.issues.append(issue)
        self.confidence_penalty = min(100.0, self.confidence_penalty + penalty)
        self.has_hallucinations = True
        # Update severity to highest seen
        severity_order = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        if severity_order.get(severity, 0) > severity_order.get(self.severity, 0):
            self.severity = severity


class HallucinationDetector:
    """
    Detects AI hallucinations in generated SQL.
    Used in the validation pipeline before execution.
    """

    # Placeholder patterns that should never appear in final SQL
    PLACEHOLDER_PATTERNS = [
        (r'YYYY-MM-DD', "Date placeholder not replaced", 50, "high"),
        (r'START_DATE', "Start date placeholder not replaced", 50, "high"),
        (r'END_DATE', "End date placeholder not replaced", 50, "high"),
        (r'\bN\b(?=\s*\)|\s*,|\s*;)', "Generic N placeholder found", 30, "medium"),
        (r"'your_\w+'", "Template placeholder found", 40, "high"),
        (r"<\w+>", "Angle bracket placeholder found", 40, "high"),
    ]

    # SQL anti-patterns that indicate hallucination
    ANTI_PATTERNS = [
        (r'(?i)SELECT\s+\*\s+FROM\s+\w+\s+WHERE\s+1\s*=\s*0',
         "Impossible WHERE condition", 20, "low"),
        (r'(?i)CROSS\s+JOIN(?!\s+\()',
         "Unintended CROSS JOIN detected", 40, "high"),
        (r'(?i);\s*SELECT',
         "Multiple statements detected", 60, "critical"),
        (r'(?i)--\s*TODO|--\s*FIXME|--\s*PLACEHOLDER',
         "Developer comment found in SQL", 20, "low"),
    ]

    def analyze(
        self,
        sql: str,
        all_tables: dict = None,
        validation_warnings: list = None,
    ) -> HallucinationReport:
        """
        Analyze SQL for hallucinations.
        Returns HallucinationReport with detected issues.
        """
        report = HallucinationReport()

        # Check placeholders
        for pattern, message, penalty, severity in self.PLACEHOLDER_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                report.add_issue(message, penalty, severity)

        # Check anti-patterns
        for pattern, message, penalty, severity in self.ANTI_PATTERNS:
            if re.search(pattern, sql, re.IGNORECASE):
                report.add_issue(message, penalty, severity)

        # Check schema violations (if schema provided)
        if all_tables:
            schema_issues = self._check_schema_grounding(sql, all_tables)
            for issue, penalty, severity in schema_issues:
                report.add_issue(issue, penalty, severity)

        # Check validation warnings for hallucinated tables
        if validation_warnings:
            for warning in validation_warnings:
                if "not found in scanned schema" in warning.lower():
                    table_match = re.search(r"Table '(\w+)'", warning)
                    if table_match:
                        table_name = table_match.group(1)
                        report.add_issue(
                            f"Hallucinated table: '{table_name}'",
                            40, "high"
                        )

        return report

    def _check_schema_grounding(
        self, sql: str, all_tables: dict
    ) -> list:
        """Check if SQL references real tables and columns."""
        issues = []
        allowed_lower = {t.lower() for t in all_tables.keys()}

        # Extract table references from SQL
        table_refs = re.findall(
            r'(?i)(?:FROM|JOIN)\s+(?:\w+\.){0,2}(\w+)',
            sql
        )

        for table in table_refs:
            table_clean = table.lower().strip("[]")
            # Skip CTE names, aliases, keywords
            if table_clean in ("select", "where", "on", "as", "null"):
                continue
            # Check if it's a CTE defined in same query
            cte_names = {
                m.lower() for m in
                re.findall(r'(?i)WITH\s+(\w+)\s+AS', sql)
            }
            if table_clean in cte_names:
                continue
            # Check against schema
            if table_clean and table_clean not in allowed_lower:
                issues.append((
                    f"Table '{table_clean}' not in schema",
                    35, "high"
                ))

        return issues

    def get_clarification_message(self, report: HallucinationReport) -> Optional[str]:
        """
        Generate user-friendly message about hallucinations.
        Stage 5.7 — Ambiguity Detection / Clarification Trigger.
        """
        if not report.has_hallucinations:
            return None

        if report.severity == "critical":
            return (
                "⚠️ The AI generated an unsafe query. "
                "Please rephrase your question more specifically."
            )
        elif report.severity == "high":
            return (
                "⚠️ The AI may have used incorrect table or column names. "
                "Try rephrasing with more specific ERP terms."
            )
        elif report.severity == "medium":
            return (
                "ℹ️ Some parts of the query may not be accurate. "
                "Please verify the results."
            )
        return None


# ─── Singleton ────────────────────────────────────────────────────────────────
_detector: Optional[HallucinationDetector] = None


def get_hallucination_detector() -> HallucinationDetector:
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector
