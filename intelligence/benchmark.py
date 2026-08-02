"""
intelligence/benchmark.py — Benchmark & Accuracy Evaluation System
Stage 8.3: AI Accuracy Benchmarking (ground truth dataset + evaluation metrics + latency)

Responsibilities:
- Define ground truth Q&A pairs for ERP42
- Measure SQL accuracy against ground truth
- Track latency per pipeline stage
- Generate benchmark report
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Ground Truth Dataset (Stage 8.3 — Business Query Benchmark) ────────────
# Verified Q&A pairs for ERP42test database
# Used to measure system accuracy objectively

GROUND_TRUTH = [
    {
        "id": "GT001",
        "question": "Wie viele Aufträge haben wir?",
        "expected_sql_pattern": r"COUNT.*auftrag",
        "expected_tables": ["auftrag"],
        "intent_type": "aggregation",
        "difficulty": "easy",
        "domain": "sales",
    },
    {
        "id": "GT002",
        "question": "Gesamtumsatz im Jahr 2023",
        "expected_sql_pattern": r"SUM.*Bruttosumm.*2023|YEAR.*Datum.*2023",
        "expected_tables": ["auftrag"],
        "intent_type": "aggregation",
        "difficulty": "medium",
        "domain": "finance",
    },
    {
        "id": "GT003",
        "question": "Die 10 umsatzstärksten Aufträge mit Kundeninformationen",
        "expected_sql_pattern": r"TOP\s+10.*JOIN.*kunde|JOIN.*kunde.*TOP\s+10",
        "expected_tables": ["auftrag", "kunde"],
        "intent_type": "ranking",
        "difficulty": "hard",
        "domain": "sales",
    },
    {
        "id": "GT004",
        "question": "Wie viele Aufträge hat jeder Kunde?",
        "expected_sql_pattern": r"COUNT.*GROUP BY.*Kundennumm",
        "expected_tables": ["auftrag"],
        "intent_type": "aggregation",
        "difficulty": "medium",
        "domain": "sales",
    },
    {
        "id": "GT005",
        "question": "Wer sind die Top 5 Kunden nach Umsatz?",
        "expected_sql_pattern": r"TOP\s+5.*SUM.*ORDER BY",
        "expected_tables": ["auftrag", "kunde"],
        "intent_type": "ranking",
        "difficulty": "hard",
        "domain": "sales",
    },
    {
        "id": "GT006",
        "question": "Zeige alle Lieferanten",
        "expected_sql_pattern": r"FROM.*lieferan",
        "expected_tables": ["lieferan"],
        "intent_type": "listing",
        "difficulty": "easy",
        "domain": "procurement",
    },
    {
        "id": "GT007",
        "question": "Wie viele Kunden haben wir?",
        "expected_sql_pattern": r"COUNT.*kunde",
        "expected_tables": ["kunde"],
        "intent_type": "aggregation",
        "difficulty": "easy",
        "domain": "crm",
    },
    {
        "id": "GT008",
        "question": "What is the average order value?",
        "expected_sql_pattern": r"AVG.*Bruttosumm|SUM.*COUNT",
        "expected_tables": ["auftrag"],
        "intent_type": "aggregation",
        "difficulty": "medium",
        "domain": "sales",
    },
    {
        "id": "GT009",
        "question": "Zeige alle Projekte",
        "expected_sql_pattern": r"FROM.*projekt",
        "expected_tables": ["projekt"],
        "intent_type": "listing",
        "difficulty": "easy",
        "domain": "pm",
    },
    {
        "id": "GT010",
        "question": "Gesamtumsatz im Jahr 2024",
        "expected_sql_pattern": r"SUM.*Bruttosumm.*2024|YEAR.*Datum.*2024",
        "expected_tables": ["auftrag"],
        "intent_type": "aggregation",
        "difficulty": "medium",
        "domain": "finance",
    },
]


@dataclass
class BenchmarkResult:
    """Result for a single benchmark test case."""
    test_id: str
    question: str
    generated_sql: str
    expected_tables: list
    actual_tables: list
    sql_executed: bool
    row_count: int
    exec_time: float
    total_time: float
    trust_score: float

    # Accuracy scores
    table_accuracy: float = 0.0      # Correct tables used?
    sql_validity: float = 0.0        # SQL ran without error?
    pattern_match: float = 0.0       # SQL matches expected pattern?
    overall_score: float = 0.0       # Weighted average

    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0

    avg_table_accuracy: float = 0.0
    avg_sql_validity: float = 0.0
    avg_trust_score: float = 0.0
    avg_total_time: float = 0.0
    avg_exec_time: float = 0.0

    results: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    # By difficulty
    easy_accuracy: float = 0.0
    medium_accuracy: float = 0.0
    hard_accuracy: float = 0.0

    # By domain
    domain_accuracy: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"BENCHMARK REPORT — {self.timestamp.strftime('%Y-%m-%d %H:%M')}",
            f"{'='*60}",
            f"Total Tests:      {self.total_tests}",
            f"Passed (≥70%):    {self.passed}",
            f"Failed (<70%):    {self.failed}",
            f"Errors:           {self.errors}",
            f"",
            f"Table Accuracy:   {self.avg_table_accuracy:.1f}%",
            f"SQL Validity:     {self.avg_sql_validity:.1f}%",
            f"Avg Trust Score:  {self.avg_trust_score:.1f}%",
            f"",
            f"Avg Total Time:   {self.avg_total_time:.1f}s",
            f"Avg Exec Time:    {self.avg_exec_time:.3f}s",
            f"",
            f"By Difficulty:",
            f"  Easy:   {self.easy_accuracy:.1f}%",
            f"  Medium: {self.medium_accuracy:.1f}%",
            f"  Hard:   {self.hard_accuracy:.1f}%",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class BenchmarkSystem:
    """
    Runs benchmark tests against the query pipeline.
    Stage 8.3 — AI Accuracy Benchmarking.
    """

    def __init__(self, query_service):
        self.query_service = query_service

    def run_all(self, test_ids: list = None) -> BenchmarkReport:
        """
        Run all or selected benchmark tests.
        Returns complete BenchmarkReport.
        """
        tests = GROUND_TRUTH
        if test_ids:
            tests = [t for t in GROUND_TRUTH if t["id"] in test_ids]

        report = BenchmarkReport(total_tests=len(tests))
        logger.info(f"Starting benchmark: {len(tests)} tests")

        for i, test in enumerate(tests, 1):
            logger.info(
                f"[{i}/{len(tests)}] Running: {test['question'][:50]}"
            )
            result = self._run_single(test)
            report.results.append(result)

            status = "✅" if result.overall_score >= 70 else "❌"
            logger.info(
                f"  {status} Score: {result.overall_score:.0f}% | "
                f"Time: {result.total_time:.1f}s"
            )

        self._compute_report_stats(report, tests)
        return report

    def _run_single(self, test: dict) -> BenchmarkResult:
        """Run a single benchmark test."""
        import re

        start = time.time()
        result = BenchmarkResult(
            test_id=test["id"],
            question=test["question"],
            generated_sql="",
            expected_tables=test["expected_tables"],
            actual_tables=[],
            sql_executed=False,
            row_count=0,
            exec_time=0.0,
            total_time=0.0,
            trust_score=0.0,
        )

        try:
            response = self.query_service.ask(
                question=test["question"],
                use_cache=False,  # Always fresh for benchmark
            )

            result.generated_sql = response.sql
            result.actual_tables = response.tables_used or []
            result.sql_executed = (
                response.result.status.value == "success"
            )
            result.row_count = response.result.row_count
            result.exec_time = response.result.exec_time
            result.trust_score = response.trust_score.overall
            result.total_time = response.total_time
            result.error = response.result.error

            # Score: Table accuracy
            expected = set(t.lower() for t in test["expected_tables"])
            actual = set(t.lower() for t in result.actual_tables)
            if expected:
                overlap = len(expected & actual)
                result.table_accuracy = (overlap / len(expected)) * 100
            else:
                result.table_accuracy = 100.0

            # Score: SQL validity
            result.sql_validity = 100.0 if result.sql_executed else 0.0

            # Score: Pattern match
            pattern = test.get("expected_sql_pattern", "")
            if pattern and result.generated_sql:
                match = re.search(
                    pattern, result.generated_sql,
                    re.IGNORECASE | re.DOTALL
                )
                result.pattern_match = 100.0 if match else 0.0
            else:
                result.pattern_match = 50.0  # No pattern = neutral

            # Overall: weighted average
            result.overall_score = (
                result.table_accuracy * 0.35 +
                result.sql_validity * 0.40 +
                result.pattern_match * 0.25
            )

        except Exception as e:
            result.error = str(e)
            result.total_time = round(time.time() - start, 2)
            logger.error(f"Benchmark error for {test['id']}: {e}")

        return result

    def _compute_report_stats(
        self, report: BenchmarkReport, tests: list
    ):
        """Compute aggregate statistics for report."""
        if not report.results:
            return

        results = report.results
        n = len(results)

        report.avg_table_accuracy = sum(
            r.table_accuracy for r in results
        ) / n
        report.avg_sql_validity = sum(
            r.sql_validity for r in results
        ) / n
        report.avg_trust_score = sum(
            r.trust_score for r in results
        ) / n
        report.avg_total_time = sum(
            r.total_time for r in results
        ) / n
        report.avg_exec_time = sum(
            r.exec_time for r in results
        ) / n

        # Count pass/fail
        for r in results:
            if r.error and not r.sql_executed:
                report.errors += 1
            elif r.overall_score >= 70:
                report.passed += 1
            else:
                report.failed += 1

        # By difficulty
        test_map = {t["id"]: t for t in tests}
        for difficulty in ["easy", "medium", "hard"]:
            difficulty_results = [
                r for r in results
                if test_map.get(r.test_id, {}).get("difficulty") == difficulty
            ]
            if difficulty_results:
                avg = sum(
                    r.overall_score for r in difficulty_results
                ) / len(difficulty_results)
                setattr(report, f"{difficulty}_accuracy", avg)

        # By domain
        domains = set(t.get("domain", "other") for t in tests)
        for domain in domains:
            domain_results = [
                r for r in results
                if test_map.get(r.test_id, {}).get("domain") == domain
            ]
            if domain_results:
                avg = sum(
                    r.overall_score for r in domain_results
                ) / len(domain_results)
                report.domain_accuracy[domain] = round(avg, 1)
