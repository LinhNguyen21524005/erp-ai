"""
services/query_service.py — Main Query Pipeline Orchestrator
Single entry point for all user queries.

Pipeline (Stage 4 complete):
1. Parse intent (Stage 3.8 — Intent Classification Engine)
2. Check cache (Stage 4.7 — Query Cache System)
3. Semantic search + intent tables (Stage 3.7 — Hybrid Semantic Retrieval)
4. Build schema text (Stage 2.9 — Virtual Schema Builder)
5. Get Golden SQL examples (Stage 4.1 — SQL Generation Foundation)
6. Build prompt with intent hints (Stage 3.9 — Intent-to-Schema Mapping)
7. Call LLM (Stage 4.1 — SQL Generation Foundation)
8. Guardrails (Stage 4.3 — SQL Guardrail Engine)
9. RBAC check (Stage 7.2 — RBAC)
10. Hallucination detection (Stage 5.5 — Hallucination Detection Engine)
11. Execute SQL (Stage 4.5 — SQL Execution Engine)
12. Evaluate + Trust score (Stage 5.6 — Confidence Scoring System)
13. Audit log (Stage 7.4 — Audit Logging System)
14. Cache result (Stage 4.7 — Query Cache System)
15. Format response
"""

import logging
import time
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.runtime import get_runtime
from erp42_analytics.core.models import (
    QueryResponse, SQLResult, TrustScore, AttemptLog,
    AuditEntry, QueryStatus
)
from erp42_analytics.repositories.golden_sql_repo import GoldenSQLRepository
from erp42_analytics.repositories.history_repo import HistoryRepository
from erp42_analytics.services.schema_service import SchemaService
from erp42_analytics.services.semantic_service import SemanticService
from erp42_analytics.services.sql_service import (
    PromptService, SQLGuardrailService, SQLExecutorService, EvaluatorService
)
from erp42_analytics.intelligence.intent_engine import get_intent_engine
from erp42_analytics.intelligence.join_graph import get_join_graph
from erp42_analytics.intelligence.trust_scorer import get_trust_scorer
from erp42_analytics.intelligence.query_cache import get_query_cache
from erp42_analytics.intelligence.hallucination_detector import get_hallucination_detector
from erp42_analytics.intelligence.security import get_rbac, get_audit_logger
from erp42_analytics.intelligence.prompt_guard import get_prompt_guard
from erp42_analytics.intelligence.performance_monitor import (
    get_performance_monitor, LatencyTimer
)
from erp42_analytics.intelligence.query_retry import get_query_retry_engine

logger = logging.getLogger(__name__)


class QueryService:
    """
    Main pipeline orchestrator.
    Single entry point for all user queries.
    """

    def __init__(self, all_tables: dict = None):
        # Core services
        self.schema_svc = SchemaService()
        self.semantic_svc = SemanticService()
        self.prompt_svc = PromptService()
        self.guardrail_svc = SQLGuardrailService()
        self.executor_svc = SQLExecutorService()
        self.evaluator_svc = EvaluatorService()

        # Intelligence layer
        self.intent_engine = get_intent_engine()
        self.join_graph = get_join_graph()
        self.trust_scorer = get_trust_scorer()
        self.cache = get_query_cache()
        self.hallucination_detector = get_hallucination_detector()
        self.rbac = get_rbac()
        self.audit_logger = get_audit_logger()
        self.prompt_guard = get_prompt_guard()
        self.perf_monitor = get_performance_monitor()
        self.query_retry = get_query_retry_engine()

        # Repositories
        self.golden_repo = GoldenSQLRepository()
        self.history_repo = HistoryRepository()

        # Schema (loaded once at startup)
        self._all_tables = all_tables or self.schema_svc.get_all_tables()

    def ask(
        self,
        question: str,
        user_id: str = "anonymous",
        session_id: str = "",
        role: str = "analyst",
        use_cache: bool = True,
    ) -> QueryResponse:
        """
        Main query pipeline — complete Stage 4+5+7 implementation.
        """
        start_time = time.time()
        question = question.strip()

        # ── Step 0: Prompt Guard & Input Normalization (Stage 3.2) ─────────────
        guard_result = self.prompt_guard.validate(question)
        if not guard_result.allowed:
            logger.warning(
                f"Prompt guard blocked question from {user_id}: "
                f"{guard_result.reason}"
            )
            self.audit_logger.log_query(
                user_id=user_id,
                session_id=session_id,
                question=question,
                sql="",
                tables=[],
                status=QueryStatus.FAILED,
                trust_score=0.0,
                exec_time=0.0,
                row_count=0,
                error=guard_result.reason,
            )
            blocked_result = SQLResult(
                question=question,
                sql="",
                status=QueryStatus.FAILED,
                error=guard_result.reason,
            )
            return QueryResponse(
                question=question,
                sql="",
                result=blocked_result,
                trust_score=TrustScore(overall=0.0),
                explanation=f"Question rejected: {guard_result.reason}",
                warnings=[guard_result.reason],
                total_time=round(time.time() - start_time, 2),
            )

        # Use normalized question for the rest of the pipeline
        question = guard_result.normalized_question
        timer = LatencyTimer()

        # ── Step 1: Parse intent (Stage 3.8) ─────────────────────────────────
        intent = self.intent_engine.parse(question)

        # ── Step 2: Check cache (Stage 4.7) ──────────────────────────────────
        if use_cache:
            cached = self.cache.get(question)
            if cached:
                sql_result = SQLResult(
                    question=question,
                    sql=cached.get("sql", ""),
                    status=QueryStatus.SUCCESS,
                    rows=cached.get("rows", []),
                    columns=cached.get("columns", []),
                    row_count=cached.get("row_count", 0),
                )
                trust = TrustScore(overall=cached.get("trust_score", 80.0))
                return QueryResponse(
                    question=question,
                    sql=cached.get("sql", ""),
                    result=sql_result,
                    trust_score=trust,
                    explanation="[Cached result — same question answered before]",
                    total_time=round(time.time() - start_time, 2),
                )

        # ── Step 3: RBAC — filter tables by user role (Stage 7.2) ─────────────
        allowed_tables = self.rbac.filter_tables(self._all_tables, role)

        # ── Step 4: Semantic search → relevant tables (Stage 3.7) ─────────────
        timer.start("retrieval")
        table_names = self.semantic_svc.find_relevant_tables(
            question,
            n_results=config.query.n_relevant_tables,
        )
        timer.stop("retrieval")

        # Merge intent-suggested tables
        for table in intent.dimensions:
            if table not in table_names and table in allowed_tables:
                table_names.append(table)
                logger.debug(f"Intent added table: {table}")

        filtered_tables = {
            name: allowed_tables[name]
            for name in table_names
            if name in allowed_tables
        }
        if not filtered_tables:
            filtered_tables = dict(list(allowed_tables.items())[:3])

        # ── Step 5: Build schema text (Stage 2.9) ──────────────────────────────
        schema_text = self.semantic_svc.build_schema_text(filtered_tables)

        # ── Step 6: Get Golden SQL examples (Stage 4.1) ────────────────────────
        golden_examples = self.golden_repo.find_similar(
            question,
            n_results=config.query.golden_sql_max_examples,
            min_score=config.query.golden_sql_min_score,
        )

        # ── Step 7: Get instructions + time flag (Stage 3.9) ───────────────────
        instructions = self.history_repo.get_instructions()
        needs_time = self.semantic_svc.needs_current_time(question)

        # ── Step 8: Build prompt with intent hints (Stage 4.2) ─────────────────
        intent_hints = self.intent_engine.get_prompt_hints(intent)
        prompt = self.prompt_svc.build_prompt(
            question=question,
            schema_text=schema_text,
            golden_examples=golden_examples if golden_examples else None,
            instructions=instructions if instructions else None,
            current_time=needs_time,
            intent_hints=intent_hints if intent_hints else None,
        )

        # ── Steps 9-13: Generate → Validate → Execute, with retry (Stage 4.8) ──
        timer.start("sql_generation")
        attempts_log = []  # list[AttemptLog] — full history (Nhóm 1+2)
        clean_sql = ""
        sql_result = None
        validation = {}
        hallucination_report = None
        current_prompt = prompt
        max_retries = self.query_retry.max_retries

        for attempt in range(max_retries + 1):
            attempt_start = time.time()
            attempt_num = attempt + 1

            # Step 9: Call LLM
            llm_call_start = time.time()
            try:
                llm_response = get_runtime().chat(
                    model=config.model.llm_model,
                    messages=[{"role": "user", "content": current_prompt}],
                    options={
                        "temperature": config.model.llm_temperature,
                        # ROOT CAUSE FIX: this was never passed before, so
                        # Ollama fell back to the model's own default
                        # (often 128-256 tokens) and cut SQL mid-clause on
                        # anything with a JOIN + comment. 500 was also too
                        # low for multi-table queries — raised to 1536.
                        "num_predict": config.model.llm_max_tokens,
                    },
                )
                raw_sql = llm_response["message"]["content"].strip()

                # Record TTFT / tokens-per-second / token counts for this
                # call (Step 7.4 — requested by Felix). Never let a
                # monitoring failure break the actual query pipeline.
                try:
                    get_performance_monitor().record_from_runtime_response(llm_response)
                except Exception as mon_err:  # noqa: BLE001
                    logger.debug("Performance monitor recording failed: %s", mon_err)
            except Exception as e:
                logger.error(f"LLM error: {e}")
                attempts_log.append(AttemptLog(
                    attempt_number=attempt_num,
                    sql="",
                    stage_failed_at="llm_call",
                    error_message=str(e),
                    error_category="llm_error",
                    latency_ms=round((time.time() - attempt_start) * 1000, 1),
                    sql_generation_ms=round(
                        (time.time() - llm_call_start) * 1000, 1
                    ),
                    succeeded=False,
                ))
                return self._error_response(
                    question, f"LLM error: {str(e)}", start_time,
                    attempts=attempts_log,
                )
            llm_call_ms = round((time.time() - llm_call_start) * 1000, 1)

            # Step 10: Guardrails
            validation = self.guardrail_svc.validate(
                raw_sql,
                allowed_tables=list(self._all_tables.keys())
            )
            if not validation["valid"]:
                analysis = self.query_retry.analyze_failure(
                    validation["error"]
                )
                # Run hallucination check even on rejected SQL — useful
                # signal for diagnosing WHY it was rejected (Nhóm "còn sót")
                try:
                    guard_hallu = self.hallucination_detector.analyze(
                        sql=raw_sql,
                        all_tables=self._all_tables,
                        validation_warnings=[],
                    )
                    guard_hallu_severity = guard_hallu.severity
                except Exception:
                    guard_hallu_severity = None
                attempts_log.append(AttemptLog(
                    attempt_number=attempt_num,
                    sql=raw_sql,
                    stage_failed_at="guardrail",
                    error_message=validation["error"],
                    error_category=analysis.category.value,
                    hallucination_severity=guard_hallu_severity,
                    latency_ms=round((time.time() - attempt_start) * 1000, 1),
                    sql_generation_ms=llm_call_ms,
                    succeeded=False,
                ))
                if self.query_retry.should_retry(analysis, attempt):
                    current_prompt = self.query_retry.build_retry_prompt(
                        prompt, raw_sql, analysis, attempt_num
                    )
                    logger.info(
                        f"Retrying after guardrail failure "
                        f"(attempt {attempt_num}/{max_retries})"
                    )
                    continue
                return self._error_response(
                    question, validation["error"], start_time, sql=raw_sql,
                    attempts=attempts_log,
                )
            clean_sql = validation["cleaned_sql"]

            # Step 11: RBAC permission check
            allowed, reason = self.rbac.check_sql_permissions(clean_sql, role)
            if not allowed:
                attempts_log.append(AttemptLog(
                    attempt_number=attempt_num,
                    sql=clean_sql,
                    stage_failed_at="permission",
                    error_message=reason,
                    error_category="permission_denied",
                    latency_ms=round((time.time() - attempt_start) * 1000, 1),
                    succeeded=False,
                ))
                return self._error_response(
                    question, reason, start_time, sql=clean_sql,
                    attempts=attempts_log,
                )

            # Step 12: Hallucination detection
            hallucination_report = self.hallucination_detector.analyze(
                sql=clean_sql,
                all_tables=self._all_tables,
                validation_warnings=validation.get("warnings", []),
            )
            if hallucination_report.severity == "critical":
                clarification = (
                    self.hallucination_detector.get_clarification_message(
                        hallucination_report
                    )
                )
                analysis = self.query_retry.analyze_failure(
                    clarification or "Hallucinated table/column reference"
                )
                attempts_log.append(AttemptLog(
                    attempt_number=attempt_num,
                    sql=clean_sql,
                    stage_failed_at="hallucination_check",
                    error_message=clarification or "Critical SQL error detected",
                    error_category=analysis.category.value,
                    hallucination_severity=hallucination_report.severity,
                    latency_ms=round((time.time() - attempt_start) * 1000, 1),
                    sql_generation_ms=llm_call_ms,
                    succeeded=False,
                ))
                if self.query_retry.should_retry(analysis, attempt):
                    current_prompt = self.query_retry.build_retry_prompt(
                        prompt, clean_sql, analysis, attempt_num
                    )
                    logger.info(
                        f"Retrying after hallucination check "
                        f"(attempt {attempt_num}/{max_retries})"
                    )
                    continue
                return self._error_response(
                    question, clarification or "Critical SQL error detected",
                    start_time, sql=clean_sql, attempts=attempts_log,
                )

            timer.stop("sql_generation")

            # Step 13: Execute SQL
            timer.start("db_execution")
            max_rows = self.rbac.get_max_rows(role)
            sql_result = self.executor_svc.execute(
                sql=clean_sql,
                max_rows=max_rows,
                timeout=config.query.exec_timeout_seconds,
            )
            sql_result.question = question
            timer.stop("db_execution")

            attempt_latency_ms = round(
                (time.time() - attempt_start) * 1000, 1
            )

            if sql_result.status == QueryStatus.SUCCESS:
                attempts_log.append(AttemptLog(
                    attempt_number=attempt_num,
                    sql=clean_sql,
                    stage_failed_at=None,
                    error_message=None,
                    error_category=None,
                    hallucination_severity=hallucination_report.severity,
                    latency_ms=attempt_latency_ms,
                    sql_generation_ms=llm_call_ms,
                    db_execution_ms=round(sql_result.exec_time * 1000, 1),
                    succeeded=True,
                ))
                break  # Success — exit retry loop

            # Execution failed — analyze and decide whether to retry
            analysis = self.query_retry.analyze_failure(sql_result.error)
            attempts_log.append(AttemptLog(
                attempt_number=attempt_num,
                sql=clean_sql,
                stage_failed_at="execution",
                error_message=sql_result.error,
                error_category=analysis.category.value,
                hallucination_severity=hallucination_report.severity,
                latency_ms=attempt_latency_ms,
                sql_generation_ms=llm_call_ms,
                db_execution_ms=round(sql_result.exec_time * 1000, 1),
                succeeded=False,
            ))
            if self.query_retry.should_retry(analysis, attempt):
                timer.start("sql_generation")
                current_prompt = self.query_retry.build_retry_prompt(
                    prompt, clean_sql, analysis, attempt_num
                )
                logger.info(
                    f"Retrying after execution failure "
                    f"(attempt {attempt_num}/{max_retries}): "
                    f"{analysis.category.value}"
                )
                continue
            else:
                break  # Give up — return the failed result

        retry_attempts = [
            f"Attempt {a.attempt_number}: {a.stage_failed_at} — {a.error_message}"
            for a in attempts_log if not a.succeeded
        ]
        if retry_attempts:
            logger.info(
                f"Query required {len(retry_attempts)} retry attempt(s) "
                f"for: '{question[:50]}'"
            )

        # ── Step 14: RBAC — mask sensitive columns (Stage 7.2) ─────────────────
        if sql_result.status == QueryStatus.SUCCESS:
            sql_result.columns, sql_result.rows = self.rbac.mask_result(
                sql_result.columns, sql_result.rows, role
            )

        # ── Step 15: Trust score (Stage 5.6) ────────────────────────────────────
        trust_breakdown = self.trust_scorer.compute(
            sql=clean_sql,
            result=sql_result,
            intent=intent,
            table_names=table_names,
            validation_warnings=validation.get("warnings", []),
        )
        # Apply hallucination penalty
        trust_breakdown.overall = max(
            0.0,
            trust_breakdown.overall - hallucination_report.confidence_penalty
        )
        trust_score = trust_breakdown.to_trust_score()

        # ── Step 16: Evaluate (Stage 5.4 Result Consistency) ────────────────────
        evaluation = self.evaluator_svc.evaluate(
            sql=clean_sql,
            result=sql_result,
            question=question,
        )
        logger.warning(
            f"[DEBUG tables_used] clean_sql={clean_sql!r} | "
            f"evaluation.tables_used={evaluation.get('tables_used')} | "
            f"table_names(candidates)={table_names}"
        )

        # ── Step 17: Save history ───────────────────────────────────────────────
        self.history_repo.save_query(
            question=question,
            sql=clean_sql,
            success=sql_result.status == QueryStatus.SUCCESS,
            exec_time=sql_result.exec_time,
            row_count=sql_result.row_count,
            confidence_score=trust_score.overall,
            tables_used=evaluation.get("tables_used", []),
        )

        # ── Step 18: Audit log (Stage 7.4) ──────────────────────────────────────
        if config.security.audit_enabled:
            self.audit_logger.log_query(
                user_id=user_id,
                session_id=session_id,
                question=question,
                sql=clean_sql,
                tables=evaluation.get("tables_used", []),
                status=sql_result.status,
                trust_score=trust_score.overall,
                exec_time=sql_result.exec_time,
                row_count=sql_result.row_count,
                error=sql_result.error,
            )

        # ── Step 19: Cache if successful (Stage 4.7) ───────────────────────────
        if use_cache and sql_result.status == QueryStatus.SUCCESS:
            self.cache.store(
                question=question,
                sql=clean_sql,
                rows=sql_result.rows,
                columns=sql_result.columns,
                row_count=sql_result.row_count,
                trust_score=trust_score.overall,
            )

        # Record latency breakdown for performance monitoring (Stage 7.6)
        self.perf_monitor.record_latency(timer.to_breakdown())

        # ── Step 20: Build response (Stage 6.4 Result + 6.1 Chat) ──────────────
        total_time = round(time.time() - start_time, 2)
        all_warnings = (
            validation.get("warnings", []) +
            evaluation.get("issues", []) +
            hallucination_report.issues
        )
        if retry_attempts:
            all_warnings.append(
                f"Query succeeded after {len(retry_attempts)} retry "
                f"attempt(s) due to SQL errors (Stage 4.8 self-correction)"
            )

        # Worst hallucination severity seen across ALL attempts, not just
        # the final one — surfaces cases where an earlier attempt
        # hallucinated badly even though the final SQL is clean.
        severity_order = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        max_severity = "none"
        for a in attempts_log:
            if a.hallucination_severity and severity_order.get(
                a.hallucination_severity, 0
            ) > severity_order.get(max_severity, 0):
                max_severity = a.hallucination_severity

        return QueryResponse(
            question=question,
            sql=clean_sql,
            result=sql_result,
            trust_score=trust_score,
            explanation=self._build_explanation(sql_result, evaluation, table_names),
            tables_used=evaluation.get("tables_used", []),
            golden_examples_used=golden_examples,
            warnings=all_warnings,
            total_time=total_time,
            attempts=attempts_log,
            hallucination_severity=hallucination_report.severity,
            max_hallucination_severity=max_severity,
        )

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _build_explanation(
        self, result: SQLResult, evaluation: dict, table_names: list
    ) -> str:
        if result.status != QueryStatus.SUCCESS:
            return f"SQL execution failed: {result.error}"
        tables = evaluation.get("tables_used") or table_names
        table_str = ", ".join(
            [f"ERP42test.dbo.{t}" for t in tables]
        ) if tables else "unknown tables"
        return (
            f"Queried {table_str} and retrieved "
            f"{result.row_count} rows in {result.exec_time}s. "
            f"{'Query returned results.' if result.row_count > 0 else 'No data found.'} "
            f"✓ Query executed in {result.exec_time}s."
        )

    def _error_response(
        self, question: str, error: str,
        start_time: float, sql: str = "",
        attempts: list = None,
    ) -> QueryResponse:
        result = SQLResult(
            question=question, sql=sql,
            status=QueryStatus.FAILED, error=error,
        )
        return QueryResponse(
            question=question, sql=sql,
            result=result,
            trust_score=TrustScore(overall=0.0),
            explanation=f"Error: {error}",
            total_time=round(time.time() - start_time, 2),
            attempts=attempts or [],
        )

    def reload_tables(self):
        self._all_tables = self.schema_svc.get_all_tables()
        logger.info(f"Schema reloaded: {len(self._all_tables)} tables")