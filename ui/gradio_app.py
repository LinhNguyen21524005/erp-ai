"""
ui/gradio_app.py — Gradio Chat Interface
Migrates: appf_v2.py
Uses new architecture: QueryService as single entry point.

Features:
- Chat interface (German/English)
- Admin panel (Golden SQL, Instructions, Schema management)
- Result formatting with trust score
- Query history
"""

import logging
from typing import Optional

import gradio as gr

from erp42_analytics.core.config import config
from erp42_analytics.core.models import QueryResponse, QueryStatus, ConfidenceLevel
from erp42_analytics.services.query_service import QueryService
from erp42_analytics.services.schema_service import SchemaService
from erp42_analytics.services.semantic_service import SemanticService
from erp42_analytics.repositories.golden_sql_repo import GoldenSQLRepository
from erp42_analytics.repositories.history_repo import HistoryRepository
from erp42_analytics.intelligence.benchmark import BenchmarkSystem, GROUND_TRUTH
from erp42_analytics.intelligence.pilot import get_feedback_collector, get_pilot_manager, FeedbackEntry
from erp42_analytics.intelligence.performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)


class ResultFormatter:
    """Formats QueryResponse into readable markdown for Gradio."""

    CONFIDENCE_EMOJI = {
        ConfidenceLevel.HIGH: "🟢",
        ConfidenceLevel.MEDIUM: "🟡",
        ConfidenceLevel.LOW: "🔴",
    }

    def format(self, response: QueryResponse) -> str:
        """Format complete QueryResponse to markdown string."""
        parts = []

        # SQL block
        parts.append(f"🔍 **Generated SQL:**\n```sql\n{response.sql}\n```")

        # Trust score
        emoji = self.CONFIDENCE_EMOJI.get(
            response.trust_score.level, "⚪"
        )
        parts.append(
            f"{emoji} **Confidence:** {response.trust_score.label} "
            f"({response.trust_score.overall:.0f}%)"
        )

        # Warnings
        if response.warnings:
            for w in response.warnings:
                parts.append(f"⚠️ **Warning:** {w}")

        # Results
        result = response.result
        if result.status == QueryStatus.SUCCESS:
            if result.row_count > 0:
                parts.append(
                    f"📊 **Result:** {result.row_count} rows "
                    f"(in {result.exec_time}s)"
                )
                parts.append(self._format_table(result.columns, result.rows))
            else:
                parts.append(
                    "📊 **Result:** No data found matching your query.\n"
                    "Try rephrasing your question or adjusting the filters."
                )
        else:
            parts.append(f"❌ **Error:** {result.error}")

        # Explanation
        parts.append(f"💡 **Explanation:** {response.explanation}")

        # Golden SQL note
        if response.golden_examples_used:
            n = len(response.golden_examples_used)
            parts.append(f"ℹ️ Based on {n} similar verified quer{'y' if n==1 else 'ies'}")

        # Response time
        parts.append(f"_⏱ Total response time: {response.total_time}s_")

        return "\n\n".join(parts)

    def format_error(self, error: str) -> str:
        return (
            f"🔴 **Error**\n\n"
            f"❌ {error}\n\n"
            f"_Please rephrase your question or check the Admin panel._"
        )

    def _format_table(self, columns: list, rows: list) -> str:
        """Format SQL results as markdown table."""
        if not columns or not rows:
            return ""

        # Header
        header = " | ".join(str(c) for c in columns)
        separator = " | ".join(
            "-" * max(len(str(c)), 6) for c in columns
        )
        lines = [f"```\n{header}", separator]

        # Rows (max 20 for display)
        display_rows = rows[:20]
        for row in display_rows:
            line = " | ".join(
                str(v)[:30] if v is not None else "None"
                for v in row
            )
            lines.append(line)

        if len(rows) > 20:
            lines.append(f"... ({len(rows) - 20} more rows)")

        lines.append("```")
        return "\n".join(lines)

    def format_stats(self, stats: dict) -> str:
        """Format system statistics."""
        return (
            f"📊 **System Stats**\n"
            f"- Total queries: {stats.get('total_queries', 0)}\n"
            f"- Success rate: {stats.get('success_rate', 0)}%\n"
            f"- Tables scanned: {stats.get('tables_scanned', 0)}\n"
            f"- Golden SQL pairs: {stats.get('golden_sql_count', 0)}\n"
            f"- Model: {config.model.llm_model}"
        )


class ERP42GradioApp:
    """
    Main Gradio application.
    Wires together QueryService + ResultFormatter + Gradio UI.
    """

    def __init__(self):
        self.formatter = ResultFormatter()
        self._query_svc: Optional[QueryService] = None
        self._schema_svc = SchemaService()
        self._semantic_svc = SemanticService()
        self._golden_repo = GoldenSQLRepository()
        self._history_repo = HistoryRepository()
        self._feedback = get_feedback_collector()
        self._pilot = get_pilot_manager()
        self._perf_monitor = get_performance_monitor()

    def _get_query_service(self) -> QueryService:
        """Lazy-init QueryService."""
        if self._query_svc is None:
            self._query_svc = QueryService()
        return self._query_svc

    # ─── Chat Handler ─────────────────────────────────────────────────────────

    def ask(self, question: str, history: list) -> str:
        """Main chat handler — called by Gradio ChatInterface."""
        if not question.strip():
            return "Please enter a question."

        try:
            svc = self._get_query_service()
            response = svc.ask(
                question=question,
                user_id="gradio_user",
                session_id="gradio_session",
            )
            return self.formatter.format(response)

        except Exception as e:
            logger.error(f"Ask error: {e}", exc_info=True)
            return self.formatter.format_error(str(e))

    # ─── Admin Handlers ───────────────────────────────────────────────────────

    def add_golden_sql(self, question: str, sql: str) -> str:
        if not question.strip() or not sql.strip():
            return "❌ Both question and SQL are required."
        success = self._golden_repo.add(question.strip(), sql.strip())
        if success:
            return f"✅ Added Golden SQL: '{question[:60]}'"
        return "❌ Failed to add Golden SQL."

    def add_instruction(self, instruction: str) -> str:
        if not instruction.strip():
            return "❌ Instruction cannot be empty."
        self._history_repo.save_instruction(instruction.strip())
        return f"✅ Instruction added: '{instruction[:60]}'"

    def rescan_schema(self) -> str:
        results = self._schema_svc.scan_all_tables()
        all_tables = self._schema_svc.get_all_tables()

        # Reload query service with new schema
        self._query_svc = QueryService(all_tables=all_tables)

        # Reindex embeddings
        self._semantic_svc.index_tables(all_tables, force_reindex=True)

        return (
            f"✅ Schema rescanned: {results['scanned']} tables\n"
            f"❌ Failed: {results['failed']} tables\n"
            f"✅ Embeddings reindexed: {self._semantic_svc.count()} tables\n"
            f"📊 Total in storage: {len(all_tables)} tables"
        )

    def get_stats(self) -> str:
        stats = self._history_repo.get_stats()
        return self.formatter.format_stats(stats)

    def get_history(self) -> str:
        history = self._history_repo.get_recent(limit=10)
        if not history:
            return "No query history yet."
        lines = ["**Recent Queries:**"]
        for h in history:
            status = "✅" if h["success"] else "❌"
            exec_time = h.get("exec_time", 0) or 0
            question = h.get("question", "")[:60]
            lines.append(f"{status} [{exec_time:.1f}s] {question}")
        return "\n".join(lines)

    # ─── Benchmark Handler ───────────────────────────────────────────────────

    def run_benchmark(self) -> str:
        """Run benchmark tests and return report."""
        try:
            svc = self._get_query_service()
            benchmark = BenchmarkSystem(svc)
            report = benchmark.run_all()
            lines = [report.summary()]
            lines.append("\n**Detailed Results:**")
            for r in report.results:
                status = "✅" if r.overall_score >= 70 else "❌"
                lines.append(
                    f"{status} [{r.test_id}] {r.question[:40]} | "
                    f"Score: {r.overall_score:.0f}% | "
                    f"Time: {r.total_time:.1f}s"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Benchmark error: {e}"

    def submit_feedback(
        self, question: str, sql: str,
        rating: int, is_correct: bool,
        correction: str = "", comment: str = ""
    ) -> str:
        """Submit user feedback on a query result."""
        try:
            entry = FeedbackEntry(
                question=question.strip(),
                sql=sql.strip(),
                rating=rating,
                is_correct=is_correct,
                correction=correction.strip() if correction else None,
                comment=comment.strip() if comment else None,
            )
            success = self._feedback.submit(entry)
            if success:
                msg = f"✅ Feedback submitted (rating: {rating}⭐)"
                if not is_correct and correction:
                    msg += "\n✅ Corrected SQL auto-added to Golden SQL Store"
                return msg
            return "❌ Failed to submit feedback"
        except Exception as e:
            return f"❌ Error: {e}"

    def get_pilot_kpis(self) -> str:
        """Get pilot KPI summary."""
        try:
            kpis = self._pilot.get_kpis()
            return kpis.summary()
        except Exception as e:
            return f"❌ Error: {e}"

    def get_performance_snapshot(self) -> str:
        """Get performance monitor snapshot — Stage 7.6."""
        try:
            snapshot = self._perf_monitor.get_full_snapshot()
            res = snapshot["resources"]
            lat = snapshot["latency"]

            lines = [
                "**System Resources**",
                f"CPU: {res['cpu_percent']}%",
                f"RAM: {res['ram_percent']}% "
                f"({res['ram_used_gb']:.1f}GB / {res['ram_total_gb']:.1f}GB)",
                f"Disk: {res['disk_percent']}%",
                "",
                "**Latency Breakdown** (avg over last "
                f"{lat['sample_count']} queries)",
                f"Retrieval:      {lat.get('avg_retrieval_ms', 0):.0f}ms",
                f"SQL Generation: {lat.get('avg_sql_generation_ms', 0):.0f}ms",
                f"DB Execution:   {lat.get('avg_db_execution_ms', 0):.0f}ms",
                f"Validation:     {lat.get('avg_validation_ms', 0):.0f}ms",
                f"Total:          {lat.get('avg_total_ms', 0):.0f}ms",
                "",
                f"Bottleneck: {lat.get('bottleneck', 'n/a')}",
                f"Meets <5s target: "
                f"{'✅ Yes' if lat.get('meets_target') else '❌ No'}",
            ]
            if not snapshot["psutil_available"]:
                lines.append(
                    "\n⚠️ psutil not installed — CPU/RAM tracking disabled. "
                    "Run: pip install psutil"
                )
            return "\n".join(lines)
        except Exception as e:
            return f"❌ Error: {e}"

    # ─── Startup ──────────────────────────────────────────────────────────────

    def startup(self):
        """Initialize system on app start."""
        print(f"🚀 Starting {config.app.app_name} v{config.app.app_version}")

        # Load or scan schema
        if self._schema_svc.is_schema_stale():
            print("📡 Scanning database schema...")
            results = self._schema_svc.scan_all_tables()
            print(f"✅ Scanned: {results['scanned']} tables")
        else:
            count = self._schema_svc.get_table_count()
            print(f"✅ Loaded schema: {count} tables")

        all_tables = self._schema_svc.get_all_tables()
        print(f"📊 Ready with {len(all_tables)} tables")

        # Index embeddings if needed
        if not self._semantic_svc.is_indexed():
            print("🔍 Indexing schema embeddings (one-time setup)...")
            self._semantic_svc.index_tables(all_tables)
            print(f"✅ Indexed: {self._semantic_svc.count()} tables")
        else:
            print(f"✅ Embeddings ready: {self._semantic_svc.count()} tables")

        # Init query service
        self._query_svc = QueryService(all_tables=all_tables)
        print("✅ Query service ready")

    # ─── Build UI ─────────────────────────────────────────────────────────────

    def build_ui(self) -> gr.Blocks:
        """Build and return Gradio app."""

        with gr.Blocks(
            title=config.app.app_name,
            theme=gr.themes.Soft(),
        ) as demo:

            gr.Markdown(f"# 🗄️ {config.app.app_name}")
            gr.Markdown(
                "Ask questions about your ERP database in **English or German**. "
                f"Running 100% locally on `{config.model.llm_model}` — no cloud, no data leaks."
            )

            with gr.Tabs():

                # ── Chat Tab ──────────────────────────────────────────────────
                with gr.Tab("💬 Chat"):
                    gr.ChatInterface(
                        fn=self.ask,
                        title="",
                        examples=[
                            "Wie viele Aufträge haben wir?",
                            "Die 10 umsatzstärksten Aufträge mit Kundeninformationen",
                            "Gesamtumsatz im Jahr 2023",
                            "Wer sind die Top 5 Kunden nach Umsatz?",
                            "Zeige alle Lieferanten",
                            "Wie viele Aufträge hat jeder Kunde?",
                        ],
                    )

                # ── Admin Tab ─────────────────────────────────────────────────
                with gr.Tab("⚙️ Admin"):
                    gr.Markdown("## Admin Panel")

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📚 Add Golden SQL")
                            gr.Markdown(
                                "_Add verified Q&A pairs to improve accuracy_"
                            )
                            golden_q = gr.Textbox(
                                label="Question",
                                placeholder="Gesamtumsatz im Jahr 2023"
                            )
                            golden_sql = gr.Textbox(
                                label="Verified SQL",
                                placeholder="SELECT SUM(Bruttosumm) ...",
                                lines=4,
                            )
                            golden_btn = gr.Button(
                                "Add Golden SQL", variant="primary"
                            )
                            golden_out = gr.Textbox(
                                label="Result", interactive=False
                            )
                            golden_btn.click(
                                fn=self.add_golden_sql,
                                inputs=[golden_q, golden_sql],
                                outputs=golden_out,
                            )

                        with gr.Column():
                            gr.Markdown("### 📋 Add Business Instruction")
                            gr.Markdown(
                                "_Rules the AI must always follow_"
                            )
                            instr_input = gr.Textbox(
                                label="Instruction",
                                placeholder="Never use the Salary column",
                                lines=2,
                            )
                            instr_btn = gr.Button(
                                "Add Instruction", variant="primary"
                            )
                            instr_out = gr.Textbox(
                                label="Result", interactive=False
                            )
                            instr_btn.click(
                                fn=self.add_instruction,
                                inputs=instr_input,
                                outputs=instr_out,
                            )

                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 🔄 Schema Management")
                            rescan_btn = gr.Button(
                                "Rescan Database Schema",
                                variant="secondary"
                            )
                            rescan_out = gr.Textbox(
                                label="Result", interactive=False, lines=4
                            )
                            rescan_btn.click(
                                fn=self.rescan_schema, outputs=rescan_out
                            )

                        with gr.Column():
                            gr.Markdown("### 📊 System Stats")
                            stats_btn = gr.Button(
                                "Refresh Stats", variant="secondary"
                            )
                            stats_out = gr.Textbox(
                                label="Stats", interactive=False, lines=6
                            )
                            stats_btn.click(
                                fn=self.get_stats, outputs=stats_out
                            )

                    gr.Markdown("### 🕐 Recent Query History")
                    history_btn = gr.Button("Load History")
                    history_out = gr.Textbox(
                        label="History", interactive=False, lines=10
                    )
                    history_btn.click(
                        fn=self.get_history, outputs=history_out
                    )

                # ── Benchmark Tab ─────────────────────────────────────────────
                with gr.Tab("📊 Benchmark"):
                    gr.Markdown("## Accuracy Benchmark")
                    gr.Markdown(
                        "_Run standardized tests to measure system accuracy_"
                    )
                    benchmark_btn = gr.Button(
                        "🚀 Run Benchmark (10 tests)",
                        variant="primary"
                    )
                    benchmark_out = gr.Textbox(
                        label="Results", interactive=False, lines=20
                    )
                    benchmark_btn.click(
                        fn=self.run_benchmark, outputs=benchmark_out
                    )

                # ── Feedback Tab ─────────────────────────────────────────────
                with gr.Tab("💬 Feedback"):
                    gr.Markdown("## Submit Query Feedback")
                    gr.Markdown(
                        "_Help improve the system by rating query results_"
                    )
                    with gr.Row():
                        with gr.Column():
                            fb_question = gr.Textbox(
                                label="Question", lines=2
                            )
                            fb_sql = gr.Textbox(
                                label="Generated SQL", lines=3
                            )
                            fb_rating = gr.Slider(
                                1, 5, value=3, step=1,
                                label="Rating (1-5 stars)"
                            )
                            fb_correct = gr.Checkbox(
                                label="SQL was correct", value=True
                            )
                        with gr.Column():
                            fb_correction = gr.Textbox(
                                label="Corrected SQL (if wrong)",
                                lines=3,
                                placeholder="SELECT ... (leave empty if correct)"
                            )
                            fb_comment = gr.Textbox(
                                label="Comment (optional)", lines=2
                            )
                    fb_btn = gr.Button("Submit Feedback", variant="primary")
                    fb_out = gr.Textbox(
                        label="Result", interactive=False
                    )
                    fb_btn.click(
                        fn=self.submit_feedback,
                        inputs=[
                            fb_question, fb_sql, fb_rating,
                            fb_correct, fb_correction, fb_comment
                        ],
                        outputs=fb_out,
                    )

                # ── Performance Tab ───────────────────────────────────────────
                with gr.Tab("⚡ Performance"):
                    gr.Markdown("## Performance Monitor")
                    gr.Markdown(
                        "_CPU/RAM usage and per-stage latency breakdown "
                        "(Stage 7.6)_"
                    )
                    perf_btn = gr.Button(
                        "Refresh Snapshot", variant="secondary"
                    )
                    perf_out = gr.Textbox(
                        label="Performance Snapshot",
                        interactive=False, lines=15
                    )
                    perf_btn.click(
                        fn=self.get_performance_snapshot, outputs=perf_out
                    )

                # ── Pilot KPIs Tab ────────────────────────────────────────────
                with gr.Tab("📈 Pilot KPIs"):
                    gr.Markdown("## Pilot Program KPIs")
                    gr.Markdown(
                        "_Track system performance and business value metrics_"
                    )
                    kpi_btn = gr.Button(
                        "Refresh KPIs", variant="secondary"
                    )
                    kpi_out = gr.Textbox(
                        label="KPI Dashboard",
                        interactive=False, lines=15
                    )
                    kpi_btn.click(
                        fn=self.get_pilot_kpis, outputs=kpi_out
                    )

        return demo


def create_app() -> gr.Blocks:
    """Factory function — creates and initializes the app."""
    app = ERP42GradioApp()
    app.startup()
    return app.build_ui()
