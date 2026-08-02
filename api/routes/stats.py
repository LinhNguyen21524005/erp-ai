"""
api/routes/stats.py — Performance Stats Export API (Step 7.4)

Requested by Felix (ibes AI Consultant): TTFT, tokens/second, average
tokens per request, and system load (GPU, CPU, RAM, disk read/write).

Endpoints:
    GET /stats                → current snapshot (resources + latency +
                                  inference aggregates) — same shape as
                                  /health but focused purely on performance
    GET /stats/inference       → inference-only aggregate (the 3 numbers
                                  Felix specifically asked about)
    GET /stats/export          → full JSON export incl. raw history
    GET /stats/export.csv      → inference history as CSV (Excel-friendly)

Note for Paul: if you're building the same tracking, this module can be
the single source of truth — call get_performance_monitor().record_from_
runtime_response(...) from your own call sites and consume these same
endpoints, rather than maintaining a parallel tracker.
"""

import csv
import io

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from erp42_analytics.intelligence.performance_monitor import get_performance_monitor

router = APIRouter()


@router.get("/")
async def current_stats():
    """Current performance snapshot — resources, latency, inference."""
    perf = get_performance_monitor()
    return perf.get_full_snapshot()


@router.get("/inference")
async def inference_stats():
    """
    The specific metrics requested by Felix:
    TTFT, tokens/second, average input+output tokens per request.
    """
    perf = get_performance_monitor()
    return perf.get_inference_stats()


@router.get("/export")
async def export_stats_json():
    """Full JSON export — snapshot + raw per-call inference history."""
    perf = get_performance_monitor()
    return perf.export_stats()


@router.get("/export.csv")
async def export_stats_csv():
    """Inference history as a downloadable CSV file."""
    perf = get_performance_monitor()
    rows = perf.export_stats_csv_rows()

    buffer = io.StringIO()
    fieldnames = [
        "timestamp", "runtime", "ttft_ms", "generation_ms",
        "total_duration_ms", "tokens_per_second",
        "input_tokens", "output_tokens", "total_tokens", "approximate",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inference_stats.csv"},
    )
