"""
api/routes/health.py — Health Check Endpoint
Step 1.8: Startup Pipeline health validation.
Stage 7.6: Performance Monitoring (CPU/RAM/latency breakdown).
"""

import time
from fastapi import APIRouter
from erp42_analytics.core.config import config
from erp42_analytics.api.queue.task_queue import get_task_queue
from erp42_analytics.intelligence.performance_monitor import get_performance_monitor

router = APIRouter()


@router.get("/")
async def health_check():
    """System health check — checks all components."""
    queue = get_task_queue()
    queue_stats = queue.get_queue_stats()
    perf = get_performance_monitor()

    return {
        "status": "healthy",
        "version": config.app.app_version,
        "model": config.model.llm_model,
        "database": config.db.database,
        "queue": queue_stats,
        "performance": perf.get_full_snapshot(),
        "timestamp": time.time(),
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check — is system ready to serve requests?"""
    from erp42_analytics.api.dependencies import get_query_service
    svc = get_query_service()
    tables = len(svc._all_tables)

    return {
        "ready": tables > 0,
        "tables_loaded": tables,
        "model": config.model.llm_model,
    }
