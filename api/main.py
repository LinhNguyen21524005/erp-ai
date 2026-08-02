"""
api/main.py — FastAPI Local Orchestration Server
Step 1.3: Central runtime coordinator for all AI requests (Local FastAPI Orchestration Server).

Features:
- Async request handling (no UI freeze)
- Task queue for concurrent users (25-30)
- Session isolation per user
- Streaming response support
- Health check endpoint
- Request tracing
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from erp42_analytics.core.config import config
from erp42_analytics.api.queue.task_queue import get_task_queue
from erp42_analytics.api.routes import chat, query, schema, health, stats

logger = logging.getLogger(__name__)


# ─── Lifespan (startup + shutdown) ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup."""
    logger.info(f"Starting {config.app.app_name} v{config.app.app_version}")

    # 1. Initialize task queue
    queue = get_task_queue()
    await queue.start()
    logger.info(f"Task queue started: {queue.max_workers} workers")

    # 2. Initialize query service (loads schema + embeddings)
    from erp42_analytics.api.dependencies import get_query_service
    svc = get_query_service()
    logger.info(f"Query service ready: {len(svc._all_tables)} tables")

    logger.info("✅ FastAPI server ready")
    yield

    # Shutdown
    await queue.stop()
    logger.info("FastAPI server stopped")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=config.app.app_name,
    version=config.app.app_version,
    description="Enterprise AI ERP Analytics — Local API Server",
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────

# CORS — allow desktop app and Gradio UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_tracing(request: Request, call_next):
    """Request tracing middleware — Stage 7.5 AI Execution Monitoring."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Attach request_id to request state
    request.state.request_id = request_id

    logger.debug(f"[{request_id}] {request.method} {request.url.path}")

    response = await call_next(request)

    duration = round((time.time() - start_time) * 1000, 2)
    logger.debug(f"[{request_id}] {response.status_code} in {duration}ms")

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration}ms"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler — Stage 6.2 SQL & Query Transparency (error path)."""
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "request_id": getattr(request.state, "request_id", "unknown"),
        }
    )


# ─── Routes ───────────────────────────────────────────────────────────────────

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(schema.router, prefix="/schema", tags=["Schema"])
app.include_router(stats.router, prefix="/stats", tags=["Performance Stats"])


@app.get("/")
async def root():
    return {
        "name": config.app.app_name,
        "version": config.app.app_version,
        "status": "running",
        "docs": "/docs",
    }
