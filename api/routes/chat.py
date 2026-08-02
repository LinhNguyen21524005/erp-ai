"""
api/routes/chat.py — Chat Endpoint
Handles natural language queries with async task queue.
"""

import asyncio
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from erp42_analytics.api.dependencies import get_query_service
from erp42_analytics.api.queue.task_queue import get_task_queue

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = ""
    user_id: Optional[str] = "anonymous"
    role: Optional[str] = "analyst"
    stream: Optional[bool] = False


class ChatResponse(BaseModel):
    task_id: str
    question: str
    sql: str
    result_rows: list
    result_columns: list
    row_count: int
    exec_time: float
    total_time: float
    trust_score: float
    confidence: str
    explanation: str
    warnings: list
    tables_used: list
    golden_examples_used: int
    from_cache: bool = False


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint — handles NL query with task queue.
    Supports 25-30 concurrent users via async processing.
    """
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    queue = get_task_queue()
    svc = get_query_service()

    # Submit to task queue and wait
    task = await queue.submit_and_wait(
        question=request.question,
        fn=svc.ask,
        session_id=request.session_id,
        user_id=request.user_id,
        role=request.role,
        timeout=600.0,
    )

    if task.status.value == "failed":
        raise HTTPException(500, task.error or "Query failed")

    response = task.result

    return ChatResponse(
        task_id=task.id,
        question=request.question,
        sql=response.sql,
        result_rows=response.result.rows or [],
        result_columns=response.result.columns or [],
        row_count=response.result.row_count,
        exec_time=response.result.exec_time,
        total_time=response.total_time,
        trust_score=response.trust_score.overall,
        confidence=response.trust_score.label,
        explanation=response.explanation,
        warnings=response.warnings or [],
        tables_used=response.tables_used or [],
        golden_examples_used=len(response.golden_examples_used or []),
    )


@router.post("/async")
async def chat_async(request: ChatRequest):
    """
    Async chat — returns task_id immediately, client polls for result.
    Better for long-running queries (>30s).
    """
    queue = get_task_queue()
    svc = get_query_service()

    task = await queue.submit(
        question=request.question,
        fn=svc.ask,
        session_id=request.session_id,
        user_id=request.user_id,
        role=request.role,
    )

    return {
        "task_id": task.id,
        "status": task.status.value,
        "message": "Query submitted. Poll /chat/status/{task_id} for result.",
        "queue_position": queue._queue.qsize(),
    }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Poll task status for async queries."""
    queue = get_task_queue()
    task = queue.get_task(task_id)

    if not task:
        raise HTTPException(404, f"Task {task_id} not found")

    response_data = task.to_dict()

    if task.status.value == "completed" and task.result:
        response = task.result
        response_data["result"] = {
            "sql": response.sql,
            "rows": response.result.rows or [],
            "columns": response.result.columns or [],
            "row_count": response.result.row_count,
            "trust_score": response.trust_score.overall,
            "confidence": response.trust_score.label,
            "explanation": response.explanation,
        }

    return response_data


@router.get("/queue")
async def get_queue_stats():
    """Get current queue statistics — for monitoring."""
    queue = get_task_queue()
    return queue.get_queue_stats()
