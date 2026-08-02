"""
api/routes/query.py — Direct SQL Query Endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from erp42_analytics.api.dependencies import get_query_service
from erp42_analytics.repositories.golden_sql_repo import GoldenSQLRepository
from erp42_analytics.repositories.history_repo import HistoryRepository

router = APIRouter()


class GoldenSQLRequest(BaseModel):
    question: str
    sql: str
    tables_used: Optional[list] = []


class InstructionRequest(BaseModel):
    instruction: str


@router.post("/golden-sql")
async def add_golden_sql(request: GoldenSQLRequest):
    """Add verified Q&A pair to Golden SQL store."""
    golden = GoldenSQLRepository()
    success = golden.add(
        question=request.question,
        sql=request.sql,
        tables_used=request.tables_used,
    )
    if not success:
        raise HTTPException(500, "Failed to add Golden SQL")
    return {"message": f"Golden SQL added: '{request.question[:50]}'"}


@router.get("/golden-sql")
async def list_golden_sql():
    """List all Golden SQL entries."""
    golden = GoldenSQLRepository()
    entries = golden.get_all()
    return {
        "count": len(entries),
        "entries": [
            {"id": e.id, "question": e.question, "sql": e.sql}
            for e in entries
        ]
    }


@router.post("/instructions")
async def add_instruction(request: InstructionRequest):
    """Add business instruction."""
    history = HistoryRepository()
    history.save_instruction(request.instruction)
    return {"message": f"Instruction added: '{request.instruction[:60]}'"}


@router.get("/history")
async def get_history(limit: int = 10):
    """Get recent query history."""
    history = HistoryRepository()
    return history.get_recent(limit=limit)


@router.get("/stats")
async def get_stats():
    """Get system statistics."""
    history = HistoryRepository()
    return history.get_stats()
