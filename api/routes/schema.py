"""
api/routes/schema.py — Schema Management Endpoints
"""

from fastapi import APIRouter, BackgroundTasks
from erp42_analytics.api.dependencies import get_query_service
from erp42_analytics.services.schema_service import SchemaService
from erp42_analytics.services.semantic_service import SemanticService

router = APIRouter()


@router.get("/")
async def get_schema_info():
    """Get schema overview."""
    svc = get_query_service()
    schema_svc = SchemaService()
    sem_svc = SemanticService()

    return {
        "tables_loaded": len(svc._all_tables),
        "tables_scanned": schema_svc.get_table_count(),
        "embeddings_indexed": sem_svc.count(),
        "schema_stale": schema_svc.is_schema_stale(),
    }


@router.post("/rescan")
async def rescan_schema(background_tasks: BackgroundTasks):
    """Trigger schema rescan in background."""
    def _rescan():
        schema_svc = SchemaService()
        sem_svc = SemanticService()

        results = schema_svc.scan_all_tables()
        all_tables = schema_svc.get_all_tables()
        sem_svc.index_tables(all_tables, force_reindex=True)

        # Reload query service
        svc = get_query_service()
        svc._all_tables = all_tables
        svc.reload_tables()

        return results

    background_tasks.add_task(_rescan)
    return {"message": "Schema rescan started in background"}


@router.get("/tables")
async def list_tables(limit: int = 50):
    """List scanned tables."""
    svc = get_query_service()
    tables = list(svc._all_tables.keys())
    return {
        "total": len(tables),
        "tables": tables[:limit],
    }


@router.get("/tables/{table_name}")
async def get_table_info(table_name: str):
    """Get info for specific table."""
    svc = get_query_service()
    info = svc._all_tables.get(table_name.lower())
    if not info:
        return {"error": f"Table '{table_name}' not found"}
    return {
        "table": table_name,
        "columns": info.get("columns", []),
        "row_count": info.get("row_count", 0),
        "business_description": info.get("business_description"),
    }
