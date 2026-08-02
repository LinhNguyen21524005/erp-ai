"""
api/dependencies.py — Shared API Dependencies
FastAPI dependency injection for shared services.
"""

import logging
from functools import lru_cache
from typing import Optional

from erp42_analytics.services.query_service import QueryService
from erp42_analytics.services.schema_service import SchemaService
from erp42_analytics.services.semantic_service import SemanticService

logger = logging.getLogger(__name__)

_query_service: Optional[QueryService] = None


def get_query_service() -> QueryService:
    """Get singleton QueryService — initialized once at startup."""
    global _query_service
    if _query_service is None:
        logger.info("Initializing QueryService...")
        schema_svc = SchemaService()

        # Load or scan schema
        if schema_svc.is_schema_stale():
            logger.info("Schema stale — rescanning...")
            schema_svc.scan_all_tables()

        all_tables = schema_svc.get_all_tables()

        # Init embeddings
        sem_svc = SemanticService()
        if not sem_svc.is_indexed():
            logger.info("Indexing schema embeddings...")
            sem_svc.index_tables(all_tables)

        _query_service = QueryService(all_tables=all_tables)
        logger.info(
            f"QueryService ready: {len(all_tables)} tables"
        )

    return _query_service
