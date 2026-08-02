"""
services/schema_service.py — Schema Intelligence Service
Migrates: schema_scanner.py + metadata_enricher.py
Covers: Stage 2 (Data & Schema Intelligence — Steps 2.1, 2.2, 2.3, 2.5)

Responsibilities:
- Scan ERP database schema (389 tables)
- Auto-detect foreign keys statistically
- Enrich tables with AI-generated business descriptions
- Manage schema lifecycle (stale detection, incremental updates)
"""

import json
import logging
import time
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.runtime import get_runtime
from erp42_analytics.core.database import get_mssql
from erp42_analytics.repositories.schema_repo import SchemaRepository

logger = logging.getLogger(__name__)

# ─── Cardinality Config (from Dataherald sql_server_scanner.py) ───────────────
MIN_CATEGORY_VALUE = 1    # Skip columns with only 1 distinct value
MAX_CATEGORY_VALUE = 100  # Only store sample values for low-cardinality columns
MAX_SAMPLE_ROWS = 3       # Sample rows per table for context


# ─── Metadata Enrichment Prompt ───────────────────────────────────────────────
ENRICHMENT_PROMPT = """You are a German ERP database expert for ibes AG ERP42 system.

Generate a RICH semantic description for this database table used for semantic search.
Follow this EXACT format:

"{table_name} [German synonyms] [English synonyms]. 
Contains [key columns with German/English meanings].
Use for: [5-8 natural language questions in German AND English].
Often joined with [related tables] for [purpose]."

Rules:
- Output ONLY the description, no explanations, no markdown
- Include both German and English terms
- Keep under 200 words total

Table name: {table_name}
Columns: {columns}

Description:"""


class SchemaService:
    """
    Manages schema scanning, FK detection, and metadata enrichment.
    Single service for all schema intelligence operations.
    """

    def __init__(self):
        self.repo = SchemaRepository()
        self.mssql = get_mssql()

    # ─── Schema Scanning ──────────────────────────────────────────────────────

    def scan_all_tables(self) -> dict:
        """
        Scan all tables from ERP database.
        Returns summary: {scanned, failed, total}
        """
        logger.info("Starting schema scan...")
        start = time.time()

        try:
            # Get all table names
            tables_query = """
                SELECT TABLE_NAME 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_TYPE = 'BASE TABLE'
                ORDER BY TABLE_NAME
            """
            columns, rows = self.mssql.execute(tables_query)
            table_names = [row[0] for row in rows]
            total = len(table_names)
            logger.info(f"Found {total} tables to scan")

        except Exception as e:
            logger.error(f"Failed to get table list: {e}")
            return {"scanned": 0, "failed": 1, "total": 0}

        scanned = 0
        failed = 0

        for i, table_name in enumerate(table_names):
            try:
                self._scan_single_table(table_name)
                scanned += 1

                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start
                    logger.info(
                        f"Progress: {i+1}/{total} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )

            except Exception as e:
                logger.warning(f"Failed to scan {table_name}: {e}")
                failed += 1

        elapsed = time.time() - start
        logger.info(
            f"Scan complete: {scanned} scanned, "
            f"{failed} failed in {elapsed:.0f}s"
        )
        return {"scanned": scanned, "failed": failed, "total": total}

    def _scan_single_table(self, table_name: str):
        """
        Scan a single table's columns with cardinality detection.
        Learned from Dataherald sql_server_scanner.py:
        - Detect low-cardinality columns (1-100 distinct values)
        - Store sample values for those columns → helps model use correct filter values
        - Get sample rows for table context
        """
        db_prefix = f"[{config.db.database}].[{config.db.schema}]"

        # Get columns with metadata
        col_query = f"""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_NAME = '{table_name}'
            ORDER BY ORDINAL_POSITION
        """
        _, col_rows = self.mssql.execute(col_query)
        if not col_rows:
            return

        # Build column list with cardinality values
        columns = []
        for row in col_rows:
            col_name, data_type, is_nullable = row[0], row[1], row[2]
            col_info = {
                "name": col_name,
                "type": data_type,
                "nullable": is_nullable == "YES",
                "low_cardinality": False,
                "categories": None,
            }

            # Cardinality detection (Dataherald pattern)
            # Only for text/varchar columns — numeric columns skip
            if data_type.lower() in ("varchar", "nvarchar", "char", "nchar", "text"):
                categories = self._get_cardinality_values(
                    table_name, col_name, db_prefix
                )
                if categories is not None:
                    col_info["low_cardinality"] = True
                    col_info["categories"] = categories

            columns.append(col_info)

        # Get row count
        try:
            _, count_rows = self.mssql.execute(
                f"SELECT COUNT(*) FROM {db_prefix}.[{table_name}]"
            )
            row_count = count_rows[0][0] if count_rows else 0
        except Exception:
            row_count = 0

        # Get sample rows (Dataherald: get_table_examples)
        examples = self._get_sample_rows(table_name, db_prefix)

        # Save to repository
        self.repo.save_table(
            table_name, columns,
            row_count=row_count,
            examples=examples,
        )

    def _get_cardinality_values(
        self, table_name: str, col_name: str, db_prefix: str
    ) -> list | None:
        """
        Get distinct values for low-cardinality columns.
        Learned from Dataherald SqlServerScanner.cardinality_values()

        Returns list of distinct values if 1 < count <= 100, else None.
        Example: column 'Land' → ['DE', 'AT', 'CH', 'FR']
        This helps model know to use WHERE Land = 'DE' not WHERE Land = 'Germany'
        """
        try:
            # Fast approximate count using APPROX_COUNT_DISTINCT (SQL Server 2019+)
            # Fallback to COUNT(DISTINCT) if not available
            try:
                count_query = (
                    f"SELECT APPROX_COUNT_DISTINCT([{col_name}]) "
                    f"FROM {db_prefix}.[{table_name}]"
                )
                _, rs = self.mssql.execute(count_query)
            except Exception:
                count_query = (
                    f"SELECT COUNT(DISTINCT [{col_name}]) "
                    f"FROM {db_prefix}.[{table_name}]"
                )
                _, rs = self.mssql.execute(count_query)

            if not rs or not rs[0] or rs[0][0] is None:
                return None

            count = rs[0][0]
            if not (MIN_CATEGORY_VALUE < count <= MAX_CATEGORY_VALUE):
                return None

            # Get actual distinct values
            cardinality_query = (
                f"SELECT TOP 101 [{col_name}] "
                f"FROM (SELECT DISTINCT [{col_name}] FROM {db_prefix}.[{table_name}]) AS subquery "
                f"WHERE [{col_name}] IS NOT NULL"
            )
            _, cardinality_rows = self.mssql.execute(cardinality_query)
            return [str(row[0]) for row in cardinality_rows if row[0] is not None]

        except Exception as e:
            logger.debug(f"Cardinality check failed for {table_name}.{col_name}: {e}")
            return None

    def _get_sample_rows(self, table_name: str, db_prefix: str) -> list:
        """
        Get sample rows from table for context.
        Learned from Dataherald get_table_examples()
        Returns list of dicts: [{col: val, ...}, ...]
        """
        try:
            _, col_rows = self.mssql.execute(
                f"SELECT TOP 1 COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION"
            )
            if not col_rows:
                return []

            _, rows = self.mssql.execute(
                f"SELECT TOP {MAX_SAMPLE_ROWS} * FROM {db_prefix}.[{table_name}]"
            )
            _, col_info = self.mssql.execute(
                f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS "
                f"WHERE TABLE_NAME = '{table_name}' ORDER BY ORDINAL_POSITION"
            )
            col_names = [r[0] for r in col_info]

            samples = []
            for row in rows:
                sample = {}
                for i, val in enumerate(row):
                    if i < len(col_names) and val is not None:
                        sample[col_names[i]] = str(val)[:50]  # truncate long values
                samples.append(sample)
            return samples

        except Exception as e:
            logger.debug(f"Sample rows failed for {table_name}: {e}")
            return []

    # ─── FK Detection (Stage 2.5 — Statistical Foreign Key Detection) ──────────

    def detect_foreign_keys(self, all_tables: dict = None) -> int:
        """
        Statistical FK detection — finds relationships even without
        explicit FK constraints.
        Returns count of detected relationships.
        """
        if all_tables is None:
            all_tables = self.repo.get_all_tables()

        detected = 0

        # Pattern-based detection for ERP42
        # Common FK patterns: table has column ending in "numm" or "id"
        # that matches another table's primary key
        fk_patterns = self._build_fk_patterns(all_tables)

        for pattern in fk_patterns:
            try:
                self.repo.save_join(
                    from_table=pattern["from_table"],
                    from_col=pattern["from_col"],
                    to_table=pattern["to_table"],
                    to_col=pattern["to_col"],
                    confidence=pattern["confidence"],
                    is_verified=False,
                )
                detected += 1
            except Exception as e:
                logger.debug(f"FK save failed: {e}")

        logger.info(f"Detected {detected} FK relationships")
        return detected

    def _build_fk_patterns(self, all_tables: dict) -> list:
        """
        Build FK patterns based on ERP42 naming conventions.
        ERP42 pattern: Kundennumm → kunde.Kundennumm
        """
        patterns = []

        # Known ERP42 FK relationships (from domain knowledge)
        known_fks = [
            # auftrag relationships
            {"from": ("auftrag", "Kundennumm"), "to": ("kunde", "Kundennumm"), "conf": 0.95},
            {"from": ("auftrag", "Mitarbeite"), "to": ("mitarbei", "Mitarbeite"), "conf": 0.95},
            {"from": ("auftrag", "Projekt"), "to": ("projekt", "Projekt"), "conf": 0.90},
            # rechnung relationships
            {"from": ("rechnung", "Kundennumm"), "to": ("kunde", "Kundennumm"), "conf": 0.95},
            {"from": ("rechnung", "Mitarbeite"), "to": ("mitarbei", "Mitarbeite"), "conf": 0.90},
            # bestell relationships
            {"from": ("bestell", "Lieferante"), "to": ("lieferan", "Lieferante"), "conf": 0.95},
            # angebot relationships
            {"from": ("angebot", "Kundennumm"), "to": ("kunde", "Kundennumm"), "conf": 0.90},
            # projekt relationships
            {"from": ("projekt", "Kundennumm"), "to": ("kunde", "Kundennumm"), "conf": 0.90},
        ]

        for fk in known_fks:
            from_table, from_col = fk["from"]
            to_table, to_col = fk["to"]

            # Verify both tables exist in schema
            if from_table in all_tables and to_table in all_tables:
                from_cols = [c["name"] if isinstance(c, dict) else c
                             for c in all_tables[from_table].get("columns", [])]
                to_cols = [c["name"] if isinstance(c, dict) else c
                           for c in all_tables[to_table].get("columns", [])]

                if from_col in from_cols and to_col in to_cols:
                    patterns.append({
                        "from_table": from_table,
                        "from_col": from_col,
                        "to_table": to_table,
                        "to_col": to_col,
                        "confidence": fk["conf"],
                    })

        return patterns

    # ─── Metadata Enrichment (Stage 2.3 — Metadata Enrichment Engine) ──────────

    def enrich_tables(
        self,
        model: str = None,
        skip_existing: bool = True,
        max_tables: int = None,
    ) -> dict:
        """
        Use local LLM to generate business descriptions for tables.
        Run once offline before deploying.
        """
        model = model or config.model.enrichment_model
        all_tables = self.repo.get_all_tables()

        if skip_existing:
            tables_to_enrich = {
                name: info for name, info in all_tables.items()
                if not info.get("business_description")
            }
        else:
            tables_to_enrich = all_tables

        if max_tables:
            tables_to_enrich = dict(
                list(tables_to_enrich.items())[:max_tables]
            )

        total = len(tables_to_enrich)
        logger.info(f"Enriching {total} tables with {model}")

        success = 0
        failed = 0
        start = time.time()

        for i, (table_name, info) in enumerate(tables_to_enrich.items(), 1):
            cols = [c["name"] if isinstance(c, dict) else c
                    for c in info.get("columns", [])]
            if not cols:
                continue

            try:
                description = self._generate_description(
                    table_name, cols, model
                )
                self.repo.save_business_description(table_name, description)
                success += 1

                if i % 10 == 0 or i == total:
                    elapsed = time.time() - start
                    avg = elapsed / i
                    remaining = (total - i) * avg
                    logger.info(
                        f"Progress: {i}/{total} | "
                        f"Elapsed: {elapsed/60:.1f}min | "
                        f"Remaining: {remaining/60:.1f}min"
                    )

            except Exception as e:
                logger.error(f"Enrichment failed for {table_name}: {e}")
                failed += 1

        total_time = time.time() - start
        return {
            "success": success,
            "failed": failed,
            "total_time_minutes": round(total_time / 60, 1)
        }

    def _generate_description(
        self, table_name: str, columns: list, model: str
    ) -> str:
        """Generate AI description for a single table."""
        col_str = ", ".join(columns[:25])
        prompt = ENRICHMENT_PROMPT.format(
            table_name=table_name,
            columns=col_str
        )
        response = get_runtime().chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0,
                "num_predict": config.model.enrichment_max_tokens,
            }
        )
        description = response["message"]["content"].strip()
        description = description.replace("Description:", "").strip()
        if description.startswith('"') and description.endswith('"'):
            description = description[1:-1]
        return description

    # ─── Lifecycle Management ─────────────────────────────────────────────────

    def is_schema_stale(self, max_age_hours: int = None) -> bool:
        """Check if schema needs rescanning."""
        hours = max_age_hours or config.app.schema_max_age_hours
        return self.repo.is_schema_stale(hours)

    def get_table_count(self) -> int:
        """Get number of scanned tables."""
        return self.repo.get_table_count()

    def get_all_tables(self) -> dict:
        """Get all scanned table metadata."""
        return self.repo.get_all_tables()
