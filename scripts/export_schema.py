"""
scripts/export_schema.py — Quick Schema Exporter

Standalone tool to dump the structure of one or more tables from any
SQL Server database (ERP42test today, the real ibes DB later — same
tool, just change --tables or point ERP_DATABASE elsewhere).

Does NOT touch ChromaDB, embeddings, or the repository layer — this is
a lightweight INFORMATION_SCHEMA reader for quick human/LLM inspection,
useful before running the full SchemaService.scan_all_tables() pipeline.

Outputs, per table:
    - columns: name, data type, nullable, default, identity
    - primary key columns
    - foreign keys (outgoing) — this table → referenced table
    - foreign keys (incoming) — other tables referencing this one
    - row count (cheap COUNT(*), useful sanity check on empty vs seeded DB)

Usage:
    # Default: the 10 core tables from semantic_service.IMPORTANT_COLUMNS
    python -m erp42_analytics.scripts.export_schema

    # Explicit table list
    python -m erp42_analytics.scripts.export_schema \
        --tables auftrag kunde rechnung lieferan

    # Every table in the database (use with care on the real ibes DB)
    python -m erp42_analytics.scripts.export_schema --all

    # Write markdown too (nice for pasting into a chat / doc)
    python -m erp42_analytics.scripts.export_schema --markdown

Output files (default: schema_export/):
    schema_export/schema_<db>.json      — full structured dump
    schema_export/schema_<db>.md        — human-readable (if --markdown)
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logger = logging.getLogger(__name__)

# These are the ACTUAL 10 core tables, taken directly from
# services/semantic_service.py :: IMPORTANT_COLUMNS — not a guess.
# Note the legacy 8-char-truncated names (dBase-era ERP convention):
# "lieferan" (Lieferant/supplier), "mitarbei" (Mitarbeiter/employee),
# "bestell" (Bestellung/purchase order), "fibubuch" (Finanzbuchhaltung/
# general ledger postings).
DEFAULT_CORE_TABLES = [
    "auftrag",    # sales order
    "kunde",      # customer
    "rechnung",   # invoice
    "lieferan",   # supplier          (truncated: Lieferant)
    "bestell",    # purchase order    (truncated: Bestellung)
    "lager",      # inventory/stock
    "fibubuch",   # general ledger / accounting postings
    "mitarbei",   # employee          (truncated: Mitarbeiter)
    "projekt",    # project
    "angebot",    # quotation/offer
]


def _get_mssql():
    from erp42_analytics.core.database import get_mssql
    return get_mssql()


def _safe_table_literal(table: str) -> str:
    """
    MSSQLManager.execute() takes a raw SQL string (no parameter binding).
    Table names here always come from INFORMATION_SCHEMA.TABLES (i.e.
    already-validated real table names from list_all_tables()), never
    from free-form user input — but we double-check the character set
    before inlining, as a defensive guard against typos/paste errors.
    """
    import re
    if not re.fullmatch(r"[A-Za-z0-9_]+", table):
        raise ValueError(
            f"Unexpected characters in table name '{table}' — refusing "
            f"to inline into SQL. Only letters, digits, underscore allowed."
        )
    return table


def list_all_tables(mssql) -> list:
    cols, rows = mssql.execute(
        """
        SELECT TABLE_NAME
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
    )
    return [r[0] for r in rows]


def get_columns(mssql, table: str) -> list:
    t = _safe_table_literal(table)
    cols, rows = mssql.execute(
        f"""
        SELECT
            c.COLUMN_NAME,
            c.DATA_TYPE,
            c.CHARACTER_MAXIMUM_LENGTH,
            c.NUMERIC_PRECISION,
            c.NUMERIC_SCALE,
            c.IS_NULLABLE,
            c.COLUMN_DEFAULT,
            COLUMNPROPERTY(
                OBJECT_ID(c.TABLE_SCHEMA + '.' + c.TABLE_NAME),
                c.COLUMN_NAME, 'IsIdentity'
            ) AS IS_IDENTITY
        FROM INFORMATION_SCHEMA.COLUMNS c
        WHERE c.TABLE_NAME = '{t}'
        ORDER BY c.ORDINAL_POSITION
        """
    )
    result = []
    for r in rows:
        name, dtype, char_len, num_prec, num_scale, nullable, default, is_identity = r
        type_str = dtype
        if char_len is not None and char_len > 0:
            type_str += f"({char_len})"
        elif char_len == -1:
            type_str += "(max)"
        elif num_prec is not None and dtype in ("decimal", "numeric"):
            type_str += f"({num_prec},{num_scale})"
        result.append({
            "name": name,
            "type": type_str,
            "nullable": nullable == "YES",
            "default": default,
            "is_identity": bool(is_identity),
        })
    return result


def get_primary_key(mssql, table: str) -> list:
    t = _safe_table_literal(table)
    cols, rows = mssql.execute(
        f"""
        SELECT ku.COLUMN_NAME
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE ku
            ON tc.CONSTRAINT_NAME = ku.CONSTRAINT_NAME
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
            AND tc.TABLE_NAME = '{t}'
        ORDER BY ku.ORDINAL_POSITION
        """
    )
    return [r[0] for r in rows]


def get_outgoing_fks(mssql, table: str) -> list:
    """FKs defined ON this table, pointing to other tables."""
    t = _safe_table_literal(table)
    cols, rows = mssql.execute(
        f"""
        SELECT
            fk.name AS fk_name,
            c1.name AS column_name,
            OBJECT_NAME(fk.referenced_object_id) AS ref_table,
            c2.name AS ref_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
            ON fkc.constraint_object_id = fk.object_id
        JOIN sys.columns c1
            ON c1.object_id = fkc.parent_object_id
            AND c1.column_id = fkc.parent_column_id
        JOIN sys.columns c2
            ON c2.object_id = fkc.referenced_object_id
            AND c2.column_id = fkc.referenced_column_id
        WHERE OBJECT_NAME(fk.parent_object_id) = '{t}'
        """
    )
    return [
        {"column": r[1], "references": f"{r[2]}.{r[3]}", "fk_name": r[0]}
        for r in rows
    ]


def get_incoming_fks(mssql, table: str) -> list:
    """FKs defined on OTHER tables, pointing TO this table."""
    t = _safe_table_literal(table)
    cols, rows = mssql.execute(
        f"""
        SELECT
            OBJECT_NAME(fk.parent_object_id) AS from_table,
            c1.name AS from_column,
            c2.name AS to_column
        FROM sys.foreign_keys fk
        JOIN sys.foreign_key_columns fkc
            ON fkc.constraint_object_id = fk.object_id
        JOIN sys.columns c1
            ON c1.object_id = fkc.parent_object_id
            AND c1.column_id = fkc.parent_column_id
        JOIN sys.columns c2
            ON c2.object_id = fkc.referenced_object_id
            AND c2.column_id = fkc.referenced_column_id
        WHERE OBJECT_NAME(fk.referenced_object_id) = '{t}'
        """
    )
    return [
        {"from_table": r[0], "from_column": r[1], "to_column": r[2]}
        for r in rows
    ]


def get_row_count(mssql, table: str) -> int:
    t = _safe_table_literal(table)
    try:
        cols, rows = mssql.execute(f"SELECT COUNT(*) FROM [{t}]")
        return int(rows[0][0]) if rows else 0
    except Exception as e:  # noqa: BLE001
        logger.warning("Row count failed for %s: %s", table, e)
        return -1  # signals "could not count" vs a real 0


def export_table(mssql, table: str) -> dict:
    logger.info("Exporting: %s", table)
    return {
        "table": table,
        "columns": get_columns(mssql, table),
        "primary_key": get_primary_key(mssql, table),
        "foreign_keys_out": get_outgoing_fks(mssql, table),
        "foreign_keys_in": get_incoming_fks(mssql, table),
        "row_count": get_row_count(mssql, table),
    }


def export_schema(tables: list, database_name: str = "") -> dict:
    mssql = _get_mssql()
    result = {
        "database": database_name,
        "exported_at": datetime.now().isoformat(),
        "tables": {},
        "not_found": [],
    }

    all_tables = set(list_all_tables(mssql))
    for t in tables:
        if t not in all_tables:
            logger.warning(
                "Table '%s' not found in database (skipping). "
                "Available tables sample: %s",
                t, sorted(all_tables)[:10],
            )
            result["not_found"].append(t)
            continue
        result["tables"][t] = export_table(mssql, t)

    return result


# ── Rendering ────────────────────────────────────────────────────────────

def to_markdown(export: dict) -> str:
    lines = [
        f"# Schema Export — {export['database'] or '(unnamed DB)'}",
        f"_Exported {export['exported_at']}_",
        "",
    ]
    if export["not_found"]:
        lines.append(
            f"⚠ Not found in database: {', '.join(export['not_found'])}"
        )
        lines.append("")

    for tname, t in export["tables"].items():
        lines.append(f"## {tname}")
        lines.append(f"Row count: **{t['row_count']}**")
        if t["primary_key"]:
            lines.append(f"Primary key: `{', '.join(t['primary_key'])}`")
        lines.append("")
        lines.append("| Column | Type | Nullable | Identity |")
        lines.append("|---|---|---|---|")
        for c in t["columns"]:
            lines.append(
                f"| {c['name']} | {c['type']} | "
                f"{'yes' if c['nullable'] else 'no'} | "
                f"{'yes' if c['is_identity'] else ''} |"
            )
        if t["foreign_keys_out"]:
            lines.append("")
            lines.append("**References (outgoing FKs):**")
            for fk in t["foreign_keys_out"]:
                lines.append(f"- `{fk['column']}` → `{fk['references']}`")
        if t["foreign_keys_in"]:
            lines.append("")
            lines.append("**Referenced by (incoming FKs):**")
            for fk in t["foreign_keys_in"]:
                lines.append(
                    f"- `{fk['from_table']}.{fk['from_column']}` "
                    f"→ this table.`{fk['to_column']}`"
                )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Export table structure from a SQL Server database "
                    "(columns, keys, FKs, row counts)."
    )
    parser.add_argument(
        "--tables", nargs="+", default=None,
        help=f"Table names to export. Default: the {len(DEFAULT_CORE_TABLES)} "
             f"core tables ({', '.join(DEFAULT_CORE_TABLES)})",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Export every table in the database (ignores --tables)",
    )
    parser.add_argument(
        "--markdown", action="store_true",
        help="Also write a human-readable .md file",
    )
    parser.add_argument(
        "--output-dir", default="schema_export",
        help="Output directory (default: schema_export/)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from erp42_analytics.core.config import config
    db_name = config.db.database

    if args.all:
        mssql = _get_mssql()
        tables = list_all_tables(mssql)
        logger.info("Exporting ALL %d tables from '%s'", len(tables), db_name)
    else:
        tables = args.tables or DEFAULT_CORE_TABLES
        logger.info(
            "Exporting %d table(s) from '%s': %s",
            len(tables), db_name, ", ".join(tables),
        )

    export = export_schema(tables, database_name=db_name)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"schema_{db_name}.json"
    json_path.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Wrote: %s", json_path)

    if args.markdown:
        md_path = out_dir / f"schema_{db_name}.md"
        md_path.write_text(to_markdown(export), encoding="utf-8")
        logger.info("Wrote: %s", md_path)

    n_ok = len(export["tables"])
    n_missing = len(export["not_found"])
    print(
        f"\nDone: {n_ok} table(s) exported"
        + (f", {n_missing} not found ({', '.join(export['not_found'])})"
           if n_missing else "")
    )


if __name__ == "__main__":
    main()
