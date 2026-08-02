"""
intelligence/join_graph.py — Join Graph Engine
Stage 2.6: Build and query relationship graph between ERP42 tables.

Responsibilities:
- Build graph of table relationships
- Find shortest join path between tables
- Suggest required JOINs for multi-table queries
- Detect circular/invalid joins
"""

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

from erp42_analytics.core.models import JoinRelationship
from erp42_analytics.repositories.schema_repo import SchemaRepository

logger = logging.getLogger(__name__)


@dataclass
class JoinPath:
    """A path of JOINs connecting multiple tables."""
    tables: list = field(default_factory=list)
    joins: list = field(default_factory=list)  # List[JoinRelationship]
    total_confidence: float = 1.0

    def to_sql_joins(self, db_prefix: str = "ERP42test.dbo") -> str:
        """Convert join path to SQL JOIN clauses."""
        if not self.joins:
            return ""
        parts = []
        for join in self.joins:
            parts.append(
                f"JOIN {db_prefix}.{join.to_table} "
                f"ON {join.from_table}.{join.from_column} "
                f"= {join.to_table}.{join.to_column}"
            )
        return "\n".join(parts)


class JoinGraph:
    """
    Directed graph of table relationships.
    Nodes = tables, Edges = join relationships.

    Used by:
    - Intent engine to suggest required JOINs
    - SQL generator to build correct JOIN clauses
    - Validation engine to verify join correctness
    """

    # Known ERP42 relationships (Stage 2.5 — Statistical FK Detection results)
    CORE_RELATIONSHIPS = [
        # auftrag is the central table
        JoinRelationship("auftrag", "Kundennumm", "kunde", "Kundennumm", 0.95, True),
        JoinRelationship("auftrag", "Mitarbeite", "mitarbei", "Mitarbeite", 0.95, True),
        JoinRelationship("auftrag", "Projekt", "projekt", "Projekt", 0.90, True),
        # rechnung
        JoinRelationship("rechnung", "Kundennumm", "kunde", "Kundennumm", 0.95, True),
        JoinRelationship("rechnung", "Mitarbeite", "mitarbei", "Mitarbeite", 0.90, True),
        JoinRelationship("rechnung", "Projekt", "projekt", "Projekt", 0.85, True),
        # bestell
        JoinRelationship("bestell", "Lieferante", "lieferan", "Lieferante", 0.95, True),
        JoinRelationship("bestell", "Mitarbeite", "mitarbei", "Mitarbeite", 0.85, True),
        # angebot
        JoinRelationship("angebot", "Kundennumm", "kunde", "Kundennumm", 0.90, True),
        JoinRelationship("angebot", "Mitarbeite", "mitarbei", "Mitarbeite", 0.85, True),
        # projekt
        JoinRelationship("projekt", "Kundennumm", "kunde", "Kundennumm", 0.90, True),
        # lager
        JoinRelationship("lager", "Warengrupp", "warengru", "Warengrupp", 0.85, False),
    ]

    def __init__(self):
        self.repo = SchemaRepository()
        self._graph: dict = defaultdict(list)  # {table: [JoinRelationship]}
        self._reverse: dict = defaultdict(list)  # {to_table: [JoinRelationship]}
        self._built = False

    def build(self, include_db_joins: bool = True):
        """
        Build the join graph from:
        1. Hard-coded core relationships (verified)
        2. DB-detected relationships (from schema_repo)
        """
        # Load core relationships
        for rel in self.CORE_RELATIONSHIPS:
            self._add_edge(rel)

        # Load additional from DB
        if include_db_joins:
            db_joins = self.repo.get_all_joins()
            for row in db_joins:
                rel = JoinRelationship(
                    from_table=row["from_table"],
                    from_column=row["from_column"],
                    to_table=row["to_table"],
                    to_column=row["to_column"],
                    confidence=row["confidence"],
                    is_verified=bool(row["is_verified"]),
                )
                self._add_edge(rel)

        self._built = True
        total = sum(len(v) for v in self._graph.values())
        logger.info(f"Join graph built: {len(self._graph)} tables, {total} relationships")

    def _add_edge(self, rel: JoinRelationship):
        """Add bidirectional edge to graph."""
        # Forward: from_table → to_table
        if rel not in self._graph[rel.from_table]:
            self._graph[rel.from_table].append(rel)
        # Reverse: to_table → from_table
        reverse = JoinRelationship(
            from_table=rel.to_table,
            from_column=rel.to_column,
            to_table=rel.from_table,
            to_column=rel.from_column,
            confidence=rel.confidence,
            is_verified=rel.is_verified,
        )
        if reverse not in self._reverse[rel.to_table]:
            self._reverse[rel.to_table].append(reverse)

    def find_join_path(
        self, from_table: str, to_table: str
    ) -> Optional[JoinPath]:
        """
        BFS to find shortest join path between two tables.
        Returns JoinPath or None if no path found.
        """
        if not self._built:
            self.build()

        if from_table == to_table:
            return JoinPath(tables=[from_table], joins=[])

        # BFS
        queue = deque([(from_table, [from_table], [])])
        visited = {from_table}

        while queue:
            current, path_tables, path_joins = queue.popleft()

            # Get all neighbors (forward + reverse)
            neighbors = (
                self._graph.get(current, []) +
                self._reverse.get(current, [])
            )

            for rel in neighbors:
                next_table = (
                    rel.to_table if rel.from_table == current
                    else rel.from_table
                )

                if next_table in visited:
                    continue

                new_tables = path_tables + [next_table]
                new_joins = path_joins + [rel]

                if next_table == to_table:
                    confidence = min(r.confidence for r in new_joins)
                    return JoinPath(
                        tables=new_tables,
                        joins=new_joins,
                        total_confidence=confidence,
                    )

                visited.add(next_table)
                queue.append((next_table, new_tables, new_joins))

        logger.debug(f"No join path found: {from_table} → {to_table}")
        return None

    def get_direct_joins(self, table: str) -> list:
        """Get all tables directly joinable from given table."""
        if not self._built:
            self.build()

        joins = (
            self._graph.get(table, []) +
            self._reverse.get(table, [])
        )
        return joins

    def suggest_joins_for_tables(self, tables: list) -> list:
        """
        Given a list of tables that need to be in a query,
        suggest the minimal set of JOIN relationships.

        Example:
            tables = ["auftrag", "kunde", "mitarbei"]
            → returns JOINs to connect all 3
        """
        if not self._built:
            self.build()

        if len(tables) <= 1:
            return []

        # Use first table as anchor, find paths to all others
        anchor = tables[0]
        all_joins = []
        seen_pairs = set()

        for target in tables[1:]:
            path = self.find_join_path(anchor, target)
            if path:
                for join in path.joins:
                    pair = (join.from_table, join.to_table)
                    if pair not in seen_pairs:
                        all_joins.append(join)
                        seen_pairs.add(pair)
                        seen_pairs.add((join.to_table, join.from_table))

        return all_joins

    def are_tables_related(self, table_a: str, table_b: str) -> bool:
        """Check if two tables have a known relationship."""
        path = self.find_join_path(table_a, table_b)
        return path is not None

    def get_graph_summary(self) -> dict:
        """Get summary of the join graph."""
        if not self._built:
            self.build()
        return {
            "tables": len(self._graph),
            "relationships": sum(len(v) for v in self._graph.values()),
            "verified": sum(
                1 for rels in self._graph.values()
                for r in rels if r.is_verified
            ),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
_join_graph: Optional[JoinGraph] = None


def get_join_graph() -> JoinGraph:
    """Get global join graph singleton."""
    global _join_graph
    if _join_graph is None:
        _join_graph = JoinGraph()
        _join_graph.build()
    return _join_graph
