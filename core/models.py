"""
core/models.py — Core Data Models
Shared data structures used across all layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class QueryStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    PENDING = "pending"


class ConfidenceLevel(Enum):
    HIGH = "high"       # score >= 80
    MEDIUM = "medium"   # score >= 50
    LOW = "low"         # score < 50


class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


# ─── Schema Models ────────────────────────────────────────────────────────────

@dataclass
class ColumnInfo:
    """Single column metadata."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None
    sample_values: list = field(default_factory=list)
    null_ratio: float = 0.0
    distinct_count: int = 0


@dataclass
class TableInfo:
    """Single table metadata."""
    name: str
    schema: str = "dbo"
    database: str = "ERP42test"
    columns: list = field(default_factory=list)  # List[ColumnInfo]
    row_count: int = 0
    foreign_keys: list = field(default_factory=list)
    business_description: Optional[str] = None
    is_core_table: bool = False
    last_scanned: Optional[datetime] = None

    @property
    def full_name(self) -> str:
        return f"{self.database}.{self.schema}.{self.name}"

    @property
    def column_names(self) -> list:
        return [c.name if isinstance(c, ColumnInfo) else c.get("name", "") 
                for c in self.columns]


@dataclass
class JoinRelationship:
    """Relationship between two tables."""
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    confidence: float = 1.0
    is_verified: bool = False
    relationship_type: str = "many_to_one"  # one_to_one, one_to_many, many_to_many


# ─── Query Models ─────────────────────────────────────────────────────────────

@dataclass
class QueryIntent:
    """Structured representation of user's query intent."""
    raw_question: str
    language: str = "de"  # de, en, mixed
    intent_type: str = "retrieval"  # retrieval, aggregation, comparison, ranking
    metrics: list = field(default_factory=list)
    dimensions: list = field(default_factory=list)
    filters: dict = field(default_factory=dict)
    time_grain: Optional[str] = None  # day, month, quarter, year
    limit: Optional[int] = None
    needs_join: bool = False
    confidence: float = 1.0


@dataclass
class SQLResult:
    """Result of SQL generation and execution."""
    question: str
    sql: str
    status: QueryStatus = QueryStatus.PENDING
    rows: list = field(default_factory=list)
    columns: list = field(default_factory=list)
    row_count: int = 0
    exec_time: float = 0.0
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    tables_used: list = field(default_factory=list)


@dataclass
class TrustScore:
    """Enterprise trust score for a query result."""
    overall: float = 0.0            # 0-100
    retrieval_confidence: float = 0.0
    sql_reliability: float = 0.0
    validation_score: float = 0.0
    data_freshness: float = 100.0
    join_confidence: float = 100.0
    # Carried over from TrustBreakdown (Stage 5.6) so the full
    # component breakdown survives into the final response, not
    # just the 4 fields the lossy to_trust_score() used to keep.

    @property
    def level(self) -> ConfidenceLevel:
        if self.overall >= 80:
            return ConfidenceLevel.HIGH
        elif self.overall >= 50:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    @property
    def label(self) -> str:
        labels = {
            ConfidenceLevel.HIGH: "High",
            ConfidenceLevel.MEDIUM: "Medium",
            ConfidenceLevel.LOW: "Low"
        }
        return labels[self.level]


@dataclass
class AttemptLog:
    """
    Record of a single SQL generation attempt within the retry loop
    (Stage 4.8). One question may produce 1-3 of these (1 initial +
    up to max_retries corrections).
    """
    attempt_number: int = 1
    sql: str = ""
    stage_failed_at: Optional[str] = None
    # One of: None (succeeded), "guardrail", "hallucination_check",
    # "execution", "permission"
    error_message: Optional[str] = None
    error_category: Optional[str] = None
    # From ErrorAnalyzer: column_not_found, table_not_found,
    # syntax_error, timeout, datatype_mismatch, join_error,
    # permission_denied, unknown
    hallucination_severity: Optional[str] = None
    # Severity for THIS specific attempt's SQL (Stage 5.5):
    # none, low, medium, high, critical — independent of whether
    # the attempt ultimately succeeded at execution.
    latency_ms: float = 0.0  # total wall-clock time for this attempt
    sql_generation_ms: float = 0.0  # time spent in the LLM call (Step 9)
    db_execution_ms: float = 0.0    # time spent executing SQL (Step 13),
    # 0.0 if the attempt never reached execution (failed earlier)
    succeeded: bool = False


@dataclass
class QueryResponse:
    """Complete response returned to user."""
    question: str
    sql: str
    result: SQLResult
    trust_score: TrustScore
    explanation: str = ""
    tables_used: list = field(default_factory=list)
    golden_examples_used: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    total_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    attempts: list = field(default_factory=list)  # list[AttemptLog]
    hallucination_severity: Optional[str] = None
    # Severity of the FINAL accepted attempt only (Stage 5.5)
    max_hallucination_severity: Optional[str] = None
    # Worst severity seen across ALL attempts, including ones that
    # were corrected by retry — surfaces cases where an earlier
    # attempt hallucinated badly even though the final SQL is clean.


# ─── Golden SQL Models ────────────────────────────────────────────────────────

@dataclass
class GoldenSQL:
    """Verified question-SQL pair."""
    id: Optional[str] = None
    question: str = ""
    sql: str = ""
    tables_used: list = field(default_factory=list)
    created_at: Optional[datetime] = None
    verified: bool = True
    score: float = 0.0  # similarity score when retrieved


# ─── Audit Models ─────────────────────────────────────────────────────────────

@dataclass
class AuditEntry:
    """Audit log entry for enterprise compliance."""
    timestamp: datetime = field(default_factory=datetime.now)
    user_id: str = "anonymous"
    session_id: str = ""
    question: str = ""
    sql_generated: str = ""
    tables_accessed: list = field(default_factory=list)
    status: QueryStatus = QueryStatus.PENDING
    trust_score: float = 0.0
    exec_time: float = 0.0
    row_count: int = 0
    error: Optional[str] = None
