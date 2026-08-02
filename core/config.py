"""
core/config.py — Central Configuration
Single source of truth for all settings.
Load from environment variables or config.yaml if present.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# ─── Base Paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
CACHE_DIR = BASE_DIR / "cache"
VECTOR_DIR = BASE_DIR / "vector_store"

# Auto-create dirs
for d in [DATA_DIR, LOGS_DIR, CACHE_DIR, VECTOR_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Database Config ──────────────────────────────────────────────────────────
@dataclass
class DatabaseConfig:
    """ERP database connection settings."""
    server: str = os.getenv("ERP_SERVER", r"localhost\SQLEXPRESS")
    database: str = os.getenv("ERP_DATABASE", "ERP42test")
    driver: str = os.getenv("ERP_DRIVER", "ODBC Driver 17 for SQL Server")
    trusted_connection: bool = True
    trust_server_certificate: bool = True
    schema: str = "dbo"

    @property
    def connection_string(self) -> str:
        return (
            f"mssql+pyodbc://@{self.server}/{self.database}"
            f"?driver={self.driver.replace(' ', '+')}"
            f"&trusted_connection=yes"
            f"&TrustServerCertificate=yes"
        )

    @property
    def full_table_prefix(self) -> str:
        return f"{self.database}.{self.schema}"


# ─── AI Model Config ──────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Local AI model settings."""
    # SQL generation model
    llm_model: str = os.getenv("LLM_MODEL", "phi3:mini")
    llm_temperature: float = 0.0
    # 500 was too low — truncated any SQL with 2+ JOINs or inline comments
    # mid-clause (e.g. cut off inside "WHERE NOT EXISTS ("), which the
    # guardrail's paren-balancing then "repaired" into syntactically
    # plausible but still-broken SQL. See sql_service.py SQLGuardrailService.
    llm_max_tokens: int = 1536

    # Embedding model (via ChromaDB built-in)
    embedding_model: str = "all-MiniLM-L6-v2"

    # Metadata enrichment model
    enrichment_model: str = os.getenv("ENRICHMENT_MODEL", "mistral:latest")
    enrichment_max_tokens: int = 300

    # Ollama settings
    ollama_base_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")


# ─── Storage Config ───────────────────────────────────────────────────────────
@dataclass
class StorageConfig:
    """Storage paths and settings."""
    # SQLite metadata DB
    sqlite_path: str = str(DATA_DIR / "erp42.db")

    # ChromaDB vector store
    chroma_path: str = str(VECTOR_DIR / "chroma_db")

    # Collections
    schema_collection: str = "schema_tables"
    golden_sql_collection: str = "golden_sql"

    # Cache
    cache_path: str = str(CACHE_DIR / "query_cache.db")
    cache_ttl_hours: int = 24


# ─── Query Engine Config ──────────────────────────────────────────────────────
@dataclass
class QueryConfig:
    """SQL generation and execution settings."""
    # Table selection
    n_relevant_tables: int = 3
    n_candidates_multiplier: int = 10  # n_candidates = max(n*multiplier, 30)
    min_candidates: int = 30

    # Retrieval quality floor — added after UAT testing showed 44% of
    # queries (40/91) pulled in irrelevant tables because find_relevant_
    # tables() always padded to exactly n_relevant_tables regardless of
    # how low the trailing scores were. Real noise in the logs scored as
    # high as 0.45-0.48 (e.g. "akontakt"/"kontakt" for a customer-count
    # question), so the floor has to sit above that band, not just catch
    # near-zero scores.
    min_relevance_score: float = 0.50
    # Webshop-domain tables (eue*, sh_*) get an even higher bar — UAT
    # logs showed eueorditem/eueart/sh_ordit/sh_order repeatedly
    # contaminating ERP-core retrieval (auftrag/lager/kunde questions
    # pulling in webshop tables at score ~0.2-0.4).
    webshop_min_relevance_score: float = 0.55

    # Hybrid scoring
    keyword_boost_amount: float = 0.4

    # SQL execution
    max_rows: int = 500
    exec_timeout_seconds: int = 30
    max_sql_retries: int = 2

    # Golden SQL
    golden_sql_min_score: float = 0.3
    golden_sql_max_examples: int = 3


# ─── Security Config ─────────────────────────────────────────────────────────
@dataclass
class SecurityConfig:
    """Security and access control settings."""
    # Dangerous SQL operations to block
    blocked_statements: list = field(default_factory=lambda: [
        "DROP", "DELETE", "UPDATE", "INSERT",
        "ALTER", "TRUNCATE", "EXEC", "EXECUTE",
        "xp_", "sp_", "OPENROWSET", "BULK"
    ])

    # RBAC (Phase 1 — basic)
    rbac_enabled: bool = False
    default_role: str = "analyst"

    # Audit logging
    audit_enabled: bool = True
    audit_log_path: str = str(LOGS_DIR / "audit.log")


# ─── App Config ───────────────────────────────────────────────────────────────
@dataclass
class AppConfig:
    """Application settings."""
    app_name: str = "ERP42 Natural Language Analytics"
    app_version: str = "2.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Gradio UI
    gradio_port: int = int(os.getenv("GRADIO_PORT", "7860"))
    gradio_share: bool = False

    # FastAPI
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_host: str = "127.0.0.1"

    # Task Queue — concurrent users
    max_workers: int = int(os.getenv("MAX_WORKERS", "3"))
    task_timeout: float = float(os.getenv("TASK_TIMEOUT", "600"))

    # Schema scanning
    schema_max_age_hours: int = 24
    schema_batch_size: int = 50


# ─── AI Runtime Config (Step 1.4 — Runtime Abstraction Layer) ─────────────────
@dataclass
class RuntimeConfig:
    """
    Selects the inference backend.

    mode:
        "local"          → Ollama on this machine (dev / RTX 4090 on-site)
        "private_cloud"  → Telekom private-cloud API (pilot stage)

    Remote settings are only read when mode="private_cloud".
    Exact endpoint/auth to be confirmed with Paul (ibes) — the connector
    speaks the OpenAI-compatible chat completions API by default.
    """
    mode: str = os.getenv("AI_RUNTIME", "local")

    remote_base_url: str = os.getenv("REMOTE_LLM_URL", "")
    remote_api_key: str = os.getenv("REMOTE_LLM_KEY", "")
    remote_model: str = os.getenv("REMOTE_LLM_MODEL", "")
    request_timeout_s: int = int(os.getenv("REMOTE_LLM_TIMEOUT", "120"))

    # If the private cloud is down at startup, degrade to local Ollama
    # (slow but alive) instead of failing hard.
    fallback_to_local: bool = (
        os.getenv("RUNTIME_FALLBACK_LOCAL", "true").lower() == "true"
    )


# ─── Master Config ────────────────────────────────────────────────────────────
@dataclass
class Config:
    """Master config — single import point for entire system."""
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    query: QueryConfig = field(default_factory=QueryConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    app: AppConfig = field(default_factory=AppConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# ─── Singleton ────────────────────────────────────────────────────────────────
_config: Config = None


def get_config() -> Config:
    """Get global config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config


# Convenience alias
config = get_config()
