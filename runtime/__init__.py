"""
runtime/ — AI Runtime Abstraction Layer (SEPT Solution Step 1.4)

Public API:
    get_runtime()      → current BaseRuntime (local Ollama or private cloud)
    switch_runtime(m)  → force-switch backend ("local" | "private_cloud")
"""

from erp42_analytics.runtime.base_runtime import BaseRuntime, RuntimeHealth
from erp42_analytics.runtime.ollama_runtime import OllamaRuntime
from erp42_analytics.runtime.remote_runtime import RemoteRuntime
from erp42_analytics.runtime.runtime_selector import (
    get_runtime,
    switch_runtime,
)

__all__ = [
    "BaseRuntime",
    "RuntimeHealth",
    "OllamaRuntime",
    "RemoteRuntime",
    "get_runtime",
    "switch_runtime",
]
