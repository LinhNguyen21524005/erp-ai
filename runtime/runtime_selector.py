"""
runtime/runtime_selector.py — Runtime Selector (Step 1.4)

Single entry point for the rest of the codebase:

    from erp42_analytics.runtime import get_runtime
    response = get_runtime().chat(model=..., messages=[...], options={...})

Which backend you get is decided by config.runtime.mode:
    "local"          → OllamaRuntime           (default; dev + RTX 4090 on-site)
    "private_cloud"  → RemoteRuntime           (Telekom cloud, pilot stage)

Failover: if mode=private_cloud and fallback_to_local=True, an unhealthy
remote endpoint at selection time falls back to local Ollama with a loud
warning — so a cloud outage degrades to slow-but-working instead of down.
"""

import logging
from typing import Optional

from erp42_analytics.core.config import config
from erp42_analytics.runtime.base_runtime import BaseRuntime
from erp42_analytics.runtime.ollama_runtime import OllamaRuntime
from erp42_analytics.runtime.remote_runtime import RemoteRuntime

logger = logging.getLogger(__name__)

_runtime_singleton: Optional[BaseRuntime] = None


def _build_runtime() -> BaseRuntime:
    rc = config.runtime
    mode = (rc.mode or "local").lower()

    if mode == "private_cloud":
        try:
            remote = RemoteRuntime(
                base_url=rc.remote_base_url,
                api_key=rc.remote_api_key,
                default_model=rc.remote_model,
                timeout_s=rc.request_timeout_s,
            )
        except ValueError as e:
            logger.error("RemoteRuntime misconfigured: %s", e)
            remote = None

        if remote is not None:
            probe = remote.health()
            if probe.healthy:
                logger.info(
                    "Runtime: private_cloud selected (%s, %.0fms)",
                    rc.remote_base_url, probe.latency_ms,
                )
                return remote
            logger.warning(
                "Runtime: private_cloud unhealthy (%s)", probe.detail
            )

        if rc.fallback_to_local:
            logger.warning(
                "Runtime: FALLING BACK to local Ollama — "
                "responses will be slower until the cloud endpoint recovers."
            )
            return OllamaRuntime(base_url=config.model.ollama_base_url)

        raise RuntimeError(
            "private_cloud runtime unavailable and fallback_to_local=False"
        )

    # default: local
    logger.info("Runtime: local Ollama selected")
    return OllamaRuntime(base_url=config.model.ollama_base_url)


def get_runtime(force_rebuild: bool = False) -> BaseRuntime:
    """Return the process-wide runtime instance (lazy singleton)."""
    global _runtime_singleton
    if _runtime_singleton is None or force_rebuild:
        _runtime_singleton = _build_runtime()
    return _runtime_singleton


def switch_runtime(mode: str) -> BaseRuntime:
    """
    Explicitly switch runtime at runtime (admin panel / tests).
    Updates config.runtime.mode and rebuilds the singleton.
    """
    config.runtime.mode = mode
    logger.info("Runtime: switching to '%s'", mode)
    return get_runtime(force_rebuild=True)
