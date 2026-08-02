"""
runtime/base_runtime.py — AI Runtime Abstraction Layer (Step 1.4)

Defines the unified inference interface that every runtime backend
(local Ollama, Telekom private cloud, future backends) must implement.

Design goals:
- Call sites in services/ never import `ollama` directly. They call
  `get_runtime().chat(...)` and receive an OLLAMA-SHAPED response:
      {"message": {"content": "..."}, "usage": {...}}
  This keeps the migration diff in query_service / schema_service tiny.
- Switching between local and private-cloud inference is a config
  change (AI_RUNTIME=local | private_cloud), not a code change.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeHealth:
    """Result of a health probe against a runtime backend."""
    healthy: bool
    runtime_name: str
    latency_ms: float = 0.0
    model_available: bool = False
    detail: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class BaseRuntime(ABC):
    """
    Abstract inference backend.

    Contract for chat():
      - `messages` follows the ollama/OpenAI chat format:
            [{"role": "user", "content": "..."}]
      - `options` uses OLLAMA option names (temperature, num_predict).
        Each backend translates to its native parameter names.
      - Returns an ollama-shaped dict:
            {
              "message": {"content": str},
              "usage": {
                  "input_tokens": int | None,
                  "output_tokens": int | None,
              },
              "timing": {
                  "ttft_ms": float | None,           # time to first token
                  "generation_ms": float | None,      # time generating output
                  "total_duration_ms": float,          # full call wall-clock
                  "tokens_per_second": float | None,
                  "approximate": bool,                 # True if wall-clock-derived,
                                                        # not native runtime timing
              },
              "runtime": str,   # which backend served this call
            }
      - Raises RuntimeError (with a readable message) on failure;
        callers already handle exceptions per-attempt.
    """

    #: short identifier used in logs / audit ("ollama", "private_cloud")
    name: str = "base"

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list,
        options: Optional[dict] = None,
    ) -> dict:
        """Run one chat completion. See class docstring for contract."""
        raise NotImplementedError

    @abstractmethod
    def health(self) -> RuntimeHealth:
        """Cheap probe: is the backend reachable and the model loaded?"""
        raise NotImplementedError

    def restart(self) -> bool:
        """
        Restart / reset the backend if supported.

        Local Ollama: used by the batch runner's restart scheduler to
        clear accumulated context state (the Q078+ degeneration fix).
        Remote runtimes: stateless per-request → no-op returning True.
        """
        logger.debug("%s: restart() is a no-op for this runtime", self.name)
        return True

    # Convenience wrapper so call sites can do quick boolean checks
    def is_healthy(self) -> bool:
        try:
            return self.health().healthy
        except Exception:  # noqa: BLE001 — health must never raise
            return False
