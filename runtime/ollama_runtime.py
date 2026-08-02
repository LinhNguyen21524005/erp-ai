"""
runtime/ollama_runtime.py — Local AI Runtime (Ollama)

Wraps the existing `ollama` python client behind the BaseRuntime
interface. Response shape is passed through unchanged (it already IS
the ollama shape), with `usage` and `runtime` fields added.

restart() implements the model unload/reload cycle that fixes the
CPU context-accumulation degeneration observed after long batch runs
(Q078–Q100 incident): keep_alive=0 forces Ollama to unload the model,
the next call reloads it with a clean state.
"""

import logging
import time
from typing import Optional

import ollama

from erp42_analytics.runtime.base_runtime import BaseRuntime, RuntimeHealth

logger = logging.getLogger(__name__)


class OllamaRuntime(BaseRuntime):
    name = "ollama"

    def __init__(self, base_url: Optional[str] = None):
        # ollama lib reads OLLAMA_HOST env; explicit client for clarity
        self._client = (
            ollama.Client(host=base_url) if base_url else ollama
        )
        self._base_url = base_url or "http://localhost:11434"

    # ── inference ─────────────────────────────────────────────────────
    def chat(
        self,
        model: str,
        messages: list,
        options: Optional[dict] = None,
    ) -> dict:
        response = self._client.chat(
            model=model,
            messages=messages,
            options=options or {},
        )
        content = response["message"]["content"]

        # Ollama returns these duration fields even in NON-streaming mode
        # (nanoseconds). This means TTFT and tokens/sec are measurable
        # WITHOUT switching the pipeline to streaming responses:
        #   load_duration        — time to load/warm the model
        #   prompt_eval_duration — time processing the input prompt
        #   eval_duration        — time spent generating output tokens
        # TTFT ≈ everything that happens before the first output token,
        # i.e. load + prompt processing.
        load_ns = response.get("load_duration", 0) or 0
        prompt_eval_ns = response.get("prompt_eval_duration", 0) or 0
        eval_ns = response.get("eval_duration", 0) or 0
        eval_count = response.get("eval_count", 0) or 0

        ttft_ms = (load_ns + prompt_eval_ns) / 1e6 if (load_ns or prompt_eval_ns) else None
        generation_ms = eval_ns / 1e6 if eval_ns else None
        tokens_per_second = (
            round(eval_count / (eval_ns / 1e9), 2)
            if eval_ns and eval_count else None
        )

        return {
            "message": {"content": content},
            "usage": {
                "input_tokens": response.get("prompt_eval_count"),
                "output_tokens": eval_count or None,
            },
            "timing": {
                "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
                "generation_ms": round(generation_ms, 1) if generation_ms is not None else None,
                "total_duration_ms": round((response.get("total_duration", 0) or 0) / 1e6, 1),
                "tokens_per_second": tokens_per_second,
            },
            "runtime": self.name,
        }

    # ── health ────────────────────────────────────────────────────────
    def health(self) -> RuntimeHealth:
        start = time.time()
        try:
            listed = self._client.list()
            models = [
                m.get("name", "") for m in listed.get("models", [])
            ]
            return RuntimeHealth(
                healthy=True,
                runtime_name=self.name,
                latency_ms=round((time.time() - start) * 1000, 1),
                model_available=len(models) > 0,
                detail=f"{len(models)} model(s) available",
            )
        except Exception as e:  # noqa: BLE001
            return RuntimeHealth(
                healthy=False,
                runtime_name=self.name,
                latency_ms=round((time.time() - start) * 1000, 1),
                detail=f"unreachable: {e}",
            )

    # ── restart (degeneration fix) ────────────────────────────────────
    def restart(self, model: Optional[str] = None) -> bool:
        """
        Force-unload the model to clear accumulated runtime state.

        Sending a request with keep_alive=0 makes Ollama unload the
        model immediately after responding; the next real call reloads
        it fresh. This is the automated version of the manual
        'restart Ollama every 20–30 queries' workaround.
        """
        try:
            from erp42_analytics.core.config import config
            target = model or config.model.llm_model
            self._client.chat(
                model=target,
                messages=[{"role": "user", "content": "ok"}],
                options={"num_predict": 1},
                keep_alive=0,
            )
            logger.info(
                "OllamaRuntime: model '%s' unloaded (state cleared)",
                target,
            )
            return True
        except Exception as e:  # noqa: BLE001
            logger.warning("OllamaRuntime: restart failed: %s", e)
            return False
