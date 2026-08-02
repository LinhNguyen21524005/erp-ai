"""
runtime/remote_runtime.py — Remote AI Runtime Connector (Private Cloud API)

Connector for the Telekom private-cloud inference endpoint (Step 1.4,
"Remote AI Runtime Connector"). Implemented against the OpenAI-compatible
chat completions API (`POST {base_url}/v1/chat/completions`), which is
the de-facto standard exposed by vLLM, TGI, Ollama's OpenAI mode, and
most managed GPU offerings.

⚠ The exact Telekom API spec is confirmed with Paul (meeting 24 July).
   If it differs, only `_build_payload()` / `_parse_response()` need
   adjusting — the rest of the pipeline is untouched.

Differences vs local runtime, handled here:
- network faults → bounded retries with backoff (transport-level only;
  SQL-level retries remain the job of the existing Retry Engine)
- auth via Bearer API key
- per-request timeout (cloud GPU is fast; a hung request should fail
  fast instead of blocking the queue)
"""

import logging
import time
from typing import Optional

import requests

from erp42_analytics.runtime.base_runtime import BaseRuntime, RuntimeHealth

logger = logging.getLogger(__name__)


class RemoteRuntime(BaseRuntime):
    name = "private_cloud"

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        default_model: str = "",
        timeout_s: int = 120,
        transport_retries: int = 2,
    ):
        if not base_url:
            raise ValueError(
                "RemoteRuntime requires a base_url "
                "(set REMOTE_LLM_URL in the environment/config)."
            )
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_s = timeout_s
        self._transport_retries = transport_retries
        self._session = requests.Session()

    # ── helpers ───────────────────────────────────────────────────────
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_payload(
        self, model: str, messages: list, options: dict
    ) -> dict:
        """Translate ollama-style options → OpenAI parameter names."""
        payload = {
            "model": self._default_model or model,
            "messages": messages,
        }
        if "temperature" in options:
            payload["temperature"] = options["temperature"]
        if "num_predict" in options:  # ollama name → OpenAI name
            payload["max_tokens"] = options["num_predict"]
        return payload

    @staticmethod
    def _parse_response(data: dict) -> dict:
        """OpenAI response → ollama-shaped dict used by the pipeline."""
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {}) or {}
        return {
            "message": {"content": content},
            "usage": {
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
        }

    # ── inference ─────────────────────────────────────────────────────
    def chat(
        self,
        model: str,
        messages: list,
        options: Optional[dict] = None,
    ) -> dict:
        url = f"{self._base_url}/v1/chat/completions"
        payload = self._build_payload(model, messages, options or {})

        last_error: Optional[Exception] = None
        for attempt in range(1, self._transport_retries + 2):
            call_start = time.time()
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self._timeout_s,
                )
                resp.raise_for_status()
                wall_ms = (time.time() - call_start) * 1000
                parsed = self._parse_response(resp.json())
                parsed["runtime"] = self.name

                # Most OpenAI-compatible endpoints don't expose Ollama-style
                # load/prompt/eval duration breakdown in non-streaming mode,
                # so TTFT can't be measured precisely here (would require a
                # streaming call). We report a coarse tokens/sec using total
                # wall-clock time — labeled as an approximation, not exact.
                out_tokens = parsed["usage"].get("output_tokens")
                parsed["timing"] = {
                    "ttft_ms": None,  # not measurable without streaming
                    "generation_ms": None,
                    "total_duration_ms": round(wall_ms, 1),
                    "tokens_per_second": (
                        round(out_tokens / (wall_ms / 1000), 2)
                        if out_tokens and wall_ms else None
                    ),
                    "approximate": True,
                }
                return parsed
            except (requests.ConnectionError, requests.Timeout) as e:
                # transient network fault → retry with small backoff
                last_error = e
                wait = 2 * attempt
                logger.warning(
                    "RemoteRuntime: transport error "
                    "(attempt %d/%d): %s — retrying in %ds",
                    attempt, self._transport_retries + 1, e, wait,
                )
                time.sleep(wait)
            except requests.HTTPError as e:
                # 4xx/5xx from the endpoint → not retryable here
                body = e.response.text[:300] if e.response is not None else ""
                raise RuntimeError(
                    f"Private-cloud API error "
                    f"{e.response.status_code if e.response else '?'}: {body}"
                ) from e

        raise RuntimeError(
            f"Private-cloud endpoint unreachable after "
            f"{self._transport_retries + 1} attempts: {last_error}"
        )

    # ── health ────────────────────────────────────────────────────────
    def health(self) -> RuntimeHealth:
        start = time.time()
        # /v1/models is the cheapest OpenAI-compatible probe
        try:
            resp = self._session.get(
                f"{self._base_url}/v1/models",
                headers=self._headers(),
                timeout=10,
            )
            resp.raise_for_status()
            n_models = len(resp.json().get("data", []))
            return RuntimeHealth(
                healthy=True,
                runtime_name=self.name,
                latency_ms=round((time.time() - start) * 1000, 1),
                model_available=n_models > 0,
                detail=f"{n_models} model(s) exposed",
            )
        except Exception as e:  # noqa: BLE001
            return RuntimeHealth(
                healthy=False,
                runtime_name=self.name,
                latency_ms=round((time.time() - start) * 1000, 1),
                detail=f"unreachable: {e}",
            )

    # restart(): inherited no-op — cloud endpoint is stateless per request.
