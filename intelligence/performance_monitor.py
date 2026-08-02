"""
intelligence/performance_monitor.py — Performance Monitoring System
Stage 7.6: Track CPU/RAM/latency breakdown to ensure system handles
25-30 concurrent users without overload.

Responsibilities:
- Monitor system resources (CPU, RAM)
- Track per-stage latency breakdown (retrieval, SQL gen, execution)
- Provide a snapshot for the monitoring dashboard / health endpoint
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning(
        "psutil not installed — CPU/RAM monitoring disabled. "
        "Run: pip install psutil"
    )

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    # covers both "not installed" and "no NVIDIA GPU / driver on this box"
    _NVML_AVAILABLE = False
    logger.info(
        "pynvml unavailable or no NVIDIA GPU detected — GPU/VRAM monitoring "
        "disabled. On the RTX 4090 box, run: pip install nvidia-ml-py"
    )


@dataclass
class LatencyBreakdown:
    """Per-stage latency for a single query — Stage 7.6 latency analyzer."""
    retrieval_ms: float = 0.0
    sql_generation_ms: float = 0.0
    db_execution_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "retrieval_ms": round(self.retrieval_ms, 1),
            "sql_generation_ms": round(self.sql_generation_ms, 1),
            "db_execution_ms": round(self.db_execution_ms, 1),
            "validation_ms": round(self.validation_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


@dataclass
class ResourceSnapshot:
    """System resource snapshot at a point in time."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_percent: float = 0.0

    # GPU (Step 7.4 System Load Monitor — GPU Utilization / VRAM Usage)
    gpu_available: bool = False
    gpu_percent: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None
    vram_percent: Optional[float] = None

    # Memory read/write throughput (Step 7.4). Measured as disk I/O
    # throughput via psutil — the standard OS-level proxy for
    # "memory read/write" load requested by Felix; true RAM bus
    # bandwidth isn't exposed by the OS without specialized tooling.
    disk_read_mb_s: Optional[float] = None
    disk_write_mb_s: Optional[float] = None

    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "ram_percent": round(self.ram_percent, 1),
            "ram_used_gb": round(self.ram_used_gb, 2),
            "ram_total_gb": round(self.ram_total_gb, 2),
            "disk_percent": round(self.disk_percent, 1),
            "gpu_available": self.gpu_available,
            "gpu_percent": round(self.gpu_percent, 1) if self.gpu_percent is not None else None,
            "vram_used_mb": round(self.vram_used_mb, 1) if self.vram_used_mb is not None else None,
            "vram_total_mb": round(self.vram_total_mb, 1) if self.vram_total_mb is not None else None,
            "vram_percent": round(self.vram_percent, 1) if self.vram_percent is not None else None,
            "disk_read_mb_s": round(self.disk_read_mb_s, 2) if self.disk_read_mb_s is not None else None,
            "disk_write_mb_s": round(self.disk_write_mb_s, 2) if self.disk_write_mb_s is not None else None,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class InferenceStats:
    """
    Per-LLM-call performance metrics — requested by Felix (AI Consultant):
    TTFT, tokens/second, and input+output token counts.

    Populated from BaseRuntime.chat()'s "timing" + "usage" response fields
    (see runtime/base_runtime.py). Native Ollama timing is exact; private-
    cloud timing is wall-clock-approximate (see `approximate` flag).
    """
    runtime: str = ""
    ttft_ms: Optional[float] = None
    generation_ms: Optional[float] = None
    total_duration_ms: float = 0.0
    tokens_per_second: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    approximate: bool = False
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_tokens(self) -> Optional[int]:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @classmethod
    def from_runtime_response(cls, response: dict) -> "InferenceStats":
        """Build directly from a BaseRuntime.chat() return value."""
        timing = response.get("timing", {}) or {}
        usage = response.get("usage", {}) or {}
        return cls(
            runtime=response.get("runtime", ""),
            ttft_ms=timing.get("ttft_ms"),
            generation_ms=timing.get("generation_ms"),
            total_duration_ms=timing.get("total_duration_ms", 0.0) or 0.0,
            tokens_per_second=timing.get("tokens_per_second"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            approximate=bool(timing.get("approximate", False)),
        )

    def to_dict(self) -> dict:
        return {
            "runtime": self.runtime,
            "ttft_ms": round(self.ttft_ms, 1) if self.ttft_ms is not None else None,
            "generation_ms": round(self.generation_ms, 1) if self.generation_ms is not None else None,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "tokens_per_second": self.tokens_per_second,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "approximate": self.approximate,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceMonitor:
    """
    Stage 7.6 — Performance Monitoring System.

    Used to:
    - Answer Jens's question about 25-30 concurrent user capacity
    - Identify bottlenecks (is it retrieval? LLM? DB?)
    - Feed the monitoring dashboard
    """

    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self._latency_history: list[LatencyBreakdown] = []
        self._resource_history: list[ResourceSnapshot] = []
        self._inference_history: list[InferenceStats] = []
        self._last_disk_io = None       # for computing MB/s delta
        self._last_disk_io_time = None

    # ─── Resource Monitoring ────────────────────────────────────────────────

    def _get_gpu_stats(self) -> dict:
        """GPU utilization + VRAM via pynvml (first GPU, device 0)."""
        if not _NVML_AVAILABLE:
            return {"available": False}
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            vram_used_mb = mem.used / (1024 ** 2)
            vram_total_mb = mem.total / (1024 ** 2)
            return {
                "available": True,
                "gpu_percent": float(util.gpu),
                "vram_used_mb": vram_used_mb,
                "vram_total_mb": vram_total_mb,
                "vram_percent": (
                    100.0 * vram_used_mb / vram_total_mb if vram_total_mb else None
                ),
            }
        except Exception as e:  # noqa: BLE001
            logger.debug("GPU stats read failed: %s", e)
            return {"available": False}

    def _get_disk_io_rate(self) -> tuple:
        """
        Disk read/write throughput in MB/s, computed as a delta between
        this call and the previous one (psutil.disk_io_counters() returns
        a cumulative counter, not an instantaneous rate).
        """
        if not _PSUTIL_AVAILABLE:
            return None, None
        try:
            io = psutil.disk_io_counters()
            now = time.time()
            if io is None:
                return None, None
            if self._last_disk_io is None:
                self._last_disk_io = io
                self._last_disk_io_time = now
                return None, None  # no delta on first call

            dt = now - self._last_disk_io_time
            if dt <= 0:
                return None, None
            read_mb_s = (io.read_bytes - self._last_disk_io.read_bytes) / (1024 ** 2) / dt
            write_mb_s = (io.write_bytes - self._last_disk_io.write_bytes) / (1024 ** 2) / dt
            self._last_disk_io = io
            self._last_disk_io_time = now
            return max(0.0, read_mb_s), max(0.0, write_mb_s)
        except Exception as e:  # noqa: BLE001
            logger.debug("Disk I/O read failed: %s", e)
            return None, None

    def get_resource_snapshot(self) -> ResourceSnapshot:
        """Get current CPU/RAM/disk/GPU/VRAM usage."""
        if not _PSUTIL_AVAILABLE:
            return ResourceSnapshot()

        try:
            cpu = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            gpu = self._get_gpu_stats()
            read_mb_s, write_mb_s = self._get_disk_io_rate()

            snapshot = ResourceSnapshot(
                cpu_percent=cpu,
                ram_percent=mem.percent,
                ram_used_gb=mem.used / (1024 ** 3),
                ram_total_gb=mem.total / (1024 ** 3),
                disk_percent=disk.percent,
                gpu_available=gpu.get("available", False),
                gpu_percent=gpu.get("gpu_percent"),
                vram_used_mb=gpu.get("vram_used_mb"),
                vram_total_mb=gpu.get("vram_total_mb"),
                vram_percent=gpu.get("vram_percent"),
                disk_read_mb_s=read_mb_s,
                disk_write_mb_s=write_mb_s,
            )

            self._resource_history.append(snapshot)
            if len(self._resource_history) > self.history_size:
                self._resource_history.pop(0)

            return snapshot

        except Exception as e:
            logger.debug(f"Resource snapshot failed: {e}")
            return ResourceSnapshot()

    def is_under_pressure(self) -> tuple[bool, str]:
        """
        Check if system is under resource pressure.
        Used to decide whether to queue/reject new requests.
        """
        snapshot = self.get_resource_snapshot()

        if snapshot.cpu_percent > 90:
            return True, f"CPU usage critical: {snapshot.cpu_percent:.0f}%"
        if snapshot.ram_percent > 90:
            return True, f"RAM usage critical: {snapshot.ram_percent:.0f}%"

        return False, ""

    # ─── Latency Tracking ───────────────────────────────────────────────────

    def record_latency(self, breakdown: LatencyBreakdown):
        """Record a completed query's latency breakdown."""
        self._latency_history.append(breakdown)
        if len(self._latency_history) > self.history_size:
            self._latency_history.pop(0)

    def get_latency_stats(self) -> dict:
        """
        Get aggregate latency statistics — identifies bottlenecks.
        Targets from Stage 8.5: retrieval <500ms, SQL gen <2s, full <5s.
        """
        if not self._latency_history:
            return {
                "sample_count": 0,
                "avg_retrieval_ms": 0.0,
                "avg_sql_generation_ms": 0.0,
                "avg_db_execution_ms": 0.0,
                "avg_total_ms": 0.0,
                "bottleneck": "no data yet",
            }

        n = len(self._latency_history)
        avg_retrieval = sum(
            l.retrieval_ms for l in self._latency_history
        ) / n
        avg_sql_gen = sum(
            l.sql_generation_ms for l in self._latency_history
        ) / n
        avg_db_exec = sum(
            l.db_execution_ms for l in self._latency_history
        ) / n
        avg_validation = sum(
            l.validation_ms for l in self._latency_history
        ) / n
        avg_total = sum(
            l.total_ms for l in self._latency_history
        ) / n

        # Identify the dominant bottleneck stage
        stages = {
            "retrieval": avg_retrieval,
            "sql_generation (LLM)": avg_sql_gen,
            "db_execution": avg_db_exec,
            "validation": avg_validation,
        }
        bottleneck = max(stages, key=stages.get)

        return {
            "sample_count": n,
            "avg_retrieval_ms": round(avg_retrieval, 1),
            "avg_sql_generation_ms": round(avg_sql_gen, 1),
            "avg_db_execution_ms": round(avg_db_exec, 1),
            "avg_validation_ms": round(avg_validation, 1),
            "avg_total_ms": round(avg_total, 1),
            "bottleneck": bottleneck,
            "meets_target": avg_total < 5000,  # Stage 8.5 target: <5s
        }

    # ─── Inference Stats Tracking (TTFT, tokens/sec — requested by Felix) ───

    def record_inference(self, stats: InferenceStats):
        """Record one LLM call's performance metrics."""
        self._inference_history.append(stats)
        if len(self._inference_history) > self.history_size:
            self._inference_history.pop(0)

    def record_from_runtime_response(self, response: dict) -> InferenceStats:
        """
        Convenience: build InferenceStats directly from a
        BaseRuntime.chat() return value and record it in one call.
        Call sites (query_service.py, schema_service.py) use this right
        after get_runtime().chat(...).
        """
        stats = InferenceStats.from_runtime_response(response)
        self.record_inference(stats)
        return stats

    def get_inference_stats(self) -> dict:
        """
        Aggregate inference performance — the exact numbers Felix asked
        for: average TTFT, tokens/second, and input+output tokens per
        request, over the current history window.
        """
        hist = self._inference_history
        if not hist:
            return {
                "sample_count": 0,
                "avg_ttft_ms": None,
                "avg_tokens_per_second": None,
                "avg_input_tokens": None,
                "avg_output_tokens": None,
                "avg_total_tokens": None,
                "any_approximate": False,
            }

        def _avg(vals):
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        return {
            "sample_count": len(hist),
            "avg_ttft_ms": _avg([s.ttft_ms for s in hist]),
            "avg_tokens_per_second": _avg([s.tokens_per_second for s in hist]),
            "avg_input_tokens": _avg([s.input_tokens for s in hist]),
            "avg_output_tokens": _avg([s.output_tokens for s in hist]),
            "avg_total_tokens": _avg([s.total_tokens for s in hist]),
            "avg_total_duration_ms": _avg([s.total_duration_ms for s in hist]),
            "any_approximate": any(s.approximate for s in hist),
            "runtime_mix": {
                r: sum(1 for s in hist if s.runtime == r)
                for r in {s.runtime for s in hist}
            },
        }

    # ─── Combined Snapshot (for monitoring dashboard / health endpoint) ────

    def get_full_snapshot(self) -> dict:
        """Combined resource + latency + inference snapshot for the dashboard."""
        return {
            "resources": self.get_resource_snapshot().to_dict(),
            "latency": self.get_latency_stats(),
            "inference": self.get_inference_stats(),
            "psutil_available": _PSUTIL_AVAILABLE,
            "nvml_available": _NVML_AVAILABLE,
        }

    # ─── Performance Stats Export API (Step 7.4) ────────────────────────────
    # Consumers: Felix's AI Consultant dashboard, Performance Monitoring
    # Dashboard, Enterprise Benchmark Reports. Exposed over HTTP via
    # api/routes/stats.py (GET /stats, GET /stats/export).

    def export_stats(self) -> dict:
        """Full JSON export: current snapshot + raw inference history."""
        return {
            "snapshot": self.get_full_snapshot(),
            "inference_history": [s.to_dict() for s in self._inference_history],
            "exported_at": datetime.now().isoformat(),
        }

    def export_stats_csv_rows(self) -> list:
        """
        Flat rows for CSV export of the inference history — one row per
        recorded LLM call. Pair with core/csv writer or return directly
        from a FastAPI StreamingResponse.
        """
        rows = []
        for s in self._inference_history:
            d = s.to_dict()
            rows.append(d)
        return rows


class LatencyTimer:
    """
    Context-manager-friendly stopwatch for tracking pipeline stage latency.

    Usage:
        timer = LatencyTimer()
        timer.start("retrieval")
        ... do retrieval ...
        timer.stop("retrieval")
        timer.start("sql_generation")
        ... call LLM ...
        timer.stop("sql_generation")
        breakdown = timer.to_breakdown()
    """

    def __init__(self):
        self._starts: dict[str, float] = {}
        self._durations: dict[str, float] = {}
        self._overall_start = time.time()

    def start(self, stage: str):
        self._starts[stage] = time.time()

    def stop(self, stage: str):
        if stage in self._starts:
            elapsed = (time.time() - self._starts[stage]) * 1000
            self._durations[stage] = self._durations.get(stage, 0) + elapsed

    def to_breakdown(self) -> LatencyBreakdown:
        total_ms = (time.time() - self._overall_start) * 1000
        return LatencyBreakdown(
            retrieval_ms=self._durations.get("retrieval", 0.0),
            sql_generation_ms=self._durations.get("sql_generation", 0.0),
            db_execution_ms=self._durations.get("db_execution", 0.0),
            validation_ms=self._durations.get("validation", 0.0),
            total_ms=total_ms,
        )


# ─── Singleton ────────────────────────────────────────────────────────────────
_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor
