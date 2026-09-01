"""Benchmark harness: warmup discard, percentile latency, per-stage aggregation,
background-thread resource sampling, and environment capture.
"""

from __future__ import annotations

import platform
import statistics
import threading
import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import psutil

from edgebench.detector import Detector

SAMPLE_HZ = 20


@dataclass
class LatencyStats:
    """Percentile and summary latency, in milliseconds, plus throughput in FPS."""

    p50_ms: float
    p90_ms: float
    p99_ms: float
    mean_ms: float
    stdev_ms: float
    min_ms: float
    max_ms: float
    fps: float


@dataclass
class StageBreakdown:
    """Mean per-stage wall time, in milliseconds, across the timed window."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass
class ResourceUsage:
    """Process resource usage sampled across the timed window."""

    mean_cpu_percent: float
    peak_rss_mb: float


@dataclass
class Environment:
    """Machine and runtime context needed to reproduce a result."""

    cpu_model: str
    physical_cores: int
    logical_cores: int
    os: str
    python_version: str
    onnxruntime_version: str
    execution_provider: str


@dataclass
class BenchResult:
    """Full output of one ``benchmark()`` call."""

    latency: LatencyStats
    stage_breakdown: StageBreakdown
    resources: ResourceUsage
    environment: Environment
    warmup: int
    iters: int


class _ResourceSampler:
    """Samples process CPU% and RSS on a background thread at ~SAMPLE_HZ."""

    def __init__(self, process: psutil.Process, hz: int = SAMPLE_HZ) -> None:
        self._process = process
        self._interval = 1.0 / hz
        self._cpu_samples: list[float] = []
        self._peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """Start sampling. Primes psutil's CPU% counter first, since its first
        reading (a comparison against process start) is meaningless."""
        self._process.cpu_percent()
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        """Stop sampling and return ``(mean_cpu_percent, peak_rss_mb)``."""
        self._stop.set()
        self._thread.join()
        mean_cpu = statistics.mean(self._cpu_samples) if self._cpu_samples else 0.0
        peak_rss_mb = self._peak_rss_bytes / (1024 * 1024)
        return mean_cpu, peak_rss_mb

    def _run(self) -> None:
        while not self._stop.is_set():
            self._cpu_samples.append(self._process.cpu_percent())
            self._peak_rss_bytes = max(self._peak_rss_bytes, self._process.memory_info().rss)
            time.sleep(self._interval)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * (pct / 100)
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (k - lower)


def _capture_environment(detector: Detector) -> Environment:
    """Record machine and runtime context. ``execution_provider`` falls back to
    "unknown" for backends (e.g. a future torch reference) that don't expose one."""
    return Environment(
        cpu_model=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False) or 0,
        logical_cores=psutil.cpu_count(logical=True) or 0,
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        onnxruntime_version=ort.__version__,
        execution_provider=getattr(detector, "execution_provider", "unknown"),
    )


def benchmark(
    detector: Detector, image: np.ndarray, *, warmup: int = 10, iters: int = 100
) -> BenchResult:
    """Benchmark ``detector`` on ``image``.

    Discards ``warmup`` iterations before timing — first-call cost includes lazy
    graph init and allocator warm-up, and including it is the single most common
    benchmarking error. Times the remaining ``iters`` calls with
    ``time.perf_counter_ns``, aggregates the per-stage breakdown, samples process
    CPU%/RSS on a background thread across the timed window, and captures the
    environment.
    """
    detector.warmup(warmup)

    process = psutil.Process()
    sampler = _ResourceSampler(process)

    latencies_ms: list[float] = []
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    postprocess_ms: list[float] = []

    sampler.start()
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        detector.infer(image)
        t1 = time.perf_counter_ns()

        latencies_ms.append((t1 - t0) / 1e6)
        stages = detector.stage_timings
        preprocess_ms.append(stages["preprocess_ms"])
        inference_ms.append(stages["inference_ms"])
        postprocess_ms.append(stages["postprocess_ms"])
    mean_cpu_percent, peak_rss_mb = sampler.stop()

    sorted_latencies = sorted(latencies_ms)
    p50 = _percentile(sorted_latencies, 50)

    latency = LatencyStats(
        p50_ms=p50,
        p90_ms=_percentile(sorted_latencies, 90),
        p99_ms=_percentile(sorted_latencies, 99),
        mean_ms=statistics.mean(latencies_ms),
        stdev_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        fps=1000.0 / p50 if p50 > 0 else 0.0,
    )
    stage_breakdown = StageBreakdown(
        preprocess_ms=statistics.mean(preprocess_ms),
        inference_ms=statistics.mean(inference_ms),
        postprocess_ms=statistics.mean(postprocess_ms),
    )
    resources = ResourceUsage(mean_cpu_percent=mean_cpu_percent, peak_rss_mb=peak_rss_mb)

    return BenchResult(
        latency=latency,
        stage_breakdown=stage_breakdown,
        resources=resources,
        environment=_capture_environment(detector),
        warmup=warmup,
        iters=iters,
    )
