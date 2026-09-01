"""CLI: sweep the benchmark matrix and write results/<timestamp>.json + results/latest.json.

Input size is fixed at 640 — the exported models have a static 640x640 input, so a
320 sweep would need its own export and is not yet implemented (see SPEC.md M3).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import cv2

# Let `python scripts/run_bench.py` find the edgebench package at the repo root —
# Python only puts this script's own directory (scripts/) on sys.path by default.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edgebench.backends import BACKENDS
from edgebench.bench.harness import benchmark

MODELS = ["yolo11n", "yolo26n"]
BACKEND_NAMES = ["onnx_cpu", "onnx_int8"]
GRAPH_OPT_LEVELS = ["all", "basic", "disabled"]
INPUT_SIZE = 640

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
SAMPLE_IMAGE = ROOT / "assets" / "sample.jpg"


def _intra_thread_values() -> list[int]:
    """1, 2, 4, and os.cpu_count(), deduplicated and capped at the machine's core count."""
    cpu_count = os.cpu_count() or 1
    return sorted({t for t in (1, 2, 4, cpu_count) if t <= cpu_count})


def _model_path(model: str, backend: str) -> Path:
    suffix = ".int8.onnx" if backend == "onnx_int8" else ".onnx"
    return MODELS_DIR / f"{model}{suffix}"


def run_matrix(*, warmup: int, iters: int) -> dict:
    """Sweep model x backend x graph-opt x thread-count and return the results payload."""
    image = cv2.imread(str(SAMPLE_IMAGE))
    if image is None:
        raise FileNotFoundError(f"sample image not found: {SAMPLE_IMAGE}")

    runs = []
    for model in MODELS:
        for backend_name in BACKEND_NAMES:
            model_path = _model_path(model, backend_name)
            if not model_path.exists():
                print(f"skipping {backend_name}/{model}: {model_path} not found")
                continue

            for graph_opt_level in GRAPH_OPT_LEVELS:
                for intra_op_threads in _intra_thread_values():
                    factory = BACKENDS[backend_name]
                    detector = factory(
                        str(model_path),
                        intra_op_threads=intra_op_threads,
                        graph_opt_level=graph_opt_level,
                    )
                    result = benchmark(detector, image, warmup=warmup, iters=iters)
                    runs.append(
                        {
                            "model": model,
                            "backend": backend_name,
                            "graph_opt_level": graph_opt_level,
                            "intra_op_threads": intra_op_threads,
                            "input_size": INPUT_SIZE,
                            **asdict(result),
                        }
                    )
                    print(
                        f"{backend_name:10s} {model:10s} opt={graph_opt_level:8s} "
                        f"threads={intra_op_threads:2d}  p50={result.latency.p50_ms:7.2f}ms  "
                        f"fps={result.latency.fps:6.1f}  cpu={result.resources.mean_cpu_percent:5.1f}%"
                    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "warmup": warmup,
        "iters": iters,
        "runs": runs,
    }


def main() -> None:
    """Parse CLI args, run the matrix, and write timestamped + latest result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    payload = run_matrix(warmup=args.warmup, iters=args.iters)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    timestamped_path = RESULTS_DIR / f"{timestamp}.json"
    latest_path = RESULTS_DIR / "latest.json"

    timestamped_path.write_text(json.dumps(payload, indent=2))
    shutil.copyfile(timestamped_path, latest_path)
    print(f"wrote {timestamped_path} and {latest_path}")


if __name__ == "__main__":
    main()
