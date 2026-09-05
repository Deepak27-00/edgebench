"""FastAPI app: detection endpoint, backend/benchmark metadata, static UI.

Model loading happens once at startup via the lifespan handler below, not per
request — the served app depends on onnxruntime only, never ultralytics/torch.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas import (
    BackendInfo,
    BackendsResponse,
    DetectionOut,
    DetectResponse,
    HealthResponse,
    StageTimingsOut,
)
from edgebench.backends import BACKENDS
from edgebench.detector import Detector

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "results" / "latest.json"
NOTICE_PATH = ROOT / "THIRD_PARTY_NOTICES.md"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_BACKEND = "onnx_cpu"
WARMUP_ITERS = 5


class BackendConfig(NamedTuple):
    """The model and runner presented to users as one selectable backend."""

    factory_name: str
    model_path: Path
    model: str
    label: str


BACKEND_CONFIGS = {
    "onnx_cpu": BackendConfig(
        "onnx_cpu", MODELS_DIR / "yolo11n.onnx", "yolo11n", "YOLO11n · ONNX CPU (FP32)"
    ),
    "onnx_int8": BackendConfig(
        "onnx_int8", MODELS_DIR / "yolo11n.int8.onnx", "yolo11n", "YOLO11n · ONNX CPU (INT8)"
    ),
    "onnx_cpu_yolo26n": BackendConfig(
        "onnx_cpu", MODELS_DIR / "yolo26n.onnx", "yolo26n", "YOLO26n · ONNX CPU (FP32)"
    ),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and warm up one detector per available backend, once, at startup.

    Warming up here (not on first request) keeps the first real user request
    from eating the ~2s lazy graph-init/allocator cost as a latency outlier.
    """
    detectors: dict[str, Detector] = {}
    for name, config in BACKEND_CONFIGS.items():
        if not config.model_path.exists():
            continue
        factory = BACKENDS[config.factory_name]
        detector = factory(str(config.model_path))
        detector.warmup(WARMUP_ITERS)
        detectors[name] = detector

    app.state.detectors = detectors
    yield
    app.state.detectors = {}


app = FastAPI(title="EdgeBench", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    """Liveness check: overall status plus per-backend readiness."""
    detectors: dict[str, Detector] = app.state.detectors
    ready = {name: name in detectors for name in BACKEND_CONFIGS}
    status = "ok" if detectors else "degraded"
    return HealthResponse(status=status, backends=ready)


@app.get("/api/backends", response_model=BackendsResponse)
def get_backends() -> BackendsResponse:
    """List the backends loaded at startup and their configuration."""
    backends = []
    for name, detector in app.state.detectors.items():
        config = BACKEND_CONFIGS[name]
        backends.append(
            BackendInfo(
                name=name,
                model=config.model,
                label=config.label,
                model_path=str(detector.model_path),
                graph_opt_level=detector.graph_opt_level,
                intra_op_threads=detector.intra_op_threads,
                execution_provider=detector.execution_provider,
            )
        )
    return BackendsResponse(backends=backends, default=DEFAULT_BACKEND)


@app.get("/api/benchmark")
def get_benchmark() -> dict[str, Any]:
    """Serve the committed results/latest.json benchmark results."""
    if not RESULTS_PATH.exists():
        raise HTTPException(status_code=404, detail="no benchmark results yet")
    return json.loads(RESULTS_PATH.read_text())


@app.get("/NOTICE")
def get_third_party_notices() -> FileResponse:
    """Serve the model attribution and AGPL source-availability notice."""
    return FileResponse(NOTICE_PATH, media_type="text/markdown; charset=utf-8")


@app.post("/api/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile,
    backend: str = Query(DEFAULT_BACKEND),
    conf: float | None = Query(None, ge=0.0, le=1.0),
) -> DetectResponse:
    """Run detection on an uploaded image with the given backend and confidence threshold."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file exceeds 10 MB limit")

    detector = app.state.detectors.get(backend)
    if detector is None:
        raise HTTPException(status_code=400, detail=f"unknown backend: {backend}")

    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="could not decode image")

    detections = detector.infer(image, conf_thres=conf)
    timings = detector.stage_timings

    return DetectResponse(
        backend=backend,
        detections=[DetectionOut(**vars(d)) for d in detections],
        stage_timings=StageTimingsOut(**timings),
    )


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
