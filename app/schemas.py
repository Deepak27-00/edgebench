"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel


class DetectionOut(BaseModel):
    """One detected object, in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class StageTimingsOut(BaseModel):
    """Per-stage wall time, in milliseconds, for one detect call."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class DetectResponse(BaseModel):
    """Response body for POST /api/detect."""

    backend: str
    detections: list[DetectionOut]
    stage_timings: StageTimingsOut


class BackendInfo(BaseModel):
    """One backend's configuration, as loaded at startup."""

    name: str
    model: str
    label: str
    model_path: str
    graph_opt_level: str
    intra_op_threads: int | None
    execution_provider: str


class BackendsResponse(BaseModel):
    """Response body for GET /api/backends."""

    backends: list[BackendInfo]
    default: str


class HealthResponse(BaseModel):
    """Response body for GET /healthz."""

    status: str
    backends: dict[str, bool]
