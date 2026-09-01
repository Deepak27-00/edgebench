"""Registry mapping backend names to Detector factories."""

from __future__ import annotations

from collections.abc import Callable

from edgebench.backends.onnx_cpu import OnnxCpuDetector
from edgebench.detector import Detector

BACKENDS: dict[str, Callable[..., Detector]] = {
    "onnx_cpu": OnnxCpuDetector,
}
