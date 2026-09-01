"""Registry mapping backend names to Detector factories."""

from __future__ import annotations

from collections.abc import Callable

from edgebench.backends.onnx_cpu import OnnxCpuDetector
from edgebench.backends.onnx_int8 import OnnxInt8Detector
from edgebench.detector import Detector

BACKENDS: dict[str, Callable[..., Detector]] = {
    "onnx_cpu": OnnxCpuDetector,
    "onnx_int8": OnnxInt8Detector,
}
