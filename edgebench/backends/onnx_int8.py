"""ONNX Runtime CPU backend for a dynamically INT8-quantised model.

Reuses OnnxCpuDetector's session and inference logic unchanged — the only
difference from ``onnx_cpu`` is which ``.onnx`` file gets loaded.
"""

from __future__ import annotations

from pathlib import Path

from edgebench.backends.onnx_cpu import OnnxCpuDetector

DEFAULT_INT8_MODEL_PATH = str(Path(__file__).resolve().parents[2] / "models" / "yolo11n.int8.onnx")


class OnnxInt8Detector(OnnxCpuDetector):
    """OnnxCpuDetector defaulted to the dynamically-quantised INT8 model."""

    def __init__(self, model_path: str = DEFAULT_INT8_MODEL_PATH, **kwargs) -> None:
        """Same constructor as OnnxCpuDetector; defaults to the INT8 model path."""
        super().__init__(model_path, **kwargs)
