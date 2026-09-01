"""ONNX Runtime CPU backend for detection, with per-stage timing."""

from __future__ import annotations

import time

import numpy as np
import onnxruntime as ort

from edgebench.detector import Detection, StageTimings
from edgebench.postprocess import decode
from edgebench.preprocess import letterbox

_GRAPH_OPT_LEVELS = {
    "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


class OnnxCpuDetector:
    """Detector backend running a YOLO ONNX model via onnxruntime on CPU."""

    def __init__(
        self,
        model_path: str,
        *,
        intra_op_threads: int | None = None,
        inter_op_threads: int = 1,
        graph_opt_level: str = "all",
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> None:
        """Build a session from the given threading and graph-optimisation config.

        ``intra_op_threads``/``inter_op_threads``/``graph_opt_level`` are exposed
        as constructor args, not hardcoded, so the benchmark harness can sweep them.
        """
        options = ort.SessionOptions()
        if intra_op_threads is not None:
            options.intra_op_num_threads = intra_op_threads
        options.inter_op_num_threads = inter_op_threads
        options.graph_optimization_level = _GRAPH_OPT_LEVELS[graph_opt_level]

        self._session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._conf_thres = conf_thres
        self._iou_thres = iou_thres
        self._stage_timings: StageTimings = {
            "preprocess_ms": 0.0,
            "inference_ms": 0.0,
            "postprocess_ms": 0.0,
        }

        self.model_path = model_path
        self.graph_opt_level = graph_opt_level
        self.intra_op_threads = intra_op_threads

    def warmup(self, n: int) -> None:
        """Run ``n`` throwaway inferences on a blank frame to warm up the session."""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(n):
            self.infer(dummy)

    def infer(self, img: np.ndarray, *, conf_thres: float | None = None) -> list[Detection]:
        """Run detection on a BGR uint8 HWC image and return the detections found.

        ``conf_thres`` overrides the instance default for this call only, if given.
        """
        t0 = time.perf_counter()
        blob, scale, pad_x, pad_y = letterbox(img)
        t1 = time.perf_counter()

        output = self._session.run(None, {self._input_name: blob})[0]
        t2 = time.perf_counter()

        detections = decode(
            output,
            scale,
            pad_x,
            pad_y,
            conf_thres=conf_thres if conf_thres is not None else self._conf_thres,
            iou_thres=self._iou_thres,
        )
        t3 = time.perf_counter()

        self._stage_timings = {
            "preprocess_ms": (t1 - t0) * 1000,
            "inference_ms": (t2 - t1) * 1000,
            "postprocess_ms": (t3 - t2) * 1000,
        }
        return detections

    @property
    def stage_timings(self) -> StageTimings:
        """Per-stage timings, in milliseconds, for the most recent ``infer`` call."""
        return self._stage_timings

    @property
    def execution_provider(self) -> str:
        """Active onnxruntime execution provider for this session."""
        providers = self._session.get_providers()
        return providers[0] if providers else "unknown"
