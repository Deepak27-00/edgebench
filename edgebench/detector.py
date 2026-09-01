"""Core detection types: the Detection result, the Detector protocol every
backend implements, and the COCO class names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object, in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    class_name: str


class StageTimings(TypedDict):
    """Per-stage wall time, in milliseconds, for the most recent ``infer`` call."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class Detector(Protocol):
    """Interface every inference backend implements.

    Adding a backend must never require changes to the API layer or the
    benchmark harness — both depend only on this protocol.
    """

    def warmup(self, n: int) -> None:
        """Run ``n`` throwaway inferences to trigger lazy graph init and allocator warm-up."""
        ...

    def infer(self, img: np.ndarray) -> list[Detection]:
        """Run detection on a BGR uint8 HWC image and return the detections found."""
        ...

    @property
    def stage_timings(self) -> StageTimings:
        """Per-stage timings, in milliseconds, for the most recent ``infer`` call."""
        ...


COCO_CLASSES: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
