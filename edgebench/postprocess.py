"""Decode raw YOLO11 ONNX output into Detection objects."""

from __future__ import annotations

import cv2
import numpy as np

from edgebench.detector import COCO_CLASSES, Detection
from edgebench.preprocess import unletterbox_boxes


def decode(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    *,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
) -> list[Detection]:
    """Decode a ``(1, 84, 8400)`` YOLO11 output tensor into Detections.

    Transposes to ``(8400, 84)``, splits into box ``(cx, cy, w, h)`` and 80
    class scores, takes the max-scoring class per row, filters by
    ``conf_thres``, converts to xyxy, un-letterboxes to original image
    coordinates, then applies **class-wise** NMS at ``iou_thres`` — a global
    NMS would suppress e.g. a person standing next to a car.
    """
    predictions = output[0].T  # (8400, 84)
    boxes_cxcywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(len(class_scores)), class_ids]

    keep = scores > conf_thres
    if not np.any(keep):
        return []

    boxes_cxcywh = boxes_cxcywh[keep]
    scores = scores[keep]
    class_ids = class_ids[keep]

    boxes_xyxy = _cxcywh_to_xyxy(boxes_cxcywh)
    boxes_xyxy = unletterbox_boxes(boxes_xyxy, scale, pad_x, pad_y)

    detections: list[Detection] = []
    for class_id in np.unique(class_ids):
        mask = class_ids == class_id
        cls_boxes = boxes_xyxy[mask]
        cls_scores = scores[mask]

        nms_boxes = _xyxy_to_xywh(cls_boxes).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, cls_scores.tolist(), conf_thres, iou_thres)

        for i in np.array(indices).flatten():
            x1, y1, x2, y2 = cls_boxes[i]
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=float(cls_scores[i]),
                    class_id=int(class_id),
                    class_name=COCO_CLASSES[int(class_id)],
                )
            )
    return detections


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(cx, cy, w, h)`` boxes to ``(x1, y1, x2, y2)``."""
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xyxy


def _xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` boxes to ``(x, y, w, h)`` for ``cv2.dnn.NMSBoxes``."""
    xywh = np.empty_like(boxes)
    xywh[:, 0] = boxes[:, 0]
    xywh[:, 1] = boxes[:, 1]
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh
