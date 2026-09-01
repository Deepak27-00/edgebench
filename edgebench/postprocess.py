"""Decode raw YOLO ONNX output into Detection objects.

Handles both detection head layouts behind one entry point, ``decode()``:
a one-to-many head (YOLO11, ``(1, 84, 8400)``) needing anchor decoding and
NMS, and a one-to-one / NMS-free head (YOLO26, ``(1, 300, 6)``) needing only
a confidence filter and un-letterbox.
"""

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
    """Decode a raw YOLO ONNX output tensor into Detections.

    Branches on the output's last dimension: 6 means a one-to-one (NMS-free)
    head — rows are already ``(x1, y1, x2, y2, score, class_id)`` — so it goes
    to ``decode_end2end``; anything else is treated as a one-to-many head and
    goes through anchor decoding plus class-wise NMS below.
    """
    if output.shape[-1] == 6:
        return decode_end2end(output, scale, pad_x, pad_y, conf_thres=conf_thres)
    return _decode_anchors(output, scale, pad_x, pad_y, conf_thres=conf_thres, iou_thres=iou_thres)


def decode_end2end(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    *,
    conf_thres: float = 0.25,
) -> list[Detection]:
    """Decode a ``(1, 300, 6)`` NMS-free YOLO output into Detections.

    Each of the up to 300 rows is ``(x1, y1, x2, y2, score, class_id)`` in
    letterboxed image space, with NMS already applied inside the graph.
    Postprocessing is just a confidence filter and a coordinate un-letterbox
    — no anchor decode, no NMS.
    """
    predictions = output[0]  # (300, 6)
    scores = predictions[:, 4]

    keep = scores > conf_thres
    if not np.any(keep):
        return []

    predictions = predictions[keep]
    boxes_xyxy = unletterbox_boxes(predictions[:, :4].copy(), scale, pad_x, pad_y)
    scores = predictions[:, 4]
    class_ids = predictions[:, 5].astype(int)

    return [
        Detection(
            x1=float(x1),
            y1=float(y1),
            x2=float(x2),
            y2=float(y2),
            score=float(score),
            class_id=int(class_id),
            class_name=COCO_CLASSES[int(class_id)],
        )
        for (x1, y1, x2, y2), score, class_id in zip(boxes_xyxy, scores, class_ids, strict=True)
    ]


def _decode_anchors(
    output: np.ndarray,
    scale: float,
    pad_x: float,
    pad_y: float,
    *,
    conf_thres: float,
    iou_thres: float,
) -> list[Detection]:
    """Decode a ``(1, 84, 8400)`` one-to-many YOLO output tensor into Detections.

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
