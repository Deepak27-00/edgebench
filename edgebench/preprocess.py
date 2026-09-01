"""Letterbox resize and normalisation for YOLO ONNX input, and its inverse."""

from __future__ import annotations

import cv2
import numpy as np


def letterbox(
    image: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, float, float]:
    """Letterbox-resize a BGR uint8 image and prepare it for ONNX inference.

    Resizes preserving aspect ratio, pads with ``color`` to ``new_shape``,
    converts BGR to RGB, scales to [0, 1], reorders HWC to CHW, and adds a
    batch dimension.

    Returns ``(blob, scale, pad_x, pad_y)``: ``blob`` is a float32 array of
    shape ``(1, 3, *new_shape)``, and ``(scale, pad_x, pad_y)`` are what
    ``unletterbox_boxes`` needs to map detections back to the original image.
    """
    h0, w0 = image.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w0, new_h / h0)
    resized_w, resized_h = round(w0 * scale), round(h0 * scale)
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = new_w - resized_w, new_h - resized_h
    pad_x, pad_y = pad_w / 2, pad_h / 2
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    rgb = padded[:, :, ::-1]
    normalised = rgb.astype(np.float32) / 255.0
    chw = normalised.transpose(2, 0, 1)
    blob = np.ascontiguousarray(chw[np.newaxis, ...])

    return blob, scale, pad_x, pad_y


def unletterbox_boxes(boxes: np.ndarray, scale: float, pad_x: float, pad_y: float) -> np.ndarray:
    """Map xyxy boxes from letterboxed image space back to the original image."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale
    return boxes
