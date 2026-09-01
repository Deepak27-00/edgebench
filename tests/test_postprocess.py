import numpy as np
import pytest

from edgebench.detector import COCO_CLASSES
from edgebench.postprocess import decode, decode_end2end


def test_decode_two_peaks_returns_two_boxes_at_expected_coordinates():
    output = np.zeros((1, 84, 8400), dtype=np.float32)

    # peak 0: cx=100, cy=100, w=50, h=50 -> xyxy (75, 75, 125, 125), class 0
    output[0, 0, 0] = 100.0
    output[0, 1, 0] = 100.0
    output[0, 2, 0] = 50.0
    output[0, 3, 0] = 50.0
    output[0, 4 + 0, 0] = 0.9

    # peak 1: cx=300, cy=300, w=60, h=80 -> xyxy (270, 260, 330, 340), class 5
    output[0, 0, 1] = 300.0
    output[0, 1, 1] = 300.0
    output[0, 2, 1] = 60.0
    output[0, 3, 1] = 80.0
    output[0, 4 + 5, 1] = 0.8

    detections = decode(output, scale=1.0, pad_x=0.0, pad_y=0.0)

    assert len(detections) == 2
    by_class = {d.class_id: d for d in detections}

    det0 = by_class[0]
    assert det0.class_name == COCO_CLASSES[0]
    assert (det0.x1, det0.y1, det0.x2, det0.y2) == pytest.approx((75.0, 75.0, 125.0, 125.0))
    assert det0.score == pytest.approx(0.9)

    det5 = by_class[5]
    assert det5.class_name == COCO_CLASSES[5]
    assert (det5.x1, det5.y1, det5.x2, det5.y2) == pytest.approx((270.0, 260.0, 330.0, 340.0))
    assert det5.score == pytest.approx(0.8)


def test_decode_filters_below_conf_thres():
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0, 0] = 100.0
    output[0, 1, 0] = 100.0
    output[0, 2, 0] = 50.0
    output[0, 3, 0] = 50.0
    output[0, 4 + 0, 0] = 0.1  # below default conf_thres=0.25

    assert decode(output, scale=1.0, pad_x=0.0, pad_y=0.0) == []


def test_decode_end2end_two_rows_returns_two_boxes_at_expected_coordinates():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [75.0, 75.0, 125.0, 125.0, 0.9, 0]  # already xyxy, class 0
    output[0, 1] = [270.0, 260.0, 330.0, 340.0, 0.8, 5]  # already xyxy, class 5
    # remaining 298 rows are all-zero padding: score 0.0 < conf_thres, filtered out

    detections = decode_end2end(output, scale=1.0, pad_x=0.0, pad_y=0.0)

    assert len(detections) == 2
    by_class = {d.class_id: d for d in detections}

    det0 = by_class[0]
    assert det0.class_name == COCO_CLASSES[0]
    assert (det0.x1, det0.y1, det0.x2, det0.y2) == pytest.approx((75.0, 75.0, 125.0, 125.0))
    assert det0.score == pytest.approx(0.9)

    det5 = by_class[5]
    assert det5.class_name == COCO_CLASSES[5]
    assert (det5.x1, det5.y1, det5.x2, det5.y2) == pytest.approx((270.0, 260.0, 330.0, 340.0))
    assert det5.score == pytest.approx(0.8)


def test_decode_dispatches_to_end2end_on_six_column_output():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [75.0, 75.0, 125.0, 125.0, 0.9, 0]

    assert decode(output, scale=1.0, pad_x=0.0, pad_y=0.0) == decode_end2end(
        output, scale=1.0, pad_x=0.0, pad_y=0.0
    )
