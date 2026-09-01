import numpy as np
import pytest

from edgebench.preprocess import letterbox, unletterbox_boxes


def test_letterbox_output_shape_dtype_range():
    image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    blob, _scale, _pad_x, _pad_y = letterbox(image, new_shape=(640, 640))

    assert blob.shape == (1, 3, 640, 640)
    assert blob.dtype == np.float32
    assert blob.min() >= 0.0
    assert blob.max() <= 1.0


def test_letterbox_pads_shorter_axis_only():
    # 4:3 image into a square target: width already fills the frame, height gets padded.
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _blob, scale, pad_x, pad_y = letterbox(image, new_shape=(640, 640))

    assert scale == pytest.approx(1.0)
    assert pad_x == pytest.approx(0.0, abs=1.0)
    assert pad_y == pytest.approx(80.0, abs=1.0)


def test_unletterbox_boxes_round_trip():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    _blob, scale, pad_x, pad_y = letterbox(image, new_shape=(640, 640))

    original_box = np.array([[100.0, 50.0, 300.0, 200.0]])
    letterboxed_box = original_box * scale + np.array([pad_x, pad_y, pad_x, pad_y])

    recovered = unletterbox_boxes(letterboxed_box.copy(), scale, pad_x, pad_y)

    assert recovered == pytest.approx(original_box, abs=1e-4)
