from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

SAMPLE_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "sample.jpg"


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["backends"]["onnx_cpu"] is True
    assert body["backends"]["onnx_cpu_yolo26n"] is True


def test_backends_include_yolo26n(client):
    response = client.get("/api/backends")

    assert response.status_code == 200
    body = response.json()
    yolo26n = next(backend for backend in body["backends"] if backend["model"] == "yolo26n")
    assert yolo26n["name"] == "onnx_cpu_yolo26n"
    assert yolo26n["label"] == "YOLO26n · ONNX CPU (FP32)"


def test_third_party_notice_is_available(client):
    response = client.get("/NOTICE")

    assert response.status_code == 200
    assert "Ultralytics YOLO model files" in response.text
    assert "AGPL-3.0" in response.text


def test_detect_round_trip(client):
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post("/api/detect", files={"file": ("sample.jpg", f, "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "onnx_cpu"
    assert len(body["detections"]) > 0
    assert set(body["stage_timings"]) == {"preprocess_ms", "inference_ms", "postprocess_ms"}


def test_yolo26n_detect_round_trip(client):
    with open(SAMPLE_IMAGE, "rb") as f:
        response = client.post(
            "/api/detect?backend=onnx_cpu_yolo26n",
            files={"file": ("sample.jpg", f, "image/jpeg")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "onnx_cpu_yolo26n"
    assert len(body["detections"]) > 0


def test_detect_rejects_non_image_content_type(client):
    response = client.post("/api/detect", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400


def test_detect_rejects_oversized_file(client):
    oversized = b"0" * (10 * 1024 * 1024 + 1)

    response = client.post("/api/detect", files={"file": ("big.jpg", oversized, "image/jpeg")})

    assert response.status_code == 413
