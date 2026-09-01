"""Dynamically quantise an exported ONNX model to INT8 weights.

Dev-only: depends on onnxruntime.quantization (and, transitively, sympy).
"""

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

MODEL_NAME = "yolo11n"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def quantize(model_name: str = MODEL_NAME) -> Path:
    """Dynamically quantise ``{model_name}.onnx`` weights to INT8.

    Activations remain float — dynamic quantisation only touches weights.
    Returns the path to the quantised model.
    """
    src = MODELS_DIR / f"{model_name}.onnx"
    dst = MODELS_DIR / f"{model_name}.int8.onnx"
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
    return dst


if __name__ == "__main__":
    dst_path = quantize()
    print(f"wrote {dst_path}")
