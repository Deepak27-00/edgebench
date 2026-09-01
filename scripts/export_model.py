"""Export a YOLO11 model to ONNX for CPU inference with onnxruntime.

Dev-only: depends on ultralytics, which is not part of the runtime image.
"""

from pathlib import Path

import onnx
from ultralytics import YOLO

MODEL_NAME = "yolo11n"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def export_model(model_name: str = MODEL_NAME) -> Path:
    """Download ``{model_name}.pt`` and export it to ONNX under ``models/``.

    Uses imgsz=640, opset=13, dynamic=False, simplify=True. Returns the path
    to the exported ``.onnx`` file.
    """
    model = YOLO(f"{model_name}.pt")
    exported = model.export(format="onnx", imgsz=640, opset=13, dynamic=False, simplify=True)
    onnx_path = MODELS_DIR / f"{model_name}.onnx"
    Path(exported).replace(onnx_path)
    return onnx_path


def print_io_shapes(onnx_path: Path) -> None:
    """Print the input and output tensor names and shapes of an ONNX model."""
    model = onnx.load(str(onnx_path))
    print(f"Model: {onnx_path}")
    for tensor in model.graph.input:
        dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
        print(f"  input  {tensor.name}: {dims}")
    for tensor in model.graph.output:
        dims = [d.dim_value or d.dim_param for d in tensor.type.tensor_type.shape.dim]
        print(f"  output {tensor.name}: {dims}")


if __name__ == "__main__":
    onnx_path = export_model()
    print_io_shapes(onnx_path)
