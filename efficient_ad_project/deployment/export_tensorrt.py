import argparse
from pathlib import Path
import torch

from anomalib.models import EfficientAd
from anomalib.engine import Engine
from anomalib.deploy import ExportType

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the exported ONNX model.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) used for training.")
    return parser.parse_args()

def export_to_onnx(model_path: str, output_dir: str, image_size: tuple[int, int]):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path}")
    model = EfficientAd.load_from_checkpoint(model_path)

    # Export to ONNX using Anomalib's Engine
    onnx_path = output_path / "model.onnx"
    print(f"Exporting model to ONNX format in {onnx_path}...")
    engine = Engine()
    engine.export(
        model=model,
        export_type=ExportType.ONNX,
        input_size=image_size,
        export_root=output_path,
    )
    print(f"ONNX model saved to {onnx_path}")
    print("\nONNX export complete. To build the TensorRT engine, please run the following command:")
    print(f"polygraphy run {onnx_path} --trt --save-engine {output_path / 'model.engine'} --input-shapes input:1x3x{image_size[0]}x{image_size[1]}")
    print("Note: This requires Polygraphy and TensorRT to be installed and configured correctly.")
    print("You might need to install Polygraphy with: pip install polygraphy[full]")

if __name__ == "__main__":
    args = get_args()
    export_to_onnx(args.model_path, args.output_dir, tuple(args.image_size))
