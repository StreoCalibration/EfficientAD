import argparse
from pathlib import Path

from anomalib.models import EfficientAd
from anomalib.export import export, ExportMode


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the exported OpenVINO model.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) used for training.")
    return parser.parse_args()


def export_to_openvino(model_path: str, output_dir: str, image_size: tuple[int, int]):
    """Exports a trained PyTorch model to OpenVINO format."""
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the model from the checkpoint
    print(f"Loading model from {model_path}")
    model = EfficientAd.load_from_checkpoint(model_path)

    # 2. Export the model to OpenVINO format
    print(f"Exporting model to OpenVINO format in {output_dir}...")
    export(
        model=model,
        input_size=image_size,
        export_mode=ExportMode.OPENVINO,
        export_root=output_dir,
    )

    print("\nExport complete.")
    print(f"OpenVINO model saved in {output_dir}/openvino/")


if __name__ == "__main__":
    args = get_args()
    export_to_openvino(args.model_path, args.output_dir, tuple(args.image_size))
