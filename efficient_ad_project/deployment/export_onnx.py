import torch
from anomalib.models import EfficientAd
from pathlib import Path

def export_to_onnx(model_path: str, output_dir: str, image_size: tuple[int, int]):
    """Exports a trained PyTorch model to ONNX format."""
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load the model from the checkpoint
    print(f"Loading model from {model_path}")
    model = EfficientAd.load_from_checkpoint(model_path)
    model.eval() # Set model to evaluation mode

    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 2. Create a dummy input
    dummy_input = torch.randn(1, 3, image_size[0], image_size[1]).to(device) # Batch size 1, 3 channels (RGB)

    # 3. Export the model to ONNX
    onnx_file_path = output_path / "model.onnx"
    print(f"Exporting model to ONNX format at {onnx_file_path}...")
    torch.onnx.export(model,               # model being run
                       dummy_input,         # model input (or a tuple for multiple inputs)
                       onnx_file_path,      # where to save the model (file or file-like object)
                       export_params=True,  # store the trained parameter weights inside the model file
                       opset_version=11,    # the ONNX version to export the model to
                       do_constant_folding=True, # whether to execute constant folding for optimization
                       input_names = ['input'],   # the names to assign to the input & output nodes
                       output_names = ['output'],
                       dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                     'output' : {0 : 'batch_size'}})

    print("\nExport complete.")
    print(f"ONNX model saved at {onnx_file_path}")

if __name__ == "__main__":
    model_checkpoint_path = "F:/Source/EfficientAD/results/EfficientAd/MVTecAD/bottle/v1/weights/lightning/model.ckpt"
    output_directory = "F:/Source/EfficientAD/results_Test/bottle_onnx_export" # Using results_Test as per convention
    image_input_size = (256, 256)

    export_to_onnx(model_checkpoint_path, output_directory, image_input_size)
