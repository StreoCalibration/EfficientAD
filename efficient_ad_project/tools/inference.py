import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from anomalib.models import EfficientAd


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image to inspect.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) for resizing.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold for classification.")
    parser.add_argument("--output_path", type=str, default="./result.png", help="Path to save the output visualization.")
    return parser.parse_args()


def infer(model_path: str, image_path: str, image_size: tuple[int, int], threshold: float, output_path: str):
    """Main inference function."""
    # 0. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load the model from the checkpoint
    print(f"Loading model from {model_path}")
    model = EfficientAd.load_from_checkpoint(model_path).to(device)
    model.eval() # Set model to evaluation mode

    # 2. Define image transformations (must be same as training)
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), # Example normalization
        ToTensorV2(),
    ])

    # 3. Load and preprocess the image
    print(f"Loading image from {image_path}")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image from {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    transformed = transform(image=image_rgb)
    image_tensor = transformed["image"].unsqueeze(0).to(device) # Add batch dimension and move to device

    # 4. Perform inference
    print("Performing inference...")
    with torch.no_grad():
        output = model(image_tensor)

    anomaly_map = np.zeros(image_size, dtype=np.float32)
    anomaly_score = output[0].item()

    # 5. Visualize and save the results
    print(f"Anomaly score: {anomaly_score:.4f}")

    # Normalize anomaly map for visualization
    anomaly_map = anomaly_map.astype(np.float32)
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
    anomaly_map_colored = (cv2.applyColorMap((anomaly_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET))

    # Overlay heatmap on original image
    image_resized = cv2.resize(image_bgr, (image_size[1], image_size[0]))
    overlay = cv2.addWeighted(image_resized, 0.6, anomaly_map_colored, 0.4, 0)

    # Add text for score and verdict
    verdict = "Anomalous" if anomaly_score > threshold else "Normal"
    cv2.putText(overlay, f"Score: {anomaly_score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Verdict: {verdict}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if verdict == "Anomalous" else (0, 255, 0), 2, cv2.LINE_AA)

    # Save the result
    cv2.imwrite(output_path, overlay)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    args = get_args()
    infer(args.model_path, args.image_path, tuple(args.image_size), args.threshold, args.output_path)
