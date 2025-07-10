import argparse
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time

from anomalib.models import EfficientAd


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the image or directory to inspect.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) for resizing.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold for classification.")
    parser.add_argument("--output_path", type=str, default="./result.png", help="Path to save the output visualization.")
    return parser.parse_args()


def infer(model_path: str, input_path: str, image_size: tuple[int, int], threshold: float, output_path: str):
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

    image_paths = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for filename in files:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image_paths.append(os.path.join(root, filename))
    elif os.path.isfile(input_path):
        image_paths.append(input_path)
    else:
        raise FileNotFoundError(f"Could not find input at {input_path}")

    all_inference_times = []

    for i, image_path in enumerate(image_paths):
        print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
        # 3. Load and preprocess the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not read image from {image_path}. Skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image_rgb)
        image_tensor = transformed["image"].unsqueeze(0).to(device) # Add batch dimension and move to device

        # 4. Perform inference and measure time
        print("Performing inference...")
        start_time = time.time()
        with torch.no_grad():
            output = model(image_tensor)
        end_time = time.time()
        inference_time = end_time - start_time
        all_inference_times.append(inference_time)

        anomaly_map = output.anomaly_map.squeeze().cpu().numpy().astype(np.float32)
        anomaly_score = output.pred_score.item()

        # 5. Visualize and save the results
        print(f"Anomaly score: {anomaly_score:.4f}")
        print(f"Anomaly map mean: {anomaly_map.mean():.4f}, std: {anomaly_map.std():.4f}")
        print(f"Inference time: {inference_time:.4f} seconds")

        # Normalize anomaly map for visualization
        anomaly_map = anomaly_map.astype(np.float32)
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        anomaly_map_colored = cv2.resize(cv2.applyColorMap((anomaly_map_norm * 255).astype(np.uint8), cv2.COLORMAP_JET), (image_size[1], image_size[0]))

        # Overlay heatmap on original image
        image_resized = cv2.resize(image_bgr, (image_size[1], image_size[0]))
        overlay = cv2.addWeighted(image_resized, 0.6, anomaly_map_colored, 0.4, 0)

        # Create binary mask from anomaly map for contour detection
        # Use a threshold (e.g., 0.5) on the normalized anomaly map to identify anomalous regions
        binary_mask = (anomaly_map_norm > threshold).astype(np.uint8) * 255
        
        # Debug prints for anomaly map and binary mask
        print(f"Anomaly map norm min: {anomaly_map_norm.min():.4f}, max: {anomaly_map_norm.max():.4f}")
        print(f"Binary mask unique values: {np.unique(binary_mask)}")

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the overlay image (Red, thicker for visibility)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 5) # Red contours for anomalies

        # Add text for score, verdict, and inference time
        verdict = "Anomalous" if anomaly_score > threshold else "Normal"
        cv2.putText(overlay, f"Score: {anomaly_score:.4f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Verdict: {verdict}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if verdict == "Anomalous" else (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"Time: {inference_time:.4f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Save the result
        if os.path.isdir(input_path):
            relative_path = os.path.relpath(image_path, input_path)
            output_filename = os.path.join(output_path, relative_path)
            output_dir = os.path.dirname(output_filename)
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_filename = output_path

        cv2.imwrite(output_filename, overlay)
        print(f"Result saved to {output_filename}")

        # Save the binary mask for debugging
        mask_output_filename = os.path.splitext(output_filename)[0] + "_mask" + os.path.splitext(output_filename)[1]
        cv2.imwrite(mask_output_filename, binary_mask)
        print(f"Binary mask saved to {mask_output_filename}")

        # Save the normalized anomaly map for debugging
        anomaly_map_norm_output_filename = os.path.splitext(output_filename)[0] + "_anomaly_map_norm" + os.path.splitext(output_filename)[1]
        cv2.imwrite(anomaly_map_norm_output_filename, (anomaly_map_norm * 255).astype(np.uint8))
        print(f"Normalized anomaly map saved to {anomaly_map_norm_output_filename}")

    if all_inference_times:
        avg_inference_time = sum(all_inference_times) / len(all_inference_times)
        print(f"\nAverage inference time across {len(all_inference_times)} images: {avg_inference_time:.4f} seconds")


if __name__ == "__main__":
    args = get_args()
    infer(args.model_path, args.input_path, tuple(args.image_size), args.threshold, args.output_path)
