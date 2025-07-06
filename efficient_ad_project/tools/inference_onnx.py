import argparse
import cv2
import numpy as np
import onnxruntime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import time

def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ONNX model (.onnx).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the image or directory to inspect.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) for resizing.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold for classification.")
    parser.add_argument("--output_path", type=str, default="./results_Test/bottle_test_onnx", help="Path to save the output visualization.")
    return parser.parse_args()

def infer_onnx(model_path: str, input_path: str, image_size: tuple[int, int], threshold: float, output_path: str):
    """Main inference function for ONNX models."""
    
    # 0. Set up ONNX runtime session
    print(f"Loading ONNX model from {model_path}")
    session = onnxruntime.InferenceSession(model_path, providers=onnxruntime.get_available_providers())
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]

    # 1. Define image transformations (must be same as training)
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
        # 2. Load and preprocess the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not read image from {image_path}. Skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image_rgb)
        image_tensor = transformed["image"].numpy().astype(np.float32) # Convert to numpy array
        image_tensor = np.expand_dims(image_tensor, axis=0) # Add batch dimension

        # 3. Perform inference and measure time
        print("Performing inference...")
        start_time = time.time()
        outputs = session.run(output_names, {input_name: image_tensor})
        end_time = time.time()
        inference_time = end_time - start_time
        all_inference_times.append(inference_time)

        # Assuming the ONNX model outputs are in the same order as PyTorch model (score, anomaly_map)
        anomaly_score = outputs[0].item()
        
        print(f"Raw shape of outputs[1]: {outputs[1].shape}") # Add this line for debugging

        anomaly_map = outputs[1].squeeze() # Remove batch and channel dimensions
        anomaly_map = anomaly_map.astype(np.float32) # Convert to float32 immediately
        
        # If anomaly_map is not 2D (H, W), resize it to image_size
        if len(anomaly_map.shape) != 2 or anomaly_map.shape != image_size:
            print(f"Resizing anomaly_map from {anomaly_map.shape} to {image_size}")
            anomaly_map = cv2.resize(anomaly_map, (image_size[1], image_size[0]), interpolation=cv2.INTER_LINEAR)

        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)

        # Ensure the array is contiguous and then convert to uint8
        anomaly_map_for_colormap = np.ascontiguousarray((anomaly_map_norm * 255).astype(np.uint8))
        anomaly_map_colored = (cv2.applyColorMap(anomaly_map_for_colormap, cv2.COLORMAP_JET))
        # Ensure anomaly_map_colored is 3-channel if it's not already
        if len(anomaly_map_colored.shape) == 2: # If it's 2D, convert to 3D
            anomaly_map_colored = cv2.cvtColor(anomaly_map_colored, cv2.COLOR_GRAY2BGR)

        # Overlay heatmap on original image
        image_resized = cv2.resize(image_bgr, (image_size[1], image_size[0]))

        print(f"Shape of image_resized: {image_resized.shape}, dtype: {image_resized.dtype}")
        print(f"Shape of anomaly_map_colored: {anomaly_map_colored.shape}, dtype: {anomaly_map_colored.dtype}")

        # Convert both images to float32 for weighted addition
        image_resized_float = image_resized.astype(np.float32)
        anomaly_map_colored_float = anomaly_map_colored.astype(np.float32)

        overlay = cv2.addWeighted(image_resized_float, 0.6, anomaly_map_colored_float, 0.4, 0)
        overlay = overlay.astype(np.uint8) # Convert back to uint8 for saving

        # Create binary mask from anomaly map for contour detection
        binary_mask = (anomaly_map_norm > threshold).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the overlay image
        cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2) # Yellow contours for anomalies

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

    if all_inference_times:
        avg_inference_time = sum(all_inference_times) / len(all_inference_times)
        print(f"\nAverage inference time across {len(all_inference_times)} images: {avg_inference_time:.4f} seconds")


if __name__ == "__main__":
    args = get_args()
    infer_onnx(args.model_path, args.input_path, tuple(args.image_size), args.threshold, args.output_path)
