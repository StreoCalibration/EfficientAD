import argparse
import cv2
import numpy as np
import os
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # This initializes CUDA context

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the TensorRT engine file (.engine).")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the image or directory to inspect.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width) for resizing.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly score threshold for classification.")
    parser.add_argument("--output_path", type=str, default="./result.png", help="Path to save the output visualization.")
    return parser.parse_args()

def infer(model_path: str, input_path: str, image_size: tuple[int, int], threshold: float, output_path: str):
    # 0. Load TensorRT engine
    print(f"Loading TensorRT engine from {model_path}")
    with open(model_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()

    # Allocate buffers
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.get_binding_dtype(binding).itemsize
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        
        host_mem = cuda.pagelocked_empty(size // dtype.itemsize, dtype)
        device_mem = cuda.mem_alloc(size)

        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})

    # Define image transformations (must be same as training)
    transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1]),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
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
        # Load and preprocess the image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Warning: Could not read image from {image_path}. Skipping.")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        transformed = transform(image=image_rgb)
        # Input to TensorRT is NCHW float32
        input_data = transformed["image"].unsqueeze(0).cpu().numpy().astype(np.float32)

        # Copy input data to host buffer
        np.copyto(inputs[0]['host'], input_data.ravel())

        # Perform inference and measure time
        print("Performing inference...")
        start_time = time.time()
        cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for out in outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
        stream.synchronize()
        end_time = time.time()
        inference_time = end_time - start_time
        all_inference_times.append(inference_time)

        # Get output names and map them
        output_names = [engine.get_binding_name(i) for i in range(engine.num_bindings) if not engine.binding_is_input(i)]
        output_dict = {}
        for j, name in enumerate(output_names):
            output_dict[name] = outputs[j]['host']

        anomaly_score = output_dict['pred_score'].item()
        # Reshape anomaly_map from flat array to 2D image size
        anomaly_map = output_dict['anomaly_map'].reshape(image_size).astype(np.float32)

        # Visualize and save the results
        print(f"Anomaly score: {anomaly_score:.4f}")
        print(f"Inference time: {inference_time:.4f} seconds")

        # Normalize anomaly map for visualization
        anomaly_map = anomaly_map.astype(np.float32)
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-6)
        anomaly_map_colored = cv2.applyColorMap(np.ascontiguousarray((anomaly_map_norm * 255).astype(np.uint8)), cv2.COLORMAP_JET)

        # Overlay heatmap on original image
        image_resized = cv2.resize(image_bgr, (image_size[1], image_size[0]))
        overlay = cv2.addWeighted(image_resized.astype(np.float32), 0.6, anomaly_map_colored.astype(np.float32), 0.4, 0).astype(np.uint8)

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
    infer(args.model_path, args.input_path, tuple(args.image_size), args.threshold, args.output_path)
