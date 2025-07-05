import argparse
import os
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
from src.data.synthetic_generator import generate_normal_image, generate_anomalous_image


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(description="Generate and save synthetic anomaly detection data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Base directory to save the data (e.g., 'data/pcb').")
    parser.add_argument("--num_train", type=int, default=100, help="Number of normal images for the training set.")
    parser.add_argument("--num_test_normal", type=int, default=50, help="Number of normal images for the test set.")
    parser.add_argument("--num_test_anomalous", type=int, default=50, help="Number of anomalous images for the test set.")
    parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256], help="Image size (height width).")
    return parser.parse_args()


def generate_and_save_data(output_dir: str, num_train: int, num_test_normal: int, num_test_anomalous: int, image_size: tuple[int, int]):
    """Generates and saves synthetic images to disk."""
    base_path = Path(output_dir)
    train_path = base_path / "train" / "good"
    test_normal_path = base_path / "test" / "good"
    test_anomalous_path = base_path / "test" / "anomaly" # A generic anomaly folder

    # Create directories
    for path in [train_path, test_normal_path, test_anomalous_path]:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Generate and save training images (normal)
    print(f"\nGenerating {num_train} training images...")
    for i in range(num_train):
        img = generate_normal_image(image_size)
        cv2.imwrite(str(train_path / f"{i:03d}.png"), img)

    # Generate and save normal test images
    print(f"Generating {num_test_normal} normal test images...")
    for i in range(num_test_normal):
        img = generate_normal_image(image_size)
        cv2.imwrite(str(test_normal_path / f"{i:03d}.png"), img)

    # Generate and save anomalous test images
    print(f"Generating {num_test_anomalous} anomalous test images...")
    for i in range(num_test_anomalous):
        img = generate_anomalous_image(image_size)
        cv2.imwrite(str(test_anomalous_path / f"{i:03d}.png"), img)

    print("\nData generation complete.")


if __name__ == "__main__":
    args = get_args()
    generate_and_save_data(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test_normal=args.num_test_normal,
        num_test_anomalous=args.num_test_anomalous,
        image_size=tuple(args.image_size),
    )
