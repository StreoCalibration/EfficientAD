import argparse
import yaml
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from anomalib.engine import Engine
from anomalib.models import EfficientAd
import torch
from anomalib.data import MVTecAD
from torchvision.transforms.v2 import Compose, Resize, Normalize, ToDtype

from src.data.provider import RealDatasetProvider, SyntheticDatasetProvider


def get_args():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    return parser.parse_args()


def train(config_path: str):
    """Main training function."""
    # Load the configuration file
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. Select the dataset provider based on the config
    data_source = config["data"]["source"]
    print(f"Using data source: {data_source}")

    if data_source == "real":
        provider = RealDatasetProvider(config)
        datamodule = provider.datamodule
    elif data_source == "synthetic":
        provider = SyntheticDatasetProvider(config)
        datamodule = provider.datamodule
    elif data_source == "mvtec":
        image_size = tuple(config["data"]["image_size"])
        augmentations = Compose([
            Resize(image_size),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToDtype(torch.float32, scale=True),
        ])
        datamodule = MVTecAD(
            root=config["data"]["path"],
            category=config["data"]["category"],
            train_batch_size=config["data"]["train_batch_size"],
            eval_batch_size=config["data"]["eval_batch_size"],
            augmentations=augmentations,
        )
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # 2. Initialize the Anomalib Engine
    # Create the model instance first
    model = EfficientAd(
        model_size=config["model"]["model_size"],
        # Add other model-specific parameters from config as needed
    )

    engine = Engine(
        accelerator="cuda",
        max_epochs=config["trainer"]["max_epochs"],
        # Add other engine parameters from config as needed
    )

    # 3. Start training
    print("Starting training...")
    engine.train(model=model, datamodule=datamodule)
    print("Training finished.")


if __name__ == "__main__":
    args = get_args()
    train(args.config)
