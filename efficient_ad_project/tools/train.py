import argparse
import yaml
import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from anomalib.engine import Engine
from anomalib.models import EfficientAd

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
    elif data_source == "synthetic":
        provider = SyntheticDatasetProvider(config)
    else:
        raise ValueError(f"Unknown data source: {data_source}")

    # 2. Initialize the Anomalib Engine
    # Create the model instance first
    model = EfficientAd(
        model_size=config["model"]["model_size"], # 다시 원래대로 대문자 사용
        # Add other model-specific parameters from config as needed
    )

    engine = Engine(
        accelerator="cuda",
        max_epochs=config["trainer"]["max_epochs"],
        # Add other engine parameters from config as needed
    )

    # 3. Get data loaders from the provider
    train_loader = provider.get_train_loader()
    test_loader = provider.get_test_loader()

    # 4. Start training
    print("Starting training...")
    engine.train(model=model, datamodule=provider.datamodule)
    print("Training finished.")


if __name__ == "__main__":
    args = get_args()
    train(args.config)
