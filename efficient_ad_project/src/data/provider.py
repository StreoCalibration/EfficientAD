from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import Dataset, DataLoader
from anomalib.data import Folder

from .synthetic_generator import generate_normal_image, generate_anomalous_image

class DatasetProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_train_loader(self) -> DataLoader:
        """Returns the data loader for training."""
        raise NotImplementedError

    @abstractmethod
    def get_test_loader(self) -> DataLoader:
        """Returns the data loader for testing."""
        raise NotImplementedError


class RealDatasetProvider(DatasetProvider):
    """Provides data from real datasets using anomalib."""

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration dictionary, same as pcb_config.yaml.
        """
        self.config = config
        # Use the correct 'Folder' class and parameters
        self.datamodule = Folder(
            name=config["data"]["category"], # Use category as the dataset name
            root=config["data"]["path"],
            normal_dir="train/good",
            abnormal_dir="test/anomaly",
            normal_test_dir="test/good",
            train_batch_size=self.config["data"]["train_batch_size"],
            eval_batch_size=self.config["data"]["eval_batch_size"],
            num_workers=0, # For Windows compatibility
        )
        self.datamodule.setup()

    def get_train_loader(self) -> DataLoader:
        """Returns the data loader for training."""
        return self.datamodule.train_dataloader()

    def get_test_loader(self) -> DataLoader:
        """Returns the data loader for testing."""
        return self.datamodule.test_dataloader()


class SyntheticDataset(Dataset):
    """A PyTorch dataset for generating synthetic images."""

    def __init__(self, num_samples: int, image_size: tuple[int, int], is_test: bool = False):
        self.num_samples = num_samples
        self.image_size = image_size
        self.is_test = is_test

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.is_test:
            image = generate_normal_image(self.image_size)
            label = 0
        else:
            # In test set, generate 50% normal, 50% anomalous
            if index % 2 == 0:
                image = generate_normal_image(self.image_size)
                label = 0
            else:
                image = generate_anomalous_image(self.image_size)
                label = 1

        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {"image": image_tensor, "label": torch.tensor(label)}


class SyntheticDatasetProvider(DatasetProvider):
    """Provides synthetically generated data."""

    def __init__(self, config: dict):
        """
        Args:
            config (dict): The configuration dictionary.
        """
        self.config = config
        self.image_size = tuple(self.config["data"]["image_size"])

    def get_train_loader(self) -> DataLoader:
        """Returns the data loader for training with synthetic normal images."""
        dataset = SyntheticDataset(num_samples=100, image_size=self.image_size)
        return DataLoader(
            dataset,
            batch_size=self.config["data"]["train_batch_size"],
            num_workers=0,
        )

    def get_test_loader(self) -> DataLoader:
        """Returns the data loader for testing with synthetic normal/anomalous images."""
        dataset = SyntheticDataset(num_samples=50, image_size=self.image_size, is_test=True)
        return DataLoader(
            dataset,
            batch_size=self.config["data"]["eval_batch_size"],
            num_workers=0,
        )
