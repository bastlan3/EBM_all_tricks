import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class AugmentedImageDataset(Dataset):
    """
    A PyTorch Dataset that wraps a base dataset (like CIFAR10) and applies
    a set of defined image augmentations on the fly.
    """
    def __init__(self, base_dataset, transform):
        """
        Args:
            base_dataset: The underlying dataset (e.g., from torchvision.datasets).
            transform: The augmentation pipeline from albumentations.
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # The base dataset returns (image, label). We only need the image.
        image, _ = self.base_dataset[idx]
        # Convert PIL image to numpy array for albumentations
        image_np = np.array(image)

        # Apply augmentations
        augmented = self.transform(image=image_np)
        augmented_image = augmented['image']

        return augmented_image

class HeuristicPretrainingDataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the heuristic pre-training task.
    It downloads a base dataset (CIFAR10) and applies a strong augmentation
    pipeline to generate varied images for noise/quality scoring.
    """
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 4, image_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        # Define a strong augmentation pipeline using albumentations
        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size, p=1.0),
            # Geometric transformations
            A.RandomCrop(height=self.image_size, width=self.image_size, p=0.5),
            A.HorizontalFlip(p=0.5),
            # Blur transformations
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.8),
                A.MotionBlur(blur_limit=(3, 7), p=0.8),
            ], p=0.7), # Apply one of the blurs 70% of the time

            # Noise transformations
            A.OneOf([
                A.GaussNoise(p=0.8), # Using default variance
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.8),
            ], p=0.7), # Apply one of the noises 70% of the time

            # Color transformations
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),

            # Normalize and convert to tensor
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ])

    def prepare_data(self):
        # Download the dataset
        torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            cifar_full = torchvision.datasets.CIFAR10(self.data_dir, train=True)
            self.train_dataset = AugmentedImageDataset(cifar_full, self.transform)

        if stage == 'test' or stage is None:
            cifar_test = torchvision.datasets.CIFAR10(self.data_dir, train=False)
            self.test_dataset = AugmentedImageDataset(cifar_test, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
