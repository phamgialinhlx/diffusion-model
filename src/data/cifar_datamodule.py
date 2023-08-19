from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms


class CIFARDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        image_channels: int = 3,
        image_size: int = 32,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def classes(self):
        return (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed."""
        CIFAR10(root=self.hparams.data_dir, train=True, download=True, transform=self.transforms)
        CIFAR10(root=self.hparams.data_dir, train=False, download=True, transform=self.transforms)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            self.data_train = CIFAR10(root=self.hparams.data_dir, train=True, transform=self.transforms)
            self.data_val = CIFAR10(root=self.hparams.data_dir, train=False, transform=self.transforms)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

if __name__ == "__main__":
    _ = CIFARDataModule()
    _.setup()
    images, labels = next(iter(_.train_dataloader()))
    print(images.shape)
    print(labels.shape)
    print(images.min(), images.max())
