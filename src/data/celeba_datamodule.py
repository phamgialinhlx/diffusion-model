from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import CelebA
from torchvision.transforms import transforms
import torchvision.datasets as dset

class CelebADatamodule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        image_channels: int = 3,
        image_size: int = 128,
        batch_size: int = 64,
        n_classes: int = 1,
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
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None



    def prepare_data(self):
        """Download data if needed."""
        # CelebA(
        #     root=self.hparams.data_dir,
        #     split='train',
        #     download=True,
        #     transform=self.transforms,
        # )
        # CelebA(
        #     root=self.hparams.data_dir,
        #     split='valid',
        #     download=True,
        #     transform=self.transforms,
        # )
        pass

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = dset.ImageFolder(root=self.hparams.data_dir + 'celeba/',
                           transform=transforms.Compose([
                               transforms.Resize(self.hparams.image_size),
                               transforms.CenterCrop(self.hparams.image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[27000, 3000],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True,
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
    _ = CelebADatamodule()
    _.prepare_data()
    _.setup()
    images, labels = next(iter(_.train_dataloader()))
    print(images.shape)
    print(labels.shape)
    print(images.min(), images.max())
