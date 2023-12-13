from typing import Any, Dict, Optional, Tuple

from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
import torch
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datasets import LIDCDataset

class LIDCDatamodule(LightningDataModule):
        
    def __init__(
        self, 
        data_dir='', 
        batch_size: int = 128, 
        train_val_split: Tuple[int, int] = (800, 200), 
        augmentation=False, 
        num_workers: int = 0, 
        pin_memory: bool = False
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.batch_size_per_device = batch_size

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            print(self.hparams.data_dir)
            dataset = LIDCDataset(root_dir=self.hparams.data_dir, augmentation=self.hparams.augmentation)
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(42),
            )
    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "lidc.yaml")
    cfg.data_dir = str(root / "data" / "lidc" / "data")
    datamodule = hydra.utils.instantiate(cfg)
    datamodule.setup()
    sample = next(iter(datamodule.train_dataloader()))
    print(sample['data'].shape)