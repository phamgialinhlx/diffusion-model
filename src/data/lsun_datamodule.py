from typing import Any, Dict, Optional, Tuple

import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from lightning import LightningDataModule
from torch import Tensor

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']

def list_imgs(directory_path):
    # Initialize an empty list to store image file paths
    image_paths = []

    # Traverse the directory and subdirectories
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
                image_paths.append(os.path.join(root, file))
    return image_paths


class LSUNBase(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_root = data_root
        self.image_paths = list_imgs(data_root)
        self._length = len(self.image_paths)

        self.size = size
        self.interpolation = {
            "bilinear": PIL.Image.BILINEAR,
            "bicubic": PIL.Image.BICUBIC,
            "lanczos": PIL.Image.LANCZOS,
        }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = self.image_paths[i]
        image = Image.open(example)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
              (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        return [image.reshape(3, crop, crop)]
    
class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/lsun/bedroom_train", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(data_root="data/lsun/bedroom_val", flip_p=flip_p, **kwargs)

class LSUNBedroomDataModule(LightningDataModule):
    def __init__(
        self,
        image_channels: int = 3,
        image_size: int = 256,
        batch_size: int = 128,
        num_workers: int = 0,
        transform: Tensor = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.transforms = transform
        self.data_train = None
        self.data_val = None

    def setup(self, stage: Optional[str] = None):
        if not self.data_train or not self.data_val:
            self.data_train = LSUNBedroomsTrain()
            self.data_val = LSUNBedroomsValidation()
    
    def train_dataloader(self): 
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "data" / "lsun.yaml")
    _ = hydra.utils.instantiate(cfg)
    _.setup()
    train_images = next(iter(_.train_dataloader()))[0]
    print(train_images.shape)
    print(len(train_images))
    print(train_images.min(), train_images.max())
    val_images = next(iter(_.val_dataloader()))[0]
    print(val_images.shape)
    print(len(val_images))
    print(val_images.min(), val_images.max())
