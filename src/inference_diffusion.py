from typing import List, Optional, Tuple

import hydra
import lightning as L
import pyrootutils
import torch
from lightning import LightningDataModule, LightningModule
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from torchvision.utils import make_grid, save_image
import math
from tqdm import tqdm
from PIL import Image
import numpy as np

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.models.diffusion.sampler.ddim import DDIMSampler
from src.models.diffusion.sampler.ddpm import DDPMSampler

def interpolate(latent_matrix_A, latent_matrix_B, num_steps=5):
    # Perform linear interpolation between A and B
    interpolated_matrices = []
    for t in torch.linspace(0, 1, num_steps):
        interpolated_matrix = (1 - t) * latent_matrix_A + t * latent_matrix_B
        interpolated_matrices.append(interpolated_matrix)

    # Convert the list of interpolated matrices to a tensor
    interpolated_matrices_tensor = torch.stack(interpolated_matrices)

    return interpolated_matrices_tensor


def interpolate_process(model, image_A, image_B, num_steps=5):
    latent_matrix_A = model.encode(image_A.unsqueeze(0)).sample().squeeze()
    latent_matrix_B = model.encode(image_B.unsqueeze(0)).sample().squeeze()
    print(latent_matrix_A.shape)
    print(latent_matrix_B.shape)
    interpolated_matrices = interpolate(
        latent_matrix_A, latent_matrix_B, num_steps=num_steps
    )
    interpolated_images = model.decode(interpolated_matrices)
    interpolated_images = torch.stack([image_A, *interpolated_images, image_B])
    interpolated_images = make_grid(
        interpolated_images, normalize=True, value_range=(-1, 1)
    )
    save_image(interpolated_images, "interpolated.png")


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="train_diffusion.yaml"
)
def main(cfg: DictConfig):
    cfg.task_name = "inference"
    inference(cfg)


@torch.no_grad()
def inference(cfg: DictConfig):
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    print(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup()
    print(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    assert cfg.ckpt_path is not None
    model = model.load_from_checkpoint(cfg.ckpt_path, autoencoder=model.autoencoder).to(
        "cuda"
    )

    images, labels = next(iter(datamodule.val_dataloader()))
    images = images.to("cuda")
    # from IPython import embed
    # embed()
    # model.sampler = DDIMSampler(n_infer_steps=50).to("cuda")
    # model.use_ema = False
    out_images = model.log_image(images, device="cuda")

    out_images = make_grid(
        torch.cat([images, out_images], dim=3),
        nrow=16,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(out_images, "results.png")
    # interpolate_process(autoencoder, images[120], images[40], num_steps=54)


if __name__ == "__main__":
    main()

# python src/inference_diffusion.py data=cifar.yaml data.batch_size=256 ckpt_path="/work/hpc/pgl/lung-diffusion/logs/train/runs/2023-08-29_02-07-15/checkpoints/last.ckpt" task_name="inference"
