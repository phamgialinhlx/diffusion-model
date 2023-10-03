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


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1.0, 1.0)
    x = (x + 1.0) / 2.0
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


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
    autoencoder = model.autoencoder
    autoencoder = autoencoder.load_from_checkpoint(
        cfg.model.autoencoder_ckpt_path,
        loss=autoencoder.loss,
    ).to("cuda")

    images, labels = next(iter(datamodule.val_dataloader()))
    images = images.to("cuda")

    x = model.autoencoder_encode(images)

    sample_steps = torch.arange(model.num_timesteps - 1, 0, -1).to("cuda")
    for t in tqdm(sample_steps, desc="Sampling"):
        if t > 1:
            z = torch.randn(x.shape).to("cuda")
        else:
            z = 0
        e_hat = model.net(x, t.repeat(x.shape[0]).type(torch.float))

        pre_scale = 1 / math.sqrt(model.alpha(t))
        e_scale = (1 - model.alpha(t)) / math.sqrt(1 - model.alpha_bar(t))
        post_sigma = math.sqrt(model.beta(t)) * z
        x = pre_scale * (x - e_scale * e_hat) + post_sigma

    # from IPython import embed; embed()
    out_images = model.autoencoder.decode(x)

    out_images = make_grid(
        torch.cat([images, out_images], dim=3),
        nrow=16,
        normalize=True,
        value_range=(-1, 1),
    )
    save_image(out_images, "test.png")
    # interpolate_process(autoencoder, images[120], images[40], num_steps=54)


if __name__ == "__main__":
    main()

# python src/inference_diffusion.py data=cifar.yaml data.batch_size=256 ckpt_path="/work/hpc/pgl/lung-diffusion/logs/train/runs/2023-08-29_02-07-15/checkpoints/last.ckpt" task_name="inference"
