from typing import Any, Tuple, Optional, Dict
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from tqdm import tqdm
import math
from lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.optim import Optimizer, lr_scheduler
from src.utils.ema import LitEma
from src.models.vqvae_module import VQVAE
from src.models.klvae_module import KLVAE
from src.models.diffusion.sampler import BaseSampler
from src.models.diffusion.sampler.ddpm import DDPMSampler


def gather(consts: torch.Tensor, t: torch.Tensor, device="cuda"):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.to(device).gather(-1, t.to(device))
    return c.reshape(-1, 1, 1, 1)

def load_autoencoder(ckpt_path):
    try:
        ae = KLVAE.load_from_checkpoint(ckpt_path).eval()
    except Exception as e:
        ae = VQVAE.load_from_checkpoint(ckpt_path).eval()
    ae.freeze()
    return ae

class DiffusionModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        autoencoder_ckpt_path,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        sampler: BaseSampler = DDPMSampler(),
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.autoencoder = load_autoencoder(autoencoder_ckpt_path)

        self.net = net
        self.sampler = sampler
        # exponential moving average
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.net)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.net)

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.net.parameters())
            self.model_ema.copy_to(self.net)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.net.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        t = torch.randint(
            0,
            self.sampler.n_train_steps,
            [x0.shape[0]],
            device=x0.device,
        )
        xt, noise = self.sampler.step(x0, t, noise)
        eps_theta = self.net(xt, t)
        return F.mse_loss(noise, eps_theta)

    @torch.no_grad()
    def autoencoder_encode(self, x):
        if self.autoencoder is None:
            return x
        else:
            if type(self.autoencoder) is VQVAE:
                return self.autoencoder.encode(x)[0]
            else:
                return self.autoencoder.encode(x).sample()

    def model_step(self, batch: Any):
        images, labels = batch
        latent = self.autoencoder_encode(images)
        return self.loss(latent)

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=optimizer, lr_lambda=self.hparams.scheduler.schedule
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    @torch.no_grad()
    def log_image(
        self,
        xt,
        noise: torch.Tensor = None,
        repeat_noise: bool = False,
        cond: Tensor = None,
        device: torch.device = torch.device('cpu'),
        prog_bar: bool = True,
    ) -> Tensor:
        xt = self.autoencoder_encode(xt)
        xt = torch.randn(xt.shape).to(device)
        sample_steps = (
            tqdm(self.sampler.timesteps, desc="Sampling t")
            if prog_bar
            else self.sampler.timesteps
        )
        
        if self.use_ema:
            # generate sample by ema_model
            with self.ema_scope():
                for i, t in enumerate(sample_steps):
                    
                    t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                    model_output = self.net(x=xt, timesteps=t, cond=cond)
                    xt = self.sampler.reverse_step(
                        model_output, t, xt, noise, repeat_noise
                    )
        else:
            for i, t in enumerate(sample_steps):
                t = torch.full((xt.shape[0],), t, device=device, dtype=torch.int64)
                model_output = self.net(x=xt, timesteps=t, cond=cond)
                xt = self.sampler.reverse_step(
                    model_output, t, xt, noise, repeat_noise
                )
        if self.autoencoder is not None:
            out_images = self.autoencoder.decode(xt)
        else:
            out_images = xt
        return out_images
