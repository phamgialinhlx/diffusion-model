from typing import Any, Tuple, Optional, Dict
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn
import math
from lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.optim import Optimizer, lr_scheduler
from src.utils.ema import LitEma


def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

class DiffusionModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        autoencoder,
        autoencoderconfig,
        autoencoder_ckpt_path,
        optimizer: Optimizer,
        scheduler: lr_scheduler,
        use_ema: bool = False,
        num_timesteps: int = 1000,
        beta_small: float = 0.0001,
        beta_large: float = 0.02,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.autoencoder = autoencoder
        if self.autoencoder is not None:
            self.autoencoder = autoencoder.load_from_checkpoint(
                autoencoder_ckpt_path, loss=autoencoder.loss
            ).to(self.device)
            self.autoencoder.eval()
            self.autoencoder.freeze()
        self.net = net
        self.num_timesteps = num_timesteps
        self.beta = torch.linspace(beta_small, beta_large, num_timesteps).to(self.device)
        self.alpha = torch.tensor(1. - self.beta).to(self.device)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.sigma2 = self.beta
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

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.alpha_bar = self.alpha_bar.to(x0.device)   
        mean = gather(self.alpha_bar, t) ** 0.5 * x0
        var = 1. - gather(self.alpha_bar, t)

        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        if eps is None:
            eps = torch.randn_like(x0)

        mean, var = self.q_xt_x0(x0, t)
        return mean + (var ** 0.5) * eps
    
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        eps_theta = self.net(xt, t)
        alpha_bar = gather(self.alpha_bar, t)
        alpha = gather(self.alpha, t)
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** .5
        mean = 1 / (alpha ** 0.5) * (xt - eps_coef * eps_theta)
        var = gather(self.sigma2, t)
        eps = torch.randn(xt.shape, device=xt.device)
        return mean + (var ** .5) * eps
    
    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        batch_size = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x0.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn_like(x0).to(x0.device)
        
        xt = self.q_sample(x0, t, eps=noise)
        eps_theta = self.net(xt, t)
        return F.mse_loss(noise, eps_theta)

    @torch.no_grad()
    def autoencoder_encode(self, x):
        if self.autoencoder is None:
            return x
        else:
            return self.autoencoder.encode(x).sample()

    def model_step(self, batch: Any):
        images, labels = batch
        latent = self.autoencoder_encode(images)
        return self.loss(latent)
    
    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self)-> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=self.hparams.scheduler.schedule)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


    @torch.no_grad()
    def log_image(self, images):
        x = self.autoencoder_encode(images)
        x = torch.randn(x.shape).to("cuda")

        sample_steps = torch.arange(self.num_timesteps - 1, 0, -1).to("cuda")
        if self.use_ema:
            # generate sample by ema_model
            with self.ema_scope():
                for t in sample_steps:
                    if t > 1:
                        z = torch.randn(x.shape).to("cuda")
                    else:
                        z = 0
                    e_hat = self.net(x, t.repeat(x.shape[0]).type(torch.float))
                    pre_scale = 1 / math.sqrt(self.alpha[t])
                    e_scale = (1 - self.alpha[t]) / math.sqrt(1 - self.alpha_bar[t])
                    post_sigma = math.sqrt(self.beta[t]) * z
                    x = pre_scale * (x - e_scale * e_hat) + post_sigma   
        else:
            for t in sample_steps:
                if t > 1:
                    z = torch.randn(x.shape).to("cuda")
                else:
                    z = 0
                e_hat = self.net(x, t.repeat(x.shape[0]).type(torch.float))
                pre_scale = 1 / math.sqrt(self.alpha[t])
                e_scale = (1 - self.alpha[t]) / math.sqrt(1 - self.alpha_bar[t])
                post_sigma = math.sqrt(self.beta[t]) * z
                x = pre_scale * (x - e_scale * e_hat) + post_sigma
        if self.autoencoder is not None:
            out_images = self.autoencoder.decode(x)
        else: 
            out_images = x
        return out_images

