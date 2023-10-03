from typing import List, Optional, Tuple, Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
import torch
from torchmetrics import MaxMetric, MeanMetric
from lightning import LightningModule
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import hydra
from omegaconf import DictConfig

import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.modules.diffusionmodules import Encoder, Decoder
from src.models.modules.distributions import DiagonalGaussianDistribution


class Autoencoder(LightningModule):
    """
    ## Autoencoder

    This consists of the encoder and decoder modules.
    """

    def __init__(
        self,
        autoencoderconfig,
        embed_dim: int,
        loss: nn.Module,
        image_key=0,
        lr: float = 4.5e-6,
        ckpt_path: str = None,
        colorize_nlabels=None,
        monitor=None,
    ) -> None:
        """
        :param encoder: is the encoder
        :param decoder: is the decoder
        :param embed_dim: is the number of dimensions in the quantized embedding space
        :param z_channels: is the number of channels in the embedding space
        """
        super().__init__()
        self.automatic_optimization = False
        self.image_key = image_key
        self.loss = loss
        self.encoder = Encoder(**autoencoderconfig)
        self.decoder = Decoder(**autoencoderconfig)
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(
            2 * autoencoderconfig["z_channels"], 2 * embed_dim, 1
        )
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(embed_dim, autoencoderconfig["z_channels"], 1)
        self.embed_dim = embed_dim

        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        # TODO: implement self.init_from_ckpt
        # if ckpt_path is not None:
        # self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.save_hyperparameters(logger=False)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=(-1.0, 1.0))
        self.psnr = PeakSignalNoiseRatio(data_range=(-1.0, 1.0))

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def encode(self, img: torch.Tensor) -> DiagonalGaussianDistribution:
        """
        ### Encode images to latent representation

        :param img: is the image tensor with shape `[batch_size, img_channels, img_height, img_width]`
        """
        # Get embeddings with shape `[batch_size, z_channels * 2, z_height, z_height]`
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution
        return DiagonalGaussianDistribution(moments)

    def decode(self, z: torch.Tensor):
        """
        ### Decode images from latent representation

        :param z: is the latent representation with shape `[batch_size, embed_dim, z_height, z_height]`
        """
        # Map to embedding space from the quantized representation
        z = self.post_quant_conv(z)
        # Decode the image of shape `[batch_size, channels, height, width]`
        return self.decoder(z)

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        # x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def on_train_start(self) -> None:
        self.ssim.reset()
        self.psnr.reset()

    def on_validation_start(self) -> None:
        self.ssim.reset()
        self.psnr.reset()

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self.forward(inputs)
        opt_ae, opt_disc = self.optimizers()

        # train encoder + decoder + logvar
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True
        )
        self.log_dict(
            log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )

        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # train the discriminator
        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="train",
        )
        self.log(
            "discloss",
            discloss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True,
        )
        self.log_dict(
            log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
        )

        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

        self.ssim.update(reconstructions, inputs)
        self.psnr.update(reconstructions, inputs)
        self.log(
            "train/ssim",
            self.ssim.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/psnr",
            self.psnr.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.ssim.update(reconstructions, inputs)
        self.psnr.update(reconstructions, inputs)
        self.log(
            "val/ssim",
            self.ssim.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/psnr",
            self.psnr.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=self.hparams.lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9)
        )
        # return [opt_ae, opt_disc], []
        return [{"optimizer": opt_ae}, {"optimizer": opt_disc}]

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_image(self, images):
        dec, posterior = self.forward(images)
        return dec

@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="train_ae.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    print(f"Instantiating model <{cfg.model._target_}>")
    IMG_SIZE = 32
    IMG_CHANNELS = 4
    cfg.model.autoencoderconfig.channels = IMG_SIZE
    cfg.model.autoencoderconfig.img_channels = IMG_CHANNELS
    cfg.model.autoencoderconfig.channel_multipliers = [1, 2, 4]
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    input = torch.randn(2, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    output = model(input)
    print(sum(p.numel() for p in model.parameters()))
    print(output[0].shape)
    print(output[1].sample().shape)


if __name__ == "__main__":
    main()
