import torch
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchvision.transforms import transforms


class LogImageCallback(Callback):
    def __init__(self, frequency: int = 1):
        super().__init__()
        self.count = 0
        self.frequency = frequency

    @torch.no_grad()
    def on_validation_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.count += 1
        if self.count % self.frequency == 0:
            nrows = 8
            # from IPython import embed
            # embed()
            origin = next(iter(trainer.val_dataloaders))[0].to(pl_module.device)

            image = pl_module.log_image(origin, device=pl_module.device)

            if image.shape[0] == 2 or image.shape[0] == 1:
                nrows = 1

            if image.shape[0] == 8 or image.shape[0] == 4:
                nrows = 2

            if image.shape[0] == 16 or image.shape[0] == 32:
                nrows = 4
            
            if image.shape[0] == 256:
                nrows = 16

            value_range = (-1, 1) if image.shape[1] != 1 else (0, 1)
            compare = make_grid(
                torch.cat([origin, image], dim=3),
                nrow=nrows,
                normalize=True,
                value_range=value_range,
            )

            origin = make_grid(
                origin, nrow=nrows, normalize=True, value_range=value_range
            )
            image = make_grid(
                image, nrow=nrows, normalize=True, value_range=value_range
            )

            trainer.logger.experiment.log(
                {
                    "image": [
                        wandb.Image(origin),
                        wandb.Image(image),
                        wandb.Image(compare),
                    ],
                    "caption": ["origin", "reconstruct", "compare"],
                }
            )
