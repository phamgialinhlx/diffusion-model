import torch
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchvision.transforms import transforms

class LogImageCallback(Callback):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Log image
        # print("Logging image")
        # from IPython import embed; embed()
        
        nrows = 8
        origin = next(iter(trainer.val_dataloaders))[0].to(pl_module.device)
        image = pl_module.forward(origin)[0]
        if image.shape[0] == 256:
            nrows = 16

        compare = make_grid(torch.cat([origin, image], dim=3)
                           , nrow = 16, normalize=True, value_range=(-1, 1))
        origin = make_grid(origin, nrow=nrows, normalize=True, value_range=(-1, 1))
        image = make_grid(image, nrow=nrows, normalize=True, value_range=(-1, 1))

        trainer.logger.experiment.log({"image": [wandb.Image(origin), wandb.Image(image), wandb.Image(compare)], "caption": ["origin", "reconstruct", "compare"]})
