import torch
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchvision.transforms import transforms

class LogImageCallback(Callback):
    def __init__(self):
        super().__init__()

    def normalize(self, x):
        # x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        mean = 0.5 # x.mean()
        std = 0.5 # x.std()
        x = (x - mean) / std
        return x

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
        # image = self.normalize(image)
        # origin = self.normalize(origin)

        origin = make_grid(origin, nrow=nrows, normalize=False)
        image = make_grid(image, nrow=nrows, normalize=False)
        

        trainer.logger.experiment.log({"image": [wandb.Image(origin), wandb.Image(image)], "caption": ["origin", "reconstruct"]})
