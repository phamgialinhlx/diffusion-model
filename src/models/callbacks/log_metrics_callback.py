import torch
import wandb
from torchvision.utils import make_grid
import lightning as pl
from lightning.pytorch.callbacks.callback import Callback
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class LogMetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        # self.metrics = [
        #     {
        #         "name": "ssim",
        #         "metric": StructuralSimilarityIndexMeasure(data_range=[-1.0, 1.0])
        #     },
        #     {
        #         "name": "psnr",
        #         "metric": PeakSignalNoiseRatio(data_range=[-1.0, 1.0])
        #     }
        # ]
        self.ssim = StructuralSimilarityIndexMeasure(data_range=[-1.0, 1.0])
        self.psnr = PeakSignalNoiseRatio(data_range=[-1.0, 1.0])

    @torch.no_grad()
    def compute_metrics(self, pl_module, dataloader):
        for batch in dataloader:
            origin = batch[0].to(pl_module.device)
            image = pl_module.forward(origin)[0]
        #     for metric in self.metrics:
        #         metric["metric"].update(image, origin)
            self.ssim.update(image, origin)
            self.psnr.update(image, origin)
        
    @torch.no_grad()
    def reset_metrics(self):
        # for metric in self.metrics:
        #     metric["metric"].reset()
        self.ssim.reset()
        self.psnr.reset()

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:        
        origin = next(iter(trainer.val_dataloaders))[0].to(pl_module.device)
        image = pl_module.forward(origin)[0]

        nrows = 8
        if image.shape[0] == 256:
            nrows = 16

        compare = make_grid(torch.cat([origin, image], dim=3)
                           , nrow = 16, normalize=True, value_range=(-1, 1))
        origin = make_grid(origin, nrow=nrows, normalize=True, value_range=(-1, 1))
        image = make_grid(image, nrow=nrows, normalize=True, value_range=(-1, 1))

        self.compute_metrics(pl_module, trainer.val_dataloaders)
        trainer.logger.experiment.log({"ssim": self.ssim.compute(), "psnr": self.psnr.compute()})
        self.reset_metrics()

        trainer.logger.experiment.log({"image": [wandb.Image(origin), wandb.Image(image), wandb.Image(compare)], "caption": ["origin", "reconstruct", "compare"]})
