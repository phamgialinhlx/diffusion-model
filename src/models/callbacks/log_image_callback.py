import torch
import wandb

import lightning as pl
from lightning.pytorch.callbacks.callback import Callback

class LogImageCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Log image
        print("Logging image")
        image = pl_module.forward(pl_module.example_input_array)
        trainer.logger.experiment.log({"image": [wandb.Image(image)]})