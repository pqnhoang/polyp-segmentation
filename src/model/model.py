import torch
import torch.nn as nn
import pytorch_lightning as pl
from .unet import UNet
from .resunet import Resnet50Unet
from .loss import dice_score, DiceLoss


class NeoPolypModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-4, name: str = "resunet"):
        super().__init__()
        if name == "resunet":
            self.model = Resnet50Unet(n_classes=3)
        else:
            self.model = UNet(in_channels=3, attention=False, recurrent=False)
        self.lr = lr
        self.dice_loss = DiceLoss()
        self.entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _forward(self, batch, batch_idx, name="train"):
        image, mask = batch['image'].float(), batch['mask'].long()
        logits = self(image)
        loss = self.entropy_loss(logits, mask)
        d_score = dice_score(logits, mask)
        acc = (logits.argmax(dim=1) == mask).float().mean()
        self.log_dict(
            {
                f"{name}_loss": loss,
                f"{name}_dice_score": d_score,
                f"{name}_acc": acc
            },
            on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._forward(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            patience=5,
            verbose=True,
            factor=0.5
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss',
            'interval': 'epoch',
        }