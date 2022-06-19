import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class ConvNet(pl.LightningModule):
    def __init__(self, in_channel,  out_channel, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channel, out_channel // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.MaxPool2d(2),  # B, 64, 14, 14
            nn.ReLU(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.MaxPool2d(2),  # B, 128, 7, 7
            nn.ReLU(),
        )

        self.lin_projection = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(out_channel * 7 * 7, num_classes))

    def forward(self, x):
        embedding = self.feature_extractor(x)
        output = self.lin_projection(embedding)
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self(x)
        loss = F.cross_entropy(x, y)
        acc = sum(y == x.argmax(-1)) / len(y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self(x)
        loss = F.cross_entropy(x, y)
        acc = sum(y == x.argmax(-1)) / len(y)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('valid_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
