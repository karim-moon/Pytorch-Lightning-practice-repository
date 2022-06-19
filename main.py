from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from model import ConvNet

BATCH_SIZE = 64

train_data = MNIST(root='./data/02/',
                   train=True,
                   download=True,
                   transform=transforms.ToTensor())
test_data = MNIST(root='./data/02/',
                  train=False,
                  download=True,
                  transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_data,
                          batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=test_data,
                         batch_size=BATCH_SIZE, shuffle=True)

model = ConvNet(1, 128, 10)

wandb_logger = WandbLogger(name=f'MNIST_batch_{BATCH_SIZE}', project='MNIST')

check_point_callback = ModelCheckpoint(monitor='train_acc',
                                       dirpath='./',
                                       filename='{p_id}-{epoch}-{MNIST:.3f}',
                                       save_top_k=2
                                       )

trainer = pl.Trainer(max_epochs=40, gpus=1,
                     callbacks=check_point_callback,
                     logger=wandb_logger,
                     )

trainer.fit(model,
            train_dataloader=train_loader,
            val_dataloaders=valid_loader,
            )
