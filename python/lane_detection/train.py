
import datetime
import platform
import psutil 
import cpuinfo
from module import NIA_SEGNet_module
import pytorch_lightning as pl
import torch
import random
import time
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import shutil


torch.manual_seed(777)
random.seed(777)

model = NIA_SEGNet_module()
model.batch_size = 4
# print(model.fcn)
# print(flush=True)
trainer = pl.Trainer(gpus=[0],prepare_data_per_node=True, distributed_backend="ddp", callbacks=[EarlyStopping(monitor='val_loss',patience=2)])
trainer.fit(model)
