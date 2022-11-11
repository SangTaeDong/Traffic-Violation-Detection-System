from module import NIA_SEGNet_module
import pytorch_lightning as pl

model = NIA_SEGNet_module.load_from_checkpoint(checkpoint_path="epoch=3-step=99999.ckpt")
model.batch_size = 4
# trainer = pl.Trainer(gpus=1) 
trainer = pl.Trainer(gpus=1,progress_bar_refresh_rate=0) 
# trainer = pl.Trainer(gpus=8, distributed_backend="ddp")
# trainer = pl.Trainer(gpus=2,distributed_backend="ddp") 
print(trainer.test(model))
