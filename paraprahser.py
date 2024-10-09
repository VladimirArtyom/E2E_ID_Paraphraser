from pytorch_lightning import LightningModule
from torch import Tensor
from typing import Mapping


class ParaphraserModel(LightningModule):
    def __init__(this,
                 model,
                 optimizer,
                 optimizer_lr: float):
        super().__init__()
        this.model = model 
        this.lr = optimizer_lr
        this.opt = optimizer

    def forward(this, input_ids, attention_mask, labels=None):
        output: Tensor = this.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
        return output.loss, output.logits
    

    def training_step(this, batch, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(this, batch, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(this, batch, batch_indx: int):
        loss = this.exe_step(batch, batch_indx)
        this.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def exe_step(this, batch: Mapping[str, Tensor], batch_indx: int):
        inputs_ids: Tensor = batch["input_ids"]
        attention_mask: Tensor = batch["attention_mask"]
        labels: Tensor = batch["labels"]
        loss, output = this(input_ids=inputs_ids,
                            attention_mask=attention_mask,
                            labels=labels)
        return loss
    
    def configure_optimizers(this):
        return this.opt(this.parameters(), lr=this.lr)
