import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from typing import Mapping, List
from torch import Tensor


class ParaphraserDataset(Dataset):

    def __init__(this, data: Dataset,
                 source_max_length: int,
                 target_max_length:int,  
                 tokenizer, paraphraser_token: str):
        this.data: pd.DataFrame = pd.DataFrame(data)
        this.source_max_length: int = source_max_length
        this.target_max_length: int = target_max_length
        this.tokenizer = tokenizer
        this.paraphraser_token = paraphraser_token

    def __len__(this):
        return this.data.shape[0]

    def __getitem__(this, index: int) -> Mapping[str, Tensor]:
        item = this.data.iloc[index, :]      
        input_p = item.input_paraphraser
        target_p = item.target

        input_text = f"{this.paraphraser_token} {input_p}"

        target_text = f"{target_p}"

        input_ids, attention_mask = this._encode_text(input_text, this.source_max_length)
        target_ids, _ = this._encode_text(target_text, this.target_max_length)

        target_ids[target_ids == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids
        }
        

    def _encode_text(this, input_text: str, length):

        encoded = this.tokenizer(input_text, 
                                 max_length=length,
                                 padding="max_length",
                                 truncation=True,
                                 return_attention_mask=True,
                                 add_special_tokens=True,
                                 return_tensors="pt")

        return (encoded["input_ids"], 
                encoded["attention_mask"])


class ParaphraserDataModule(LightningDataModule):
    def __init__(this, train_set, valid_set,
                 test_set, tokenizer,
                 source_max_token_length: int,
                 target_max_token_length: int,
                 paraphraser_token: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 ):
        this.train_set = train_set
        this.valid_set = valid_set
        this.test_set = test_set

        this.train_batch_size = train_batch_size
        this.val_batch_size = val_batch_size

        this.source_max_token_length = source_max_token_length
        this.target_max_token_length = target_max_token_length
        this.paraphraser_token = paraphraser_token
        this.tokenizer = tokenizer

    def setup(this, stage: str = None) -> None:
        this.train_dataset: Dataset = ParaphraserDataset(this.train_set, this.source_max_token_length,
                                                        this.target_max_token_length, this.tokenizer,
                                                        this.paraphraser_token)

        this.valid_dataset: Dataset = ParaphraserDataset(this.valid_set, this.source_max_token_length,
                                                         this.target_max_token_length, this.tokenizer,
                                                         this.paraphraser_token)

        this.test_dataset: Dataset = ParaphraserDataset(this.test_set, this.source_max_token_length,
                                                        this.target_max_token_length, this.tokenizer,
                                                        this.paraphraser_token)
    
    def train_dataloader(this) -> DataLoader:
        return DataLoader(this.train_dataset, batch_size=this.train_batch_size, shuffle=True)
    
    def val_dataloader(this) -> DataLoader:
        return DataLoader(this.valid_dataset, batch_size=this.val_batch_size, shuffle=False)

    def test_dataloader(this) -> DataLoader:
        return DataLoader(this.test_dataset, batch_size=this.val_batch_size, shuffle=False)

