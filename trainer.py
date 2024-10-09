from argparse import ArgumentParser, Namespace
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AdamW)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.trainer import Trainer
from dataset import ParaphraserDataModule
from paraprahser import ParaphraserModel
import datasets

def parser_argument() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="VosLannack/qoura_id_paraphrase")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)

    parser.add_argument("--paraphaser_token", type=str, default="<paraphrase>")
    parser.add_argument("--save_dir", type=str, default="/content/drive/MyDrive/Thesis/paraphraser/")
    parser.add_argument("--logs_dir", type=str, default="/content/drive/MyDrive/Thesis/paraphraser/model_1/logs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--model_name", type=str, default="t5-small")

    parser.add_argument("--target_max_length", type=int, default=512)
    parser.add_argument("--source_max_length", type=int, default=512)

    parser.add_argument("--acc", type=str, default="gpu")
    
    return parser.parse_args()


if __name__ == "__main__":
    args: Namespace = parser_argument()
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(args.device)

    tokenizer.add_special_tokens({
        "additional_special_tokens": [args.paraphaser_token]        
    })

    data = datasets.load_dataset(args.dataset_path)
    train_set = data["train"]
    valid_set = data["validation"]

    data_module = ParaphraserDataModule(train_set=train_set,
                                       valid_set=valid_set,
                                       train_batch_size=args.train_batch_size,
                                       val_batch_size=args.val_batch_size,
                                       source_max_token_length=args.source_max_length,
                                       target_max_token_length=args.target_max_length,
                                       paraphraser_token=args.paraphaser_token,
                                       tokenizer=tokenizer)

    model_module = ParaphraserModel(model=model,
                                   optimizer=AdamW,
                                   optimizer_lr=args.lr)
            
    model_callbacks = ModelCheckpoint(
        dirpath=args.save_dir,
        monitor="val_loss",
        filename="best-checkpoint",
        save_last=False,
        save_top_k=1,
        mode="min"
    )
    logger = TensorBoardLogger(args.logs_dir, name="paraphraser_log")

    trainer = Trainer(
        accelerator=args.acc,
        callbacks=model_callbacks,
        logger=logger,
        max_epochs=args.epochs
    )

    trainer.fit(model_module,
                data_module)


