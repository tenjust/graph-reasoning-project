import json
import logging

import h5py
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, T5Config

from baselines.AMR_KG_rel_classifier import Graph2GraphRelationClassifier, preprocess


# ==============================
#  Dataset class
# ==============================
class RebelAMRDataset(Dataset):
    """General dataset class for REBEL and REBEL+AMR datasets."""

    def __init__(self, tokenizer, hdf5_path, split="train", max_length=512, use_amr=False):
        self.tokenizer = tokenizer
        self.file = h5py.File(hdf5_path, "r")
        self.data = self.file[split] if split in self.file else self.file["train"]
        self.max_length = max_length
        self.use_amr = use_amr  # if True, append AMR-based triplets to text

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = json.loads(self.data[idx].decode("utf-8"))
        data = preprocess(item, self.tokenizer, self.use_amr)

        return {
            "input_ids": data.input_ids,
            "relative_position": data.relative_position,
            "sparsity_mask": data.sparsity_mask,
            "use_additional_bucket": data.use_additional_bucket,
            # "indices": data.indices,
            "labels": data.label,
        }



# ==============================
#  Load data
# ==============================
def load_dataset(tokenizer, train_path, val_path=None, max_length=512, use_amr=False):
    train_ds = RebelAMRDataset(tokenizer, train_path, split="train", max_length=max_length, use_amr=use_amr)
    val_ds = None
    if val_path:
        val_ds = RebelAMRDataset(tokenizer, val_path, split="train", max_length=max_length, use_amr=use_amr)
    return train_ds, val_ds


# ==============================
#  Main training
# ==============================
def main():
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="./t5_amr_rebel_output")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_amr", action="store_true", help="Input AMR-based triplets instead of the raw text")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more detail
        format="%(asctime)s [%(levelname)s]: %(message)s"
    )

    # Load tokenizer and model
    config = T5Config.from_pretrained("t5-small")
    config.num_classes = 5
    config.model_name_size = "t5-small"
    config.model_max_length = 512
    config.rel_attn_num_additional_buckets = 2  # number of additional buckets for graph-to-graph attention
    config.use_cache = False

    model = Graph2GraphRelationClassifier(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load dataset
    train_path = "baselines/GraphLanguageModels/data/rebel_dataset/REBEL_AMR_TRIPLES.train.hdf5"
    train_ds = RebelAMRDataset(model.tokenizer, train_path, split="train", max_length=args.max_length, use_amr=args.use_amr)
    val_path = "baselines/GraphLanguageModels/data/rebel_dataset/REBEL_AMR_TRIPLES.val.hdf5"
    val_ds = RebelAMRDataset(model.tokenizer, val_path, split="val", max_length=args.max_length, use_amr=args.use_amr)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch", # if val_ds else "no",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",  # disables wandb
        save_strategy="epoch",
        load_best_model_at_end=bool(val_ds),
    )

    def compute_loss(model, data):
        """Custom loss function to handle model outputs and labels in batches."""
        labels = data.get("labels")
        logits = model(**data)
        predictions = model.get_classes(logits)
        print("predictions GGG", predictions)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(predictions, labels)
        return loss

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=model.tokenizer,
        compute_loss_func=compute_loss,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)

    print("Training complete! Model saved to:", args.output_dir)


# ==============================
#  Entry point
# ==============================
if __name__ == "__main__":
    main()
