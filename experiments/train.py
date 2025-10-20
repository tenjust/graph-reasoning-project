import json
import h5py
import torch
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

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
        text = item["text"]
        label = item.get("label_str", "None")

        # Optionally add AMR info to input text
        if self.use_amr and "AMR_based_triplets" in item:
            amr_info = " ".join([" ".join(triplet) for triplet in item["AMR_based_triplets"]])
            text = text + " [AMR] " + amr_info

        source = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        target = self.tokenizer(
            label,
            truncation=True,
            padding="max_length",
            max_length=32,
            return_tensors="pt",
        )

        return {
            "input_ids": source["input_ids"].squeeze(),
            "attention_mask": source["attention_mask"].squeeze(),
            "labels": target["input_ids"].squeeze(),
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
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--output_dir", type=str, default="./t5_amr_rebel_output")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--use_amr", action="store_true", help="Include AMR-based triplets in input text")
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    # Load dataset
    train_ds, val_ds = load_dataset(tokenizer, args.train_data, args.val_data, args.max_length, args.use_amr)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch" if val_ds else "no",
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

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()

    # Save model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Training complete! Model saved to:", args.output_dir)


# ==============================
#  Entry point
# ==============================
if __name__ == "__main__":
    main()
