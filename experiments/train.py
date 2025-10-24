import torch
from transformers import Trainer, TrainingArguments, T5Config

from data.utils import load_dataset
from models.AMR_KG_rel_classifier import Graph2GraphRelationClassifier


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
    # tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    # model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    config = T5Config.from_pretrained("t5-small")
    model = Graph2GraphRelationClassifier(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Load dataset
    train_ds, val_ds = load_dataset(model.tokenizer, args.train_data, args.val_data, args.max_length, args.use_amr)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch" if val_ds else "no",
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
        tokenizer=model.tokenizer,
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
