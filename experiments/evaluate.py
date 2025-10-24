import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from experiments.train import RebelAMRDataset  # reuse your dataset class


def evaluate_model(model_dir, data_path, model_name="t5-small", split="train", num_samples=5, use_amr=False):
    """
    Evaluate the model on a subset of the dataset and print sample predictions.
    """

    # Load tokenizer and trained model
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()

    # Load dataset
    dataset = RebelAMRDataset(tokenizer, data_path, split=split, use_amr=use_amr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"\nEvaluating model from: {model_dir}")
    print(f"Dataset: {data_path}")
    print(f"Using AMR: {use_amr}\n")

    # Loop through a few examples
    for i, batch in enumerate(loader):
        if i >= num_samples:
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=64)

        input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        predicted = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = tokenizer.decode(batch["labels"][0], skip_special_tokens=True)

        print(f"🔹 Sample {i+1}")
        print("Input:", input_text)
        print("Predicted:", predicted)
        print("Target:", label)
        print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="t5-small")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--use_amr", action="store_true", help="Include AMR-based triplets in evaluation input")
    args = parser.parse_args()

    evaluate_model(
        model_dir=args.model_dir,
        data_path=args.data_path,
        model_name=args.model_name,
        split=args.split,
        num_samples=args.num_samples,
        use_amr=args.use_amr,
    )


if __name__ == "__main__":
    main()
