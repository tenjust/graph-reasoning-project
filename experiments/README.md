# Plan for Experiments

## Configurations

1. Run Moritz's code directly with his dataset.
2. Run Moritz's code with our dataset.
3. Run the T5 we upgraded with Moritz's dataset.
4. Run the T5 we upgraded with our dataset.
5. Fine-tune the T5 we upgraded with Moritz's dataset.
6. Fine-tune the T5 we upgraded with our dataset.

## Training

### Train on **REBEL (text-only)**

```bash
python experiments/train.py \
  --train_data data/REBEL_original.train.hdf5 \
  --model_name t5-small \
  --output_dir models/t5_rebel \
  --num_epochs 1 \
  --batch_size 4
```

### Train on **REBEL + AMR (includes AMR-based triplets)**

```bash
python experiments/train.py \
  --train_data data/REBEL_AMR_TRIPLES.train.hdf5 \
  --model_name t5-small \
  --output_dir models/t5_rebel_amr \
  --num_epochs 1 \
  --batch_size 4 \
  --use_amr
```

**Notes:**

* The argument `--use_amr` appends AMR-derived triplets to the input text.
* You can increase `--num_epochs` or `--batch_size` depending on your hardware.

---

## Evaluation

### Evaluate **REBEL model**

```bash
python experiments/evaluate.py \
  --model_dir models/t5_rebel \
  --data_path data/REBEL_original.train.hdf5 \
  --split train \
  --num_samples 5
```

### Evaluate **REBEL+AMR model**

```bash
python experiments/evaluate.py \
  --model_dir models/t5_rebel_amr \
  --data_path data/REBEL_AMR_TRIPLES.train.hdf5 \
  --split train \
  --num_samples 5 \
  --use_amr
```

The script prints for each example:

* The **input text**
* The **predicted relation triples**
* The **ground-truth labels**

---

## Hyperparameter Optimization

The hyperopt.py script automates the process of tuning hyperparameters such as learning rate, batch size, and number of epochs.

### Run Hyperparameter Search

```bash
python experiments/hyperopt.py
```

## File Overview

| File          | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `train.py`    | Defines dataset loading, model training, and fine-tuning of T5.        |
| `evaluate.py` | Evaluates trained models and prints predictions for manual inspection. |
| `hyperopt.py` | Runs hyperparameter tuning over multiple configurations.               |
| `README.md`   | Usage instructions for experiments.                                    |
