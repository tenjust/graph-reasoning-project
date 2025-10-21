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

## File Overview

| File          | Description                                            |
| ------------- | ------------------------------------------------------ |
| `train.py`    | Defines dataset loading and fine-tuning of T5.         |
| `evaluate.py` | Evaluates trained models and prints predictions.       |
| `utils.py`    | Reserved for helper functions (can be expanded later). |
| `plan.md`     | Notes on planned experiments.                          |
| `README.md`   | Usage instructions.                                    |
