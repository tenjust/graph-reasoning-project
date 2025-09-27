# graph-reasoning-project

## Overview

This repository is a team workspace for experiments building on the **Graph Language Models** paper.
It includes:

1. **Baseline A** — the original paper’s code (GitHub repo), used as a reference.
2. **Baseline B** — our reimplementation of the paper’s model in this repo.
3. **Experiment-ready structure** for testing new inputs, datasets, and preprocessing formats (e.g., reasoning QA datasets, AMR/KG-style triples).

## Folder Structure

```
graph_reasoning_project/
├── baselines/            # Reference to original paper code and results
├── configs/              # YAML configuration files for experiments
├── data/                 # Raw and processed datasets
├── models/               # Reimplementation of models and variants
├── experiments/          # Training and evaluation scripts
├── preprocessing/        # Scripts to preprocess input data (AMR/KG etc.)
├── scripts/              # Bash scripts to launch experiments
├── notebooks/            # Notebooks for exploration and analysis
├── README.md
├── requirements.txt
└── environment.yml
```

---

## Folder Details

* **baselines/**

  * `original_README.md` — notes on paper repo commit, commands, and results
  * `run_original.sh` — wrapper to run the paper’s baseline code
  * `results/` — stores logs and checkpoints from the paper repo

* **configs/**

  * `baseline_a.yaml` — paper’s setup (reference)
  * `baseline_b.yaml` — our reimplementation, same input as paper
  * `exp_kg.yaml` — future experiments with KG/AMR or other datasets

* **data/**

  * `raw/` — original datasets (ConceptNet, REBEL, QA datasets)
  * `processed/` — preprocessed data for experiments
  * `README.md` — instructions on downloading and preprocessing

* **models/**

  * `glm_baseline.py` — our reimplementation of the paper’s model
  * `glm_kg.py` — variant with new input preprocessing (future experiments)

* **experiments/**

  * `train.py` — training entrypoint (reads configs)
  * `evaluate.py` — evaluation pipeline
  * `utils.py` — helper functions for training and evaluation

* **preprocessing/**

  * `linearise.py` — transform input to linearised triples (paper-style)
  * `kg_format.py` — transform input to KG-style triples (future experiments)

* **scripts/**

  * `run_baseline_b.sh` — run Baseline B with its config
  * `run_exp_kg.sh` — run KG/AMR experiments with corresponding config

* **notebooks/**

  * `data_inspection.ipynb` — explore and inspect datasets
  * `results_visualization.ipynb` — visualize results and metrics

---

## Getting Started

1. **Clone this repo**

```bash
git clone <your-repo-url>
cd graph-reasoning-project
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
# OR if using conda
conda env create -f environment.yml
conda activate graph-reasoning-project
```

3. **Download datasets**

* Follow instructions in `data/README.md` for ConceptNet, REBEL, or reasoning QA datasets.

4. **Run Baseline A (paper reference)**

```bash
bash baselines/run_original.sh
```

5. **Run Baseline B (our reimplementation)**

```bash
bash scripts/run_baseline_b.sh
```

6. **Run experiments with new inputs (KG/AMR, QA datasets)**

```bash
bash scripts/run_exp_kg.sh
```

---

## Notes

* **Baseline A** = official paper implementation. Used as reference.
* **Baseline B** = reimplementation in this repo, foundation for experiments.
* **YAML configs** are the source of truth for training parameters, datasets, and model choices.
* Ensure `PYTHONPATH` includes this directory when running scripts.
