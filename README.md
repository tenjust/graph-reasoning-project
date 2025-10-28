# graph-reasoning-project

## Overview

This repository explores **graph-based reasoning** with language models, inspired by the *Graph Language Models (GLM)* paper from Heidelberg NLP.  
It combines **AMR (Abstract Meaning Representation)** and **Knowledge Graph (KG)** data for reasoning and relation classification tasks.

The project includes:
- **Reproduction of GLM baselines** from the official repository  
- **Custom datasets and preprocessing** (REBEL and AMR)  
- **Reimplementation and extension** of models for graph-based reasoning  


## Folder Structure

```
graph-reasoning-project/
├── data/ # Scripts and data for REBEL dataset preprocessing
| ├── rebel_dataset/ # General data storage directory
│ ├── amr_datasets.py
│ ├── amr_examples.py
│ ├── augment_rebel.py
│ ├── load_rebel.py
│ ├── load_rebel.sh
│ └── utils.py
│
├── experiments/ # Training, evaluation, and tuning
│ ├── train.py
│ ├── evaluate.py
│ ├── hyperopt.py
│ └── README.md
│
├── models/ # Model definitions
│ └── AMR_KG_rel_classifier.py
│ ├── GraphLanguageModels/ # Official GLM repo (cloned reference)
│ ├── scripts/
│ │ ├── run_original_graph_only.sh
│ │ ├── run_original_text_guided.sh
│ │ └── train_text_guided.sh
│ └── ...
|
├── preprocessing/ # AMR preprocessing and graph utilities
│ ├── amr_graph_before_rewriting.ipynb
│ ├── amr_graph.py
│ └── utils.py
│
├── config.py
├── setup.sh # Environment setup script
├── requirements.txt # Python dependencies
└── README.md
```

---

## Folder Details

### **data/**
Contains all dataset-related scripts and files for **REBEL** preprocessing and augmentation.

- **rebel_dataset/** — stores general dataset files and preprocessed data  
- `amr_datasets.py` — prepares AMR-formatted datasets for experiments  
- `amr_examples.py` — examples and utilities for AMR-encoded samples  
- `augment_rebel.py` — augments REBEL data with additional reasoning patterns  
- `load_rebel.py` / `load_rebel.sh` — scripts to download and preprocess the REBEL dataset  
- `utils.py` — helper functions for dataset I/O and formatting  

---

### **experiments/**
Contains all training, evaluation, and tuning scripts.

- `train.py` — main training entry point  
- `evaluate.py` — runs evaluation on validation/test datasets  
- `hyperopt.py` — performs hyperparameter search or ablation studies  
- `README.md` — additional notes or experiment-specific documentation  

---

### **models/**
Contains model architectures and baseline references.

- `AMR_KG_rel_classifier.py` — core model integrating AMR and KG representations for relation reasoning  
- **GraphLanguageModels/** — cloned reference implementation of the original *Graph Language Models* paper  
  - `scripts/run_original_graph_only.sh` — runs the paper’s **graph-only baseline**  
  - `scripts/run_original_text_guided.sh` — runs the **text + graph baseline**  
  - `scripts/train_text_guided.sh` — fine-tuning entry point for the paper’s model  

---

### **preprocessing/**
Scripts for AMR graph preprocessing and utilities.

- `amr_graph_before_rewriting.ipynb` — notebook for exploring and rewriting AMR graphs  
- `amr_graph.py` — final AMR parsing and transformation for model input  
- `utils.py` — shared helper functions for preprocessing pipelines  

---

### **config.py**
Central configuration file for dataset paths, model parameters, and preprocessing options.

---

### **setup.sh**
Environment setup script — installs dependencies and prepares the environment for running experiments.

---

### **requirements.txt**
List of Python dependencies required for the project.

---

### **README.md**
Project overview, usage instructions, and experiment documentation.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/tenjust/graph-reasoning-project.git
cd graph-reasoning-project
```


2. **Install dependencies**
You can install dependencies using a bash script:
```bash
bash setup.sh
```

3. **Prepare datasets**

Run preprocessing for the REBEL dataset:
```bash
bash data/load_rebel.sh
```

4. **Train and evaluate**

All scripts for training, evaluation, and hyperparameter tuning are located in the `experiments/` folder.  

To run experiments, see the detailed instructions in:
```bash
experiments/README.md
```

5. **Run original GLM baselines**

```bash
bash models/scripts/run_original_graph_only.sh
bash models/scripts/run_original_text_guided.sh
```

---
