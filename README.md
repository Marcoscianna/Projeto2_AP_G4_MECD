# Projeto 2 de Aprendizagem Profunda – Grupo 4

## Constituição do grupo

pg60004	Filipa Oliveira da Silva   
pg60099	Francisco Ricardo Teixeira Silva   
pg58750	Lara Cristiana Sousa Antunes   
e12961	Marco Scianna   

## Estrutura do Repositório

- `Apresentação/` Pasta com vídeo de apresentação do trabalho


## Notebooks principais

Os notebooks principais utilizados para obtenção dos resultados finais são:



## Usage (English additions)

This repository includes `MIQR_CC_solution.ipynb` which implements training, evaluation, EDA and Grad-CAM for the MIQR-CC dataset.

Quick start (CPU):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook projeto2/MIQR_CC_solution.ipynb
```

Quick start (GPU):

- Install a CUDA-enabled PyTorch build appropriate for your GPU following https://pytorch.org.
- Then run the notebook; it includes device detection and recommended DataLoader kwargs.

Experiment tracking (optional):

- The notebook contains an optional WandB integration cell. To enable it, set `USE_WANDB=True` and set `WANDB_PROJECT`.
- WandB will log per-epoch metrics and can upload model artifacts when enabled.

Notes:
- The notebook will create `_prepared_splits` from `metadata.csv` if not present and writes checkpoints to `../models/`.
- Professor-provided Portuguese text is preserved; helper cells added by the assistant are in English.


