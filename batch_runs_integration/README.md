# BERT-DTI Integration for batch_runs Framework

This directory contains the integration of the BERT-DTI model into the FezzyZHANG/batch_runs repository's model API style.

## Files Created:

1. **main.py** - Main training and evaluation script following batch_runs API
2. **config.yaml** - Configuration file in batch_runs format
3. **run.sh** - Shell script for model execution
4. **model.py** - Adapted BERT-DTI model for batch_runs compatibility
5. **dataset.py** - Dataset class compatible with batch_runs data format

## Key Adaptations:

- **Data Format**: Adapted to use parquet files with columns: `smiles`, `sequence`, `label`
- **Model Architecture**: Preserved BERT-DTI architecture but adapted input/output handling
- **Training Loop**: Simplified to fit batch_runs training paradigm
- **Metrics**: Added standard DTI metrics (AUROC, AUPRC, etc.)
- **Configuration**: Moved from JSON to YAML format

## Usage:

Copy these files to a new model directory in the batch_runs repository:
```
models/BERT_DTI/
├── main.py
├── config.yaml
├── run.sh
├── model.py
└── dataset.py
```

Then add "BERT_DTI" to the `model_names.txt` file in the batch_runs repository.
