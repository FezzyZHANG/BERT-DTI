# Integration Instructions

## Overview
This integration adapts your BERT-DTI model to work with the FezzyZHANG/batch_runs repository's standardized experiment framework.

## Key Changes Made

### 1. **Simplified Architecture**
- Removed PyTorch Lightning dependency (not used in batch_runs)
- Converted to standard PyTorch training loop
- Maintained original BERT-DTI model architecture

### 2. **Data Format Adaptation**
- **Input**: Parquet files with columns `[smiles, sequence, label]`
- **Original**: Custom dataset loading from multiple formats
- **Benefit**: Standardized across all batch_runs models

### 3. **Configuration Management**
- **Original**: JSON config files
- **New**: YAML config files (batch_runs standard)
- **Migration**: All hyperparameters preserved

### 4. **Training Loop**
- **Original**: PyTorch Lightning automated training
- **New**: Manual training loop with validation
- **Features**: Best model saving, tensorboard logging, metrics calculation

### 5. **Metrics Integration**
- Added standard DTI metrics: AUROC, AUPRC, Precision, Accuracy
- Compatible with batch_runs evaluation framework
- Outputs to standardized `metrics.json`

## Files Created

1. **`main.py`** - Main entry point following batch_runs API
2. **`config.yaml`** - Configuration in batch_runs format  
3. **`run.sh`** - Shell script for execution
4. **`model.py`** - Standalone model definition
5. **`dataset.py`** - Dataset class for parquet data

## Installation Steps

### 1. Copy to batch_runs Repository
```bash
# In the batch_runs repository
mkdir models/BERT_DTI
cp batch_runs_integration/* models/BERT_DTI/
```

### 2. Update Model Registry
Add to `model_names.txt`:
```
BERT_DTI
```

### 3. Install Dependencies
```bash
pip install transformers torch pandas pyyaml scikit-learn tqdm tensorboard
```

### 4. Prepare Data
Ensure your datasets are in parquet format with columns:
- `smiles`: Drug SMILES strings
- `sequence`: Protein amino acid sequences  
- `label`: Interaction labels (0/1 or continuous)

## Usage Examples

### Single Model Run
```bash
# From batch_runs root directory
bash models/BERT_DTI/run.sh \
    /path/to/output \
    /path/to/data/split \
    /path/to/tensorboard \
    42
```

### Batch Run (All Models, All Datasets)
```bash
bash scripts/run_all.sh
```

### Custom Configuration
Edit `models/BERT_DTI/config.yaml` to modify:
- Model architecture (`layer_features`)
- Training parameters (`learning_rate`, `batch_size`, `num_epochs`)
- Input lengths (`drug_max_length`, `prot_max_length`)
- Pretrained model usage (`pretrained.drug`, `pretrained.prot`)

## Configuration Options

### Model Architecture
```yaml
layer_features: [768, 32, 1]  # Hidden layer sizes
dropout: 0.1                  # Dropout rate
layer_limit: true             # Reduce protein model layers
prot_layers_keep: 18          # Number of BERT layers to keep
```

### Training Parameters
```yaml
train_params:
  batch_size: 32
  learning_rate: 5e-6
  num_epochs: 50
  loss_function: bce          # 'bce' or 'smooth_l1'
  val_interval: 1             # Validation frequency
```

### Input Processing
```yaml
drug_max_length: 64           # Max SMILES length
prot_max_length: 545          # Max protein sequence length
```

## Performance Considerations

### Memory Usage
- **Layer Limit**: Reduces protein BERT from 30 to 18 layers
- **Batch Size**: Adjust based on GPU memory
- **Sequence Length**: Truncate long sequences

### Training Speed
- **Mixed Precision**: Can be added to training loop
- **Gradient Accumulation**: For larger effective batch sizes
- **DataLoader Workers**: Parallel data loading

## Metrics and Evaluation

### Supported Metrics
- **AUROC**: Area under ROC curve
- **AUPRC**: Area under precision-recall curve  
- **Precision**: Classification precision
- **Accuracy**: Classification accuracy
- **MSE**: Mean squared error (for regression)

### Output Format
Results saved as `metrics.json`:
```json
{
    "auroc": 0.85,
    "auprc": 0.78,
    "precision": 0.82,
    "accuracy": 0.79
}
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce `batch_size` in config.yaml
   - Reduce `prot_max_length` or `drug_max_length`

2. **Slow Training**
   - Enable `layer_limit: true`
   - Increase `num_workers` in DataLoader

3. **Poor Performance**
   - Check data quality and distribution
   - Adjust learning rate
   - Try different loss functions

### Debugging
- Set `DEBUG = True` in main.py for verbose output
- Use tensorboard for training monitoring
- Check data statistics with dataset.get_statistics()

## Future Enhancements

### Possible Additions
1. **Attention Visualization**: Add attention weight extraction
2. **Model Ensembles**: Multiple model averaging
3. **Advanced Metrics**: More sophisticated evaluation metrics
4. **Data Augmentation**: SMILES and sequence augmentation
5. **Transfer Learning**: Fine-tuning strategies

### Integration with Original Code
- **Checkpoint Loading**: Load your existing trained models
- **Attention Analysis**: Port attention visualization tools
- **Multi-task Learning**: Extend for multiple prediction tasks

This integration preserves the core BERT-DTI functionality while making it compatible with the batch_runs standardized experiment framework, enabling systematic evaluation across multiple datasets and comparison with other DTI models.
