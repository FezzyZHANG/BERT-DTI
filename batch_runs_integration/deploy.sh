#!/bin/bash

# Deployment script for BERT-DTI integration with batch_runs
# This script helps deploy the BERT-DTI model to a batch_runs repository

set -e  # Exit on any error

echo "=== BERT-DTI batch_runs Integration Deployment ==="
echo ""

# Function to print colored output
print_success() {
    echo -e "\033[32m✓ $1\033[0m"
}

print_error() {
    echo -e "\033[31m✗ $1\033[0m"
}

print_info() {
    echo -e "\033[34mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[33m⚠ $1\033[0m"
}

# Check if batch_runs repository path is provided
if [ $# -eq 0 ]; then
    print_error "Please provide the path to your batch_runs repository"
    echo "Usage: $0 /path/to/batch_runs_repository"
    exit 1
fi

BATCH_RUNS_REPO="$1"

# Validate batch_runs repository
if [ ! -d "$BATCH_RUNS_REPO" ]; then
    print_error "Directory $BATCH_RUNS_REPO does not exist"
    exit 1
fi

if [ ! -f "$BATCH_RUNS_REPO/model_names.txt" ]; then
    print_error "$BATCH_RUNS_REPO does not appear to be a batch_runs repository (missing model_names.txt)"
    exit 1
fi

print_info "Deploying to batch_runs repository: $BATCH_RUNS_REPO"

# Create BERT_DTI model directory
MODEL_DIR="$BATCH_RUNS_REPO/models/BERT_DTI"
print_info "Creating model directory: $MODEL_DIR"

if [ -d "$MODEL_DIR" ]; then
    print_warning "Model directory already exists. Backing up..."
    mv "$MODEL_DIR" "$MODEL_DIR.backup.$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$MODEL_DIR"

# Get current script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy model files
print_info "Copying model files..."
cp "$SCRIPT_DIR/main.py" "$MODEL_DIR/"
cp "$SCRIPT_DIR/config.yaml" "$MODEL_DIR/"
cp "$SCRIPT_DIR/run.sh" "$MODEL_DIR/"
cp "$SCRIPT_DIR/model.py" "$MODEL_DIR/"
cp "$SCRIPT_DIR/dataset.py" "$MODEL_DIR/"

# Make run.sh executable
chmod +x "$MODEL_DIR/run.sh"

print_success "Model files copied successfully"

# Update model_names.txt
print_info "Updating model_names.txt..."
if grep -q "^BERT_DTI$" "$BATCH_RUNS_REPO/model_names.txt"; then
    print_warning "BERT_DTI already exists in model_names.txt"
else
    echo "BERT_DTI" >> "$BATCH_RUNS_REPO/model_names.txt"
    print_success "Added BERT_DTI to model_names.txt"
fi

# Check Python dependencies
print_info "Checking Python dependencies..."

python3 -c "
import sys
missing_packages = []

try:
    import torch
    print('✓ PyTorch available')
except ImportError:
    missing_packages.append('torch')

try:
    import transformers
    print('✓ Transformers available')
except ImportError:
    missing_packages.append('transformers')

try:
    import pandas
    print('✓ Pandas available')
except ImportError:
    missing_packages.append('pandas')

try:
    import yaml
    print('✓ PyYAML available')
except ImportError:
    missing_packages.append('pyyaml')

try:
    import sklearn
    print('✓ Scikit-learn available')
except ImportError:
    missing_packages.append('scikit-learn')

try:
    import tqdm
    print('✓ tqdm available')
except ImportError:
    missing_packages.append('tqdm')

try:
    import tensorboard
    print('✓ TensorBoard available')
except ImportError:
    missing_packages.append('tensorboard')

if missing_packages:
    print(f'\\n⚠ Missing packages: {missing_packages}')
    print('Install with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('\\n✓ All required packages are available')
"

if [ $? -eq 0 ]; then
    print_success "All dependencies are satisfied"
else
    print_warning "Some dependencies are missing. Install them before running the model."
fi

# Create sample test
print_info "Creating sample test script..."
cat > "$MODEL_DIR/test_integration.py" << 'EOF'
#!/usr/bin/env python3
"""
Test script for BERT-DTI batch_runs integration
"""
import os
import sys
import tempfile
import pandas as pd
import torch

def create_test_data():
    """Create minimal test data"""
    data = {
        'smiles': ['CCO', 'CC(=O)OC1=CC=CC=C1C(=O)O'],
        'sequence': ['MKKDLDFYEEIKNN', 'MLCAYHIHDQINKA'],
        'label': [0, 1]
    }
    return pd.DataFrame(data)

def test_model_loading():
    """Test if model can be loaded"""
    try:
        from model import BERTDTIModel
        config = {
            'drug_model_name': 'seyonec/PubChem10M_SMILES_BPE_450k',
            'prot_model_name': 'Rostlab/prot_bert_bfd',
            'layer_features': [768, 32, 1],
            'dropout': 0.1,
            'pretrained': {'drug': True, 'prot': True},
            'layer_limit': False
        }
        model = BERTDTIModel(config)
        print("✓ Model loading successful")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_dataset_loading():
    """Test if dataset can be loaded"""
    try:
        from dataset import BERTDTIDataset
        config = {
            'drug_model_name': 'seyonec/PubChem10M_SMILES_BPE_450k',
            'prot_model_name': 'Rostlab/prot_bert_bfd',
            'drug_max_length': 64,
            'prot_max_length': 545
        }
        data = create_test_data()
        dataset = BERTDTIDataset(data, config)
        sample = dataset[0]
        print("✓ Dataset loading successful")
        return True
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing BERT-DTI integration...")
    
    success = True
    success &= test_model_loading()
    success &= test_dataset_loading()
    
    if success:
        print("\n✓ All tests passed! Integration appears to be working.")
    else:
        print("\n✗ Some tests failed. Check error messages above.")
        sys.exit(1)
EOF

chmod +x "$MODEL_DIR/test_integration.py"

print_success "Sample test script created: $MODEL_DIR/test_integration.py"

# Create usage instructions
print_info "Creating usage instructions..."
cat > "$MODEL_DIR/USAGE.md" << 'EOF'
# BERT-DTI Model Usage in batch_runs

## Quick Test
```bash
cd models/BERT_DTI
python test_integration.py
```

## Run Single Experiment
```bash
# From batch_runs root directory
bash models/BERT_DTI/run.sh \
    /path/to/output \
    /path/to/data/split \
    /path/to/tensorboard \
    42
```

## Run All Experiments
```bash
bash scripts/run_all.sh
```

## Configuration
Edit `config.yaml` to modify model parameters.

## Data Format
Ensure your parquet files have columns: `smiles`, `sequence`, `label`
EOF

print_success "Usage instructions created: $MODEL_DIR/USAGE.md"

# Final summary
echo ""
echo "=== Deployment Complete ==="
print_success "BERT-DTI model has been deployed to batch_runs repository"
echo ""
echo "Next steps:"
echo "1. cd $BATCH_RUNS_REPO"
echo "2. Test the integration: python models/BERT_DTI/test_integration.py"
echo "3. Prepare your data in parquet format (see convert_data.py example)"
echo "4. Run experiments: bash scripts/run_all.sh"
echo ""
print_info "For detailed instructions, see:"
print_info "- $MODEL_DIR/USAGE.md"
print_info "- $(dirname $0)/INTEGRATION_GUIDE.md"
echo ""
print_success "Happy experimenting! 🚀"
