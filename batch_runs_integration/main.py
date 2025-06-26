# batch_run/models/BERT_DTI/main.py
import argparse
import json
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard
import torch.utils.data
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, \
    roc_curve, confusion_matrix, precision_recall_curve, precision_score, f1_score
import datetime
from tqdm import tqdm
import numpy as np

DEBUG = False

# Import transformers at runtime to avoid import issues
try:
    from transformers import AutoTokenizer, AutoConfig, RobertaModel, BertModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available. Please install with: pip install transformers")
    TRANSFORMERS_AVAILABLE = False

# remove tensorflow warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def print_with_time(*arg, **args):
    print(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"), *arg, **args)

def deleteEncodingLayers(model, num_layers_to_keep):
    """Delete encoding layers to reduce model size"""
    import copy
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList
    return copyOfModel

class BERTDTIDataset(torch.utils.data.Dataset):
    def __init__(self, data, config):
        data.reset_index(drop=True, inplace=True)
        self.data = data
        self.config = config
        
        # Initialize tokenizers
        self.d_tokenizer = AutoTokenizer.from_pretrained(config['drug_model_name'])
        self.p_tokenizer = AutoTokenizer.from_pretrained(config['prot_model_name'])
        
        print_with_time("Dataset preprocessing done.")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        smiles = self.data.iloc[idx]['smiles']
        sequence = self.data.iloc[idx]['sequence']
        label = self.data.iloc[idx]['label']
        
        # Tokenize drug (SMILES)
        d_inputs = self.d_tokenizer(
            smiles, 
            padding='max_length', 
            max_length=self.config['drug_max_length'], 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Tokenize protein (add spaces between amino acids)
        prot_sequence = ' '.join(list(sequence))
        p_inputs = self.p_tokenizer(
            prot_sequence, 
            padding='max_length', 
            max_length=self.config['prot_max_length'], 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            'drug_input_ids': d_inputs['input_ids'].squeeze(),
            'drug_attention_mask': d_inputs['attention_mask'].squeeze(),
            'prot_input_ids': p_inputs['input_ids'].squeeze(),
            'prot_attention_mask': p_inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

class BERTDTIModel(nn.Module):
    def __init__(self, config):
        super(BERTDTIModel, self).__init__()
        self.config = config
        self.best_model = None
        
        # Initialize drug model (RoBERTa)
        if config['pretrained']['drug']:
            self.d_model = RobertaModel.from_pretrained(config['drug_model_name'])
        else:
            drug_config = AutoConfig.from_pretrained(config['drug_model_name'])
            self.d_model = RobertaModel(drug_config)
        
        # Initialize protein model (BERT)
        if config['pretrained']['prot']:
            self.p_model = BertModel.from_pretrained(config['prot_model_name'])
        else:
            prot_config = AutoConfig.from_pretrained(config['prot_model_name'])
            self.p_model = BertModel(prot_config)
        
        # Apply layer limit if specified
        if config.get('layer_limit', False):
            self.p_model = deleteEncodingLayers(self.p_model, config.get('prot_layers_keep', 18))
        
        # Freeze pre-trained weights if specified
        if config.get('freeze_pretrained', {}).get('drug', False):
            for param in self.d_model.parameters():
                param.requires_grad = False
            print_with_time("Drug model weights frozen")
        
        if config.get('freeze_pretrained', {}).get('prot', False):
            for param in self.p_model.parameters():
                param.requires_grad = False
            print_with_time("Protein model weights frozen")
        
        # Decoder layers
        layers = []
        input_dim = self.d_model.config.hidden_size + self.p_model.config.hidden_size
        layer_features = config['layer_features']
        
        for i, feature_dim in enumerate(layer_features[:-1]):
            layers.append(nn.Linear(input_dim, feature_dim))
            input_dim = feature_dim
            
            if i == len(layer_features) - 2:  # Last hidden layer
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            if config['dropout'] > 0:
                layers.append(nn.Dropout(config['dropout']))
        
        layers.append(nn.Linear(input_dim, layer_features[-1]))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, batch):
        device = next(self.parameters()).device
        
        # Drug encoding
        drug_outputs = self.d_model(
            input_ids=batch['drug_input_ids'].to(device),
            attention_mask=batch['drug_attention_mask'].to(device)
        )
        
        # Protein encoding
        prot_outputs = self.p_model(
            input_ids=batch['prot_input_ids'].to(device),
            attention_mask=batch['prot_attention_mask'].to(device)
        )
        
        # Concatenate [CLS] tokens
        combined = torch.cat((
            drug_outputs.last_hidden_state[:, 0],  # [CLS] token
            prot_outputs.last_hidden_state[:, 0]   # [CLS] token
        ), dim=1)
        
        # Decode
        output = self.decoder(combined)
        return output

def train_and_evaluate(train_data, val_data, test_data, config):
    print_with_time("Training model with config:", json.dumps(config, indent=4))
    print_with_time(f"metrics: {config['train_params']['metrics']}")
    
    # Set random seed
    random.seed(config['random_seed'])
    torch.manual_seed(config['random_seed'])
    
    # Initialize tensorboard
    writer = None
    if args.tensorboard_logdir:
        writer = tensorboard.SummaryWriter(args.tensorboard_logdir)
    
    # Initialize model
    model = BERTDTIModel(config)
    if config['train_params']['loss_function'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:  # smooth_l1
        criterion = nn.SmoothL1Loss()
    
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=float(config['train_params']['learning_rate'])
    )
    
    device = torch.device(config['train_params']['device'])
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Create datasets and dataloaders
    train_dataset = BERTDTIDataset(train_data, config)
    val_dataset = BERTDTIDataset(val_data, config)
    test_dataset = BERTDTIDataset(test_data, config)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['train_params']['batch_size'], 
        shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config['train_params']['batch_size'], 
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config['train_params']['batch_size'], 
        shuffle=False
    )
    
    # Train
    if writer:
        # Add model graph (if possible)
        try:
            sample_batch = next(iter(train_loader))
            writer.add_graph(model, sample_batch)
        except:
            pass
    
    train(model, criterion, optimizer, train_loader, val_loader, config, writer)
    
    # Evaluate
    metrics = evaluate(model, test_loader, config)
    
    if writer:
        writer.close()
    
    return metrics

def train(model, criterion, optimizer, train_loader, val_loader, config, writer=None):
    device = torch.device(config['train_params']['device'])
    model.best_model = (model.state_dict(), 0)
    best_loss = float('inf')
    
    progress = tqdm(total=config['train_params']['num_epochs'], desc="Training")
    
    for epoch in range(config['train_params']['num_epochs']):
        model.train()
        epoch_loss = 0
        
        # Add tqdm for batch progress within epoch
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['train_params']['num_epochs']}", leave=False)
        
        for i, batch in enumerate(batch_progress):
            optimizer.zero_grad()
            output = model(batch)
            
            labels = batch['label'].view(-1, 1).float().to(device)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Update batch progress bar with current loss
            batch_progress.set_postfix(batch_loss=loss.item())
        
        avg_loss = epoch_loss / len(train_loader)
        
        # Validation
        if epoch % config['train_params']['val_interval'] == 0:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    output = model(batch)
                    labels = batch['label'].view(-1, 1).float().to(device)
                    val_loss += criterion(output, labels).item()
                
                val_loss /= len(val_loader)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    model.best_model = (model.state_dict(), epoch)
        
        progress.set_postfix(loss=avg_loss, val_loss=val_loss if 'val_loss' in locals() else 0)
        progress.update(1)
        
        # Tensorboard logging
        if writer:
            writer.add_scalar('Loss/train', avg_loss, epoch)
            if 'val_loss' in locals():
                writer.add_scalar('Loss/val', val_loss, epoch)
    
    print_with_time("Training done.")
    progress.close()

def evaluate(model, test_loader, config):
    print_with_time("Evaluating model...")
    print_with_time("Best model from epoch:", model.best_model[1])
    
    # Load best model
    model.load_state_dict(model.best_model[0])
    model.eval()
    
    device = torch.device(config['train_params']['device'])
    
    with torch.no_grad():
        y_true = []
        y_pred = []
        
        for batch in test_loader:
            output = model(batch)
            
            if config['train_params']['loss_function'] == 'bce':
                # Apply sigmoid for BCEWithLogitsLoss
                output = torch.sigmoid(output)
            
            y_true.extend(batch['label'].tolist())
            y_pred.extend(output.cpu().squeeze().tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    metrics = {}
    
    # Calculate metrics
    for metric in config['train_params']['metrics']:
        if metric == 'auroc':
            metrics['auroc'] = roc_auc_score(y_true, y_pred)
        elif metric == 'auprc':
            metrics['auprc'] = average_precision_score(y_true, y_pred)
        elif metric == 'precision':
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['precision'] = precision_score(y_true, y_pred_binary)
        elif metric == 'f1':
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['f1'] = f1_score(y_true, y_pred_binary)
        elif metric == 'accuracy':
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['accuracy'] = np.mean(y_true == y_pred_binary)
        elif metric == 'mse':
            metrics['mse'] = np.mean((y_true - y_pred) ** 2)
    
    print_with_time("Metrics:", metrics)
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--tensorboard_logdir', type=str, required=False)
    parser.add_argument('--seed', type=int, default=-1)
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    if args.seed != -1:
        config['random_seed'] = args.seed
    
    # Load datasets
    train_data = pd.read_parquet(args.train_path)
    val_data = pd.read_parquet(args.val_path)
    test_data = pd.read_parquet(args.test_path)
    
    # Train and evaluate
    metrics = train_and_evaluate(train_data, val_data, test_data, config)
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("Metrics saved to {}".format(os.path.join(args.output_dir, "metrics.json")))
