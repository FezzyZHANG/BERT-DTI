# batch_run/models/BERT_DTI/dataset.py
"""
Dataset class for BERT-DTI model compatible with batch_runs framework
"""
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import pandas as pd
import numpy as np

class BERTDTIDataset(Dataset):
    """
    Dataset class for drug-target interaction prediction using BERT-DTI
    
    Expected data format (parquet file):
    - smiles: Drug SMILES string
    - sequence: Protein amino acid sequence  
    - label: Interaction label (0/1 for classification, continuous for regression)
    """
    
    def __init__(self, data, config):
        """
        Initialize dataset
        
        Args:
            data: pandas DataFrame with columns [smiles, sequence, label]
            config: Configuration dictionary containing model parameters
        """
        # Reset index to ensure proper indexing
        data = data.reset_index(drop=True)
        self.data = data
        self.config = config
        
        # Validate required columns
        required_columns = ['smiles', 'sequence', 'label']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Initialize tokenizers
        self.drug_tokenizer = AutoTokenizer.from_pretrained(config['drug_model_name'])
        self.protein_tokenizer = AutoTokenizer.from_pretrained(config['prot_model_name'])
        
        # Get configuration parameters
        self.drug_max_length = config.get('drug_max_length', 64)
        self.prot_max_length = config.get('prot_max_length', 545)
        
        print(f"Dataset initialized with {len(self.data)} samples")
        print(f"Drug max length: {self.drug_max_length}")
        print(f"Protein max length: {self.prot_max_length}")
        
    def __len__(self):
        """Return the size of the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single data sample
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing tokenized inputs and label
        """
        # Get raw data
        row = self.data.iloc[idx]
        smiles = row['smiles']
        sequence = row['sequence']
        label = row['label']
        
        # Tokenize drug (SMILES)
        drug_encoding = self._tokenize_drug(smiles)
        
        # Tokenize protein (amino acid sequence)
        protein_encoding = self._tokenize_protein(sequence)
        
        return {
            'drug_input_ids': drug_encoding['input_ids'].squeeze(),
            'drug_attention_mask': drug_encoding['attention_mask'].squeeze(),
            'prot_input_ids': protein_encoding['input_ids'].squeeze(),
            'prot_attention_mask': protein_encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float32)
        }
    
    def _tokenize_drug(self, smiles):
        """
        Tokenize drug SMILES string
        
        Args:
            smiles: SMILES string representation of the drug
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        encoding = self.drug_tokenizer(
            smiles,
            padding='max_length',
            max_length=self.drug_max_length,
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def _tokenize_protein(self, sequence):
        """
        Tokenize protein amino acid sequence
        
        Args:
            sequence: Amino acid sequence string
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Add spaces between amino acids for BERT tokenization
        spaced_sequence = ' '.join(list(sequence))
        
        encoding = self.protein_tokenizer(
            spaced_sequence,
            padding='max_length',
            max_length=self.prot_max_length,
            truncation=True,
            return_tensors='pt'
        )
        return encoding
    
    def get_sample_info(self, idx):
        """
        Get human-readable information about a sample
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with sample information
        """
        row = self.data.iloc[idx]
        return {
            'index': idx,
            'smiles': row['smiles'],
            'sequence': row['sequence'][:50] + '...' if len(row['sequence']) > 50 else row['sequence'],
            'sequence_length': len(row['sequence']),
            'label': row['label']
        }
    
    def get_statistics(self):
        """
        Get dataset statistics
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'total_samples': len(self.data),
            'label_distribution': self.data['label'].value_counts().to_dict(),
            'sequence_length_stats': {
                'mean': self.data['sequence'].str.len().mean(),
                'std': self.data['sequence'].str.len().std(),
                'min': self.data['sequence'].str.len().min(),
                'max': self.data['sequence'].str.len().max()
            },
            'smiles_length_stats': {
                'mean': self.data['smiles'].str.len().mean(),
                'std': self.data['smiles'].str.len().std(),
                'min': self.data['smiles'].str.len().min(),
                'max': self.data['smiles'].str.len().max()
            }
        }
        return stats

def collate_fn(batch):
    """
    Custom collate function for DataLoader
    
    Args:
        batch: List of samples from the dataset
        
    Returns:
        Batched tensors
    """
    # Stack all tensors
    drug_input_ids = torch.stack([item['drug_input_ids'] for item in batch])
    drug_attention_mask = torch.stack([item['drug_attention_mask'] for item in batch])
    prot_input_ids = torch.stack([item['prot_input_ids'] for item in batch])
    prot_attention_mask = torch.stack([item['prot_attention_mask'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'drug_input_ids': drug_input_ids,
        'drug_attention_mask': drug_attention_mask,
        'prot_input_ids': prot_input_ids,
        'prot_attention_mask': prot_attention_mask,
        'label': labels
    }

def create_dataloader(dataset, batch_size, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset
    
    Args:
        dataset: BERTDTIDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )
