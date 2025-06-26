# batch_run/models/BERT_DTI/model.py
"""
Simplified BERT-DTI model implementation for batch_runs integration
"""
import torch
import torch.nn as nn
import copy
from transformers import AutoConfig, RobertaModel, BertModel

def deleteEncodingLayers(model, num_layers_to_keep):
    """
    Delete encoding layers from BERT model to reduce computational complexity
    
    Args:
        model: BERT model
        num_layers_to_keep: Number of layers to keep from the beginning
    
    Returns:
        Modified model with reduced layers
    """
    oldModuleList = model.encoder.layer
    newModuleList = nn.ModuleList()

    # Keep only the first num_layers_to_keep layers
    for i in range(num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # Create a copy and modify
    copyOfModel = copy.deepcopy(model)
    copyOfModel.encoder.layer = newModuleList
    
    return copyOfModel

class BERTDTIModel(nn.Module):
    """
    BERT-DTI model for drug-target interaction prediction
    
    This model combines:
    - RoBERTa for drug (SMILES) encoding
    - BERT for protein (sequence) encoding
    - Feedforward network for interaction prediction
    """
    
    def __init__(self, config):
        super(BERTDTIModel, self).__init__()
        self.config = config
        self.best_model = None  # For storing best model state during training
        
        # Initialize drug encoder (RoBERTa)
        if config['pretrained']['drug']:
            self.drug_encoder = RobertaModel.from_pretrained(config['drug_model_name'])
        else:
            drug_config = AutoConfig.from_pretrained(config['drug_model_name'])
            self.drug_encoder = RobertaModel(drug_config)
        
        # Initialize protein encoder (BERT)
        if config['pretrained']['prot']:
            self.protein_encoder = BertModel.from_pretrained(config['prot_model_name'])
        else:
            prot_config = AutoConfig.from_pretrained(config['prot_model_name'])
            self.protein_encoder = BertModel(prot_config)
        
        # Apply layer limit to protein encoder if specified
        if config.get('layer_limit', False):
            layers_to_keep = config.get('prot_layers_keep', 18)
            self.protein_encoder = deleteEncodingLayers(self.protein_encoder, layers_to_keep)
        
        # Freeze pre-trained weights if specified
        if config.get('freeze_pretrained', {}).get('drug', False):
            for param in self.drug_encoder.parameters():
                param.requires_grad = False
        
        if config.get('freeze_pretrained', {}).get('prot', False):
            for param in self.protein_encoder.parameters():
                param.requires_grad = False
        
        # Build decoder network
        self._build_decoder()
        
    def _build_decoder(self):
        """Build the feedforward decoder network"""
        layers = []
        
        # Input dimension: concatenated drug and protein representations
        input_dim = (self.drug_encoder.config.hidden_size + 
                    self.protein_encoder.config.hidden_size)
        
        layer_features = self.config['layer_features']
        dropout_rate = self.config.get('dropout', 0.0)
        
        # Build hidden layers
        for i, feature_dim in enumerate(layer_features[:-1]):
            layers.append(nn.Linear(input_dim, feature_dim))
            input_dim = feature_dim
            
            # Activation function
            if i == len(layer_features) - 2:  # Last hidden layer
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(input_dim, layer_features[-1]))
        
        self.decoder = nn.Sequential(*layers)
    
    def encode_drug(self, input_ids, attention_mask):
        """
        Encode drug SMILES using RoBERTa
        
        Args:
            input_ids: Tokenized SMILES input IDs
            attention_mask: Attention mask for input
            
        Returns:
            Drug representation vector (CLS token)
        """
        outputs = self.drug_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0]  # CLS token
    
    def encode_protein(self, input_ids, attention_mask):
        """
        Encode protein sequence using BERT
        
        Args:
            input_ids: Tokenized protein sequence input IDs
            attention_mask: Attention mask for input
            
        Returns:
            Protein representation vector (CLS token)
        """
        outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state[:, 0]  # CLS token
    
    def forward(self, batch):
        """
        Forward pass of the model
        
        Args:
            batch: Dictionary containing:
                - drug_input_ids: Drug SMILES tokens
                - drug_attention_mask: Drug attention mask
                - prot_input_ids: Protein sequence tokens  
                - prot_attention_mask: Protein attention mask
                
        Returns:
            Interaction prediction logits
        """
        device = next(self.parameters()).device
        
        # Encode drug
        drug_repr = self.encode_drug(
            input_ids=batch['drug_input_ids'].to(device),
            attention_mask=batch['drug_attention_mask'].to(device)
        )
        
        # Encode protein
        protein_repr = self.encode_protein(
            input_ids=batch['prot_input_ids'].to(device),
            attention_mask=batch['prot_attention_mask'].to(device)
        )
        
        # Concatenate representations
        combined_repr = torch.cat([drug_repr, protein_repr], dim=1)
        
        # Predict interaction
        output = self.decoder(combined_repr)
        
        return output
    
    def get_embeddings(self, batch):
        """
        Get drug and protein embeddings without prediction
        
        Args:
            batch: Input batch
            
        Returns:
            Tuple of (drug_embeddings, protein_embeddings)
        """
        device = next(self.parameters()).device
        
        with torch.no_grad():
            drug_repr = self.encode_drug(
                input_ids=batch['drug_input_ids'].to(device),
                attention_mask=batch['drug_attention_mask'].to(device)
            )
            
            protein_repr = self.encode_protein(
                input_ids=batch['prot_input_ids'].to(device),
                attention_mask=batch['prot_attention_mask'].to(device)
            )
            
        return drug_repr.cpu(), protein_repr.cpu()

    def save_best_model(self, state_dict, epoch):
        """Save the best model state"""
        self.best_model = (state_dict, epoch)
    
    def load_best_model(self):
        """Load the best model state"""
        if self.best_model is not None:
            self.load_state_dict(self.best_model[0])
            return self.best_model[1]
        return None
