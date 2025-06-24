# Example: How to prepare your data for batch_runs integration

import pandas as pd
import numpy as np

def convert_bertdti_to_batchruns_format(input_csv_path, output_parquet_path):
    """
    Convert your existing BERT-DTI data format to batch_runs compatible format
    
    Args:
        input_csv_path: Path to your existing CSV file
        output_parquet_path: Path where to save the parquet file
    """
    
    # Load your existing data
    df = pd.read_csv(input_csv_path)
    
    # Expected columns in batch_runs format
    required_columns = ['smiles', 'sequence', 'label']
    
    # Map your column names to batch_runs format
    # Adjust these mappings based on your actual column names
    column_mapping = {
        'SMILES': 'smiles',           # Your SMILES column -> 'smiles'
        'Target Sequence': 'sequence', # Your protein sequence column -> 'sequence'  
        'Y': 'label'                  # Your label column -> 'label'
    }
    
    # Rename columns
    df_converted = df.rename(columns=column_mapping)
    
    # Keep only required columns
    df_converted = df_converted[required_columns]
    
    # Data cleaning
    df_converted = df_converted.dropna()  # Remove rows with missing values
    
    # Ensure label is numeric
    df_converted['label'] = pd.to_numeric(df_converted['label'], errors='coerce')
    df_converted = df_converted.dropna()  # Remove rows where label conversion failed
    
    # Basic validation
    print(f"Original data: {len(df)} rows")
    print(f"Converted data: {len(df_converted)} rows")
    print(f"Columns: {list(df_converted.columns)}")
    print(f"Label distribution:\n{df_converted['label'].value_counts()}")
    
    # Save as parquet
    df_converted.to_parquet(output_parquet_path, index=False)
    print(f"Data saved to: {output_parquet_path}")
    
    return df_converted

def create_sample_data():
    """Create sample data in batch_runs format for testing"""
    
    # Sample SMILES strings (these are real drug SMILES)
    sample_smiles = [
        "CCO",  # Ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC1=C(C(=O)N(N1C)C2=CC=CC=C2)O",  # Phenylbutazone
    ]
    
    # Sample protein sequences (shortened for example)
    sample_sequences = [
        "MKKDLDFYEEIKNNALRNAGWYTVEAANNPVLGGDVRRGSEFWQTLKKIDREKLFSQGDFTLSAPDFKLYGGEEQPALLG",
        "MLCAYHIHDQINKAGLMSQTLTVFGFGGKTSLQKKQLSPPDFNPLLYHAGLMKGAHREMSKEEVYTGHKGLTPLTSFPHQ",
        "MSNTADKGQLLCACYSEQHKLLQGRPTSRVARLRELAEKKVDCDCGWYGGGGGMQTQHVRLGWGAGGCRVTNFTQSLLTP",
        "MQKLFGKGRGTLGLGTTGFTAFALPPQGASIGDLGALFPYGPPHVHRDLKGELLFFTNRLSAQDLLLRVFYCDGNSLLEL"
    ]
    
    # Create sample data
    data = []
    for i, (smiles, sequence) in enumerate(zip(sample_smiles, sample_sequences)):
        for label in [0, 1]:  # Create both positive and negative examples
            data.append({
                'smiles': smiles,
                'sequence': sequence,
                'label': label
            })
    
    df = pd.DataFrame(data)
    
    # Save sample data
    df.to_parquet('sample_dti_data.parquet', index=False)
    print("Sample data created: sample_dti_data.parquet")
    print(f"Shape: {df.shape}")
    print(df.head())
    
    return df

if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data()
    
    # Example of converting existing data (uncomment and modify as needed)
    # convert_bertdti_to_batchruns_format(
    #     input_csv_path="your_existing_data.csv",
    #     output_parquet_path="converted_data.parquet"
    # )
