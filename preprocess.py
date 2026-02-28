import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import json

def collect_metadata(data_dir="data/raw"):
    """Collect all metadata from speaker folders"""
    metadata = []
    
    for speaker_dir in os.listdir(data_dir):
        speaker_path = os.path.join(data_dir, speaker_dir)
        if not os.path.isdir(speaker_path):
            continue
        
        # Get speaker info from directory name
        speaker_id = speaker_dir
        gender = "Unknown"
        age = "Unknown"
        
        # Check each file in speaker directory
        for file in os.listdir(speaker_path):
            if file.endswith('.txt'):
                txt_path = os.path.join(speaker_path, file)
                wav_file = file.replace('.txt', '.wav')
                wav_path = os.path.join(speaker_path, wav_file)
                
                # Read text
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if os.path.exists(wav_path):
                    metadata.append({
                        'speaker_id': speaker_id,
                        'text': text,
                        'text_path': txt_path,
                        'audio_path': wav_path,
                        'gender': gender,
                        'age': age
                    })
    
    return pd.DataFrame(metadata)

def create_splits(metadata, config_path="config/config.yaml"):
    """Create train/val/test splits"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    splits = config['data']
    
    # Split by speaker to avoid data leakage
    speakers = metadata['speaker_id'].unique()
    train_speakers, temp_speakers = train_test_split(
        speakers, 
        test_size=splits['val_split'] + splits['test_split'],
        random_state=42
    )
    
    val_speakers, test_speakers = train_test_split(
        temp_speakers,
        test_size=splits['test_split'] / (splits['val_split'] + splits['test_split']),
        random_state=42
    )
    
    # Create splits
    train_df = metadata[metadata['speaker_id'].isin(train_speakers)]
    val_df = metadata[metadata['speaker_id'].isin(val_speakers)]
    test_df = metadata[metadata['speaker_id'].isin(test_speakers)]
    
    # Save splits
    os.makedirs("data/splits", exist_ok=True)
    train_df.to_csv("data/splits/train.txt", index=False)
    val_df.to_csv("data/splits/val.txt", index=False)
    test_df.to_csv("data/splits/test.txt", index=False)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df

def analyze_dataset(metadata):
    """Analyze the dataset statistics"""
    print("Dataset Analysis:")
    print(f"Total samples: {len(metadata)}")
    print(f"Number of speakers: {metadata['speaker_id'].nunique()}")
    print(f"Gender distribution:\n{metadata['gender'].value_counts()}")
    
    # Text length analysis
    text_lengths = metadata['text'].apply(len)
    print(f"\nText length statistics:")
    print(f"Min length: {text_lengths.min()}")
    print(f"Max length: {text_lengths.max()}")
    print(f"Mean length: {text_lengths.mean():.2f}")
    print(f"Std length: {text_lengths.std():.2f}")
    
    # Save statistics
    stats = {
        'total_samples': len(metadata),
        'num_speakers': metadata['speaker_id'].nunique(),
        'gender_distribution': metadata['gender'].value_counts().to_dict(),
        'text_length_stats': {
            'min': int(text_lengths.min()),
            'max': int(text_lengths.max()),
            'mean': float(text_lengths.mean()),
            'std': float(text_lengths.std())
        }
    }
    
    with open("data/processed/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

def main():
    """Main preprocessing function"""
    print("Starting preprocessing...")
    
    # Collect metadata
    print("Collecting metadata...")
    metadata = collect_metadata("data/raw")
    metadata.to_csv("data/processed/metadata.csv", index=False)
    
    # Analyze dataset
    print("Analyzing dataset...")
    analyze_dataset(metadata)
    
    # Create splits
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(metadata)
    
    # Clean text
    print("Cleaning text...")
    from utils.text_processor import NupeTextProcessor
    text_processor = NupeTextProcessor()
    
    os.makedirs("data/processed/text_clean", exist_ok=True)
    
    for split_df, split_name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
        split_df['text_clean'] = split_df['text'].apply(text_processor.clean_text)
        
        # Save cleaned text files
        for idx, row in split_df.iterrows():
            clean_text_path = f"data/processed/text_clean/{split_name}_{idx}.txt"
            with open(clean_text_path, 'w', encoding='utf-8') as f:
                f.write(row['text_clean'])
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()