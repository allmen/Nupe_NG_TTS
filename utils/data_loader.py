import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import platform
from tqdm import tqdm
import yaml
from utils.audio_processor import AudioProcessor
from utils.text_processor import NupeTextProcessor

class NupeTTSDataset(Dataset):
    def __init__(self, metadata_path, config_path="config/config.yaml"):
        super().__init__()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.metadata = pd.read_csv(metadata_path)
        self.audio_processor = AudioProcessor(config_path)
        self.text_processor = NupeTextProcessor(config_path)
        
        # Cache features
        self.features_cache = {}
        self.cache_dir = "data/processed/features/"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Max sequence lengths to prevent OOM
        data_config = self.config.get('data', {})
        self.max_text_length = data_config.get('max_text_length', 200)
        self.max_audio_length = data_config.get('max_audio_length', 200)
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load or extract features
        audio_path = row['audio_path']
        text = row['text']
        speaker_id = row['speaker_id']
        
        # Generate cache key
        cache_key = f"{speaker_id}_{os.path.basename(audio_path).split('.')[0]}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npz")
        
        if os.path.exists(cache_path):
            # Load from cache
            data = np.load(cache_path, allow_pickle=True)
            mel = data['mel']
        else:
            # Extract features
            features = self.audio_processor.extract_features(audio_path)
            mel = features['mel']
            
            # Save to cache
            np.savez(cache_path, mel=mel)
        
        # Process text
        text_seq = self.text_processor.text_to_sequence(text)
        
        # Convert to tensors - truncate to max lengths to prevent OOM
        mel = mel[:self.max_audio_length, :]
        text_seq = text_seq[:self.max_text_length]
        
        mel_tensor = torch.FloatTensor(mel)
        text_tensor = torch.LongTensor(text_seq)
        
        # Get speaker embedding
        speaker_tensor = torch.LongTensor([int(speaker_id.split('_')[-1])])
        
        return {
            'text': text_tensor,
            'mel': mel_tensor,
            'speaker': speaker_tensor,
            'text_length': len(text_seq),
            'mel_length': mel.shape[0],
            'audio_path': audio_path
        }
    
    def collate_fn(self, batch):
        """Custom collate function for variable length sequences"""
        # Sort batch by text length (descending)
        batch = sorted(batch, key=lambda x: x['text_length'], reverse=True)
        
        # Get max lengths
        max_text_len = max(x['text_length'] for x in batch)
        max_mel_len = max(x['mel_length'] for x in batch)
        
        # Initialize tensors
        text_batch = torch.zeros(len(batch), max_text_len, dtype=torch.long)
        mel_batch = torch.zeros(len(batch), max_mel_len, 
                               self.config['audio']['n_mels'], dtype=torch.float)
        speaker_batch = torch.zeros(len(batch), dtype=torch.long)
        text_lengths = torch.zeros(len(batch), dtype=torch.long)
        mel_lengths = torch.zeros(len(batch), dtype=torch.long)
        
        # Fill tensors
        for i, item in enumerate(batch):
            text_len = item['text_length']
            mel_len = item['mel_length']
            
            text_batch[i, :text_len] = item['text']
            mel_batch[i, :mel_len, :] = item['mel']
            speaker_batch[i] = item['speaker']
            text_lengths[i] = text_len
            mel_lengths[i] = mel_len
        
        return {
            'text': text_batch,
            'mel': mel_batch,
            'speaker': speaker_batch,
            'text_lengths': text_lengths,
            'mel_lengths': mel_lengths
        }

def create_dataloaders(config_path="config/config.yaml"):
    """Create train, validation, and test dataloaders"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets
    train_dataset = NupeTTSDataset("data/splits/train.txt", config_path)
    val_dataset = NupeTTSDataset("data/splits/val.txt", config_path)
    test_dataset = NupeTTSDataset("data/splits/test.txt", config_path)
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    
    # On Windows, num_workers > 0 can cause multiprocessing issues
    num_workers = 0 if platform.system() == 'Windows' else 4
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader