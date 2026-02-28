import re
import unicodedata
from unidecode import unidecode
import yaml
import torch

class NupeTextProcessor:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        text_config = config['text']
        self.use_phonemes = text_config['use_phonemes']
        self.add_start_end = text_config['add_start_end_tokens']
        
        # Define Nupe-specific characters
        self.nupe_characters = "abcdefghijklmnopqrstuvwxyzàáèéìíòóùúǹḿṑṹẁỳ"
        self.nupe_characters += "ABCDEFGHIJKLMNOPQRSTUVWXYZÀÁÈÉÌÍÒÓÙÚǸḾṐṹẀỲ"
        self.nupe_characters += "0123456789!?,.:;-'\" "
        
        # Create character to index mapping
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.nupe_characters)}
        self.char_to_idx['<pad>'] = 0
        self.char_to_idx['<sos>'] = len(self.char_to_idx)
        self.char_to_idx['<eos>'] = len(self.char_to_idx)
        
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
    
    def clean_text(self, text):
        """Clean Nupe text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFC', text)
        
        # Keep only Nupe characters
        text = ''.join(char for char in text if char in self.nupe_characters)
        
        return text.strip()
    
    def text_to_sequence(self, text):
        """Convert text to sequence of indices"""
        text = self.clean_text(text)
        
        sequence = []
        if self.add_start_end:
            sequence.append(self.char_to_idx['<sos>'])
        
        sequence.extend(
            self.char_to_idx.get(char, self.char_to_idx['<pad>'])
            for char in text
        )
        
        if self.add_start_end:
            sequence.append(self.char_to_idx['<eos>'])
        
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        text = ''.join(self.idx_to_char[idx] for idx in sequence 
                      if idx in self.idx_to_char and self.idx_to_char[idx] not in ['<pad>', '<sos>', '<eos>'])
        return text
    
    def get_phonemes(self, text):
        """Convert text to phonemes (if available)"""
        # For Nupe, you might need a custom phonemizer
        # This is a placeholder for actual phoneme conversion
        try:
            from phonemizer import phonemize
            phonemes = phonemize(
                text,
                language='en',  # Replace with Nupe if available
                backend='espeak',
                strip=True,
                preserve_punctuation=True
            )
            return phonemes
        except:
            return text
    
    def create_embedding_matrix(self, embedding_dim=256):
        """Create embedding matrix for text"""
        return torch.nn.Embedding(
            self.vocab_size,
            embedding_dim,
            padding_idx=self.char_to_idx['<pad>']
        )