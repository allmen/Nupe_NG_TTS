import os
os.environ['TORCH_USE_CUDA'] = '1'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
import time
from tqdm import tqdm
from pathlib import Path
import platform

from model.lstm_tts import LSTMTTS
from utils.data_loader import create_dataloaders
from utils.audio_processor import AudioProcessor

class Trainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device - prioritize CUDA (GPU) over MPS (Apple) over CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = LSTMTTS(config_path).to(self.device)
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        
        # Optimizer - convert parameters to correct types
        training_config = self.config['training']
        
        # Convert string values to appropriate types
        lr = float(training_config['learning_rate'])
        weight_decay = float(training_config['weight_decay'])
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Scheduler - remove verbose parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Create directories
        self.checkpoint_dir = Path(training_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir="logs")
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Print model summary
        self.print_model_summary()
        
    def print_model_summary(self):
        """Print model architecture and parameter count"""
        print("\nModel Architecture:")
        print("-" * 50)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("-" * 50)
        
    def compute_loss(self, mel_outputs, stop_outputs, mel_targets, mel_lengths):
        """Compute total loss"""
        # Mel loss
        mel_loss = torch.tensor(0.0, device=mel_outputs.device)
        batch_size = len(mel_outputs)
        
        # Move mel_lengths to CPU for Python-level indexing
        mel_lengths_cpu = mel_lengths.cpu().tolist()
        
        for i in range(batch_size):
            actual_length = min(int(mel_lengths_cpu[i]), mel_outputs.size(1))
            if actual_length > 0:
                mel_loss += self.mse_loss(
                    mel_outputs[i, :actual_length], 
                    mel_targets[i, :actual_length]
                )
        mel_loss /= batch_size
        
        # Stop token loss
        stop_targets = torch.zeros_like(stop_outputs)
        for i, length in enumerate(mel_lengths_cpu):
            length = int(length)
            if length > 0 and length <= stop_outputs.size(1):
                stop_targets[i, length-1:] = 1.0
        
        stop_loss = self.bce_loss(stop_outputs, stop_targets)
        
        # Total loss
        total_loss = mel_loss + stop_loss
        
        return total_loss, mel_loss, stop_loss
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        mel_loss_total = 0
        stop_loss_total = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {self.current_epoch}")
        for batch in pbar:
            # Move data to device
            text = batch['text'].to(self.device)
            text_lengths = batch['text_lengths']  # Keep on CPU for pack_padded_sequence
            mel_targets = batch['mel'].to(self.device)
            mel_lengths = batch['mel_lengths']    # Keep on CPU for indexing in compute_loss
            
            # Skip empty batches
            if text.size(0) == 0:
                continue
            
            # Forward pass
            mel_outputs, stop_outputs, _ = self.model(text, text_lengths, mel_targets)
            
            # Compute loss
            loss, mel_loss, stop_loss = self.compute_loss(
                mel_outputs, stop_outputs, mel_targets, mel_lengths
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - convert to float
            grad_clip = float(self.config['training']['grad_clip'])
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                grad_clip
            )
            
            self.optimizer.step()
            
            # Save loss values before freeing tensors
            loss_val = loss.item()
            mel_loss_val = mel_loss.item()
            stop_loss_val = stop_loss.item()
            
            # Free intermediate GPU memory
            del mel_outputs, stop_outputs, loss, mel_loss, stop_loss
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Update statistics
            total_loss += loss_val
            mel_loss_total += mel_loss_val
            stop_loss_total += stop_loss_val
            batch_count += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss_val,
                'mel_loss': mel_loss_val,
                'stop_loss': stop_loss_val
            })
        
        # Average losses
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_mel_loss = mel_loss_total / batch_count
            avg_stop_loss = stop_loss_total / batch_count
        else:
            avg_loss = avg_mel_loss = avg_stop_loss = 0.0
        
        return avg_loss, avg_mel_loss, avg_stop_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        mel_loss_total = 0
        stop_loss_total = 0
        batch_count = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for batch in pbar:
                # Move data to device
                text = batch['text'].to(self.device)
                text_lengths = batch['text_lengths']  # Keep on CPU for pack_padded_sequence
                mel_targets = batch['mel'].to(self.device)
                mel_lengths = batch['mel_lengths']    # Keep on CPU for indexing in compute_loss
                
                # Skip empty batches
                if text.size(0) == 0:
                    continue
                
                # Forward pass
                mel_outputs, stop_outputs, _ = self.model(text, text_lengths, mel_targets)
                
                # Compute loss
                loss, mel_loss, stop_loss = self.compute_loss(
                    mel_outputs, stop_outputs, mel_targets, mel_lengths
                )
                
                # Update statistics
                total_loss += loss.item()
                mel_loss_total += mel_loss.item()
                stop_loss_total += stop_loss.item()
                batch_count += 1
        
        # Average losses
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            avg_mel_loss = mel_loss_total / batch_count
            avg_stop_loss = stop_loss_total / batch_count
        else:
            avg_loss = avg_mel_loss = avg_stop_loss = 0.0
        
        return avg_loss, avg_mel_loss, avg_stop_loss
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        if is_best:
            best_path = Path("models/best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['epoch']
            self.best_loss = checkpoint['best_loss']
            print(f"Loaded checkpoint from epoch {self.current_epoch}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")
    
    def train(self, train_loader, val_loader, start_epoch=0):
        """Main training loop"""
        training_config = self.config['training']
        
        # Convert string parameters to appropriate types
        epochs = int(training_config['epochs'])
        validation_interval = int(training_config['validation_interval'])
        save_interval = int(training_config['save_interval'])
        early_stopping_patience = int(training_config['early_stopping_patience'])
        
        print(f"\nStarting training for {epochs} epochs")
        print(f"Validation every {validation_interval} epochs")
        print(f"Saving checkpoints every {save_interval} epochs")
        print(f"Early stopping patience: {early_stopping_patience} epochs")
        print("-" * 50)
        
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_mel_loss, train_stop_loss = self.train_epoch(train_loader)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/train_mel', train_mel_loss, epoch)
            self.writer.add_scalar('Loss/train_stop', train_stop_loss, epoch)
            
            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Train Mel Loss: {train_mel_loss:.4f}")
            print(f"  Train Stop Loss: {train_stop_loss:.4f}")
            
            # Validate
            if (epoch + 1) % validation_interval == 0:
                val_loss, val_mel_loss, val_stop_loss = self.validate(val_loader)
                
                # Log to TensorBoard
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Loss/val_mel', val_mel_loss, epoch)
                self.writer.add_scalar('Loss/val_stop', val_stop_loss, epoch)
                
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Mel Loss: {val_mel_loss:.4f}")
                print(f"  Val Stop Loss: {val_stop_loss:.4f}")
                
                # Update learning rate with verbose output
                current_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step(val_loss)
                new_lr = self.optimizer.param_groups[0]['lr']
                
                if new_lr != current_lr:
                    print(f"  Learning rate reduced from {current_lr:.6f} to {new_lr:.6f}")
                
                # Save checkpoint
                if (epoch + 1) % save_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
                
                # Save best model
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_checkpoint("best_model.pth", is_best=True)
                    self.patience_counter = 0
                    print(f"  New best model! Loss: {val_loss:.4f}")
                else:
                    self.patience_counter += 1
                    print(f"  No improvement for {self.patience_counter} validation checks")
                
                # Early stopping
                if self.patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation loss: {self.best_loss:.4f}")
                    break
        
        self.writer.close()
        print("\nTraining completed!")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        print(f"Final model saved")

def main():
    """Main training function"""
    # Load data
    print("Loading data...")
    try:
        train_loader, val_loader, _ = create_dataloaders()
        print(f"Train batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Make sure you have run preprocess.py first")
        return
    
    # Initialize trainer
    trainer = Trainer()
    
    # Check for existing checkpoint
    checkpoint_dir = Path("models/checkpoints")
    checkpoint_path = checkpoint_dir / "latest_checkpoint.pth"
    
    if checkpoint_path.exists():
        print(f"\nLoading checkpoint from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("\nNo checkpoint found. Starting from scratch.")
    
    # Train
    print("\nStarting training...")
    try:
        trainer.train(train_loader, val_loader, start_epoch=trainer.current_epoch)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save interrupted model
        trainer.save_checkpoint("interrupted_model.pth")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()