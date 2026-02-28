"""
How to use this:
    python evaluate.py --checkpoint models/best_model.pth --num_samples 20 --output_dir evaluation_results
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.lstm_tts import LSTMTTS
from utils.text_processor import NupeTextProcessor
from utils.audio_processor import AudioProcessor
from utils.data_loader import NupeTTSDataset


def mel_cepstral_distortion(pred_mel, target_mel):
    min_len = min(len(pred_mel), len(target_mel))
    pred = pred_mel[:min_len]
    target = target_mel[:min_len]
    from scipy.fftpack import dct
    n_mfcc = 13
    pred_mfcc = dct(pred, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    target_mfcc = dct(target, type=2, axis=1, norm='ortho')[:, :n_mfcc]
    diff = pred_mfcc[:, 1:] - target_mfcc[:, 1:]
    mcd = np.mean(np.sqrt(2 * np.sum(diff ** 2, axis=1)))
    mcd_db = (10.0 / np.log(10)) * mcd
    return mcd_db


def compute_rmse(pred_mel, target_mel):
    min_len = min(len(pred_mel), len(target_mel))
    pred = pred_mel[:min_len]
    target = target_mel[:min_len]
    
    rmse = np.sqrt(np.mean((pred - target) ** 2))
    return rmse


def compute_dtw_distance(pred_mel, target_mel):
    cost = cdist(pred_mel, target_mel, metric='euclidean')
    m, n = cost.shape
    dtw = np.full((m + 1, n + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dtw[i, j] = cost[i-1, j-1] + min(
                dtw[i-1, j],    # insertion
                dtw[i, j-1],    # deletion
                dtw[i-1, j-1]   # match
            )
    path_length = m + n
    dtw_normalized = dtw[m, n] / path_length
    
    return dtw_normalized


def compute_duration_ratio(pred_mel, target_mel):
    """
    Compute duration ratio: predicted length / target length.
    Ratio close to 1.0 = good duration prediction.
    """
    return len(pred_mel) / len(target_mel) if len(target_mel) > 0 else 0


def compute_attention_diagonal_score(attention_weights):
    """
    Args:
        attention_weights: Attention matrix (decoder_steps, encoder_steps)
    Returns:
        Diagonal score (0-1, higher is better)
    """
    if attention_weights is None:
        return 0.0
    
    T_dec, T_enc = attention_weights.shape
    
    # Create ideal diagonal attention
    ideal = np.zeros_like(attention_weights)
    for i in range(T_dec):
        j = int(i * T_enc / T_dec)
        if j < T_enc:
            ideal[i, j] = 1.0
    
    # Compute correlation with ideal diagonal
    flat_pred = attention_weights.flatten()
    flat_ideal = ideal.flatten()
    
    if np.std(flat_pred) == 0 or np.std(flat_ideal) == 0:
        return 0.0
    
    correlation, _ = stats.pearsonr(flat_pred, flat_ideal)
    
    # Also compute focus metric (how peaked the attention is per step)
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-8), axis=1)
    focus_score = 1.0 - np.mean(entropy) / np.log(T_enc)  # Normalized
    
    # Combined score
    diagonal_score = 0.5 * max(0, correlation) + 0.5 * max(0, focus_score)
    
    return diagonal_score


def compute_spectral_convergence(pred_mel, target_mel):
    """
    Lower is better.
    """
    min_len = min(len(pred_mel), len(target_mel))
    pred = pred_mel[:min_len]
    target = target_mel[:min_len]
    
    sc = np.linalg.norm(target - pred, 'fro') / np.linalg.norm(target, 'fro')
    return sc


def compute_log_spectral_distance(pred_mel, target_mel):
    min_len = min(len(pred_mel), len(target_mel))
    pred = pred_mel[:min_len]
    target = target_mel[:min_len]
    
    # Add small epsilon to avoid log(0)
    pred_log = np.log10(np.maximum(pred, 1e-10))
    target_log = np.log10(np.maximum(target, 1e-10))
    
    lsd = np.mean(np.sqrt(np.mean((pred_log - target_log) ** 2, axis=1)))
    return lsd


class TTSEvaluator:
    def __init__(self, checkpoint_path, config_path="config/config.yaml"):
        print("=" * 60)
        print("TTS Evaluation System")
        print("=" * 60)
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Device: {self.device}")
        
        # Load model
        self.model = LSTMTTS(config_path).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"Model loaded from: {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        
        # Processors
        self.text_processor = NupeTextProcessor(config_path)
        self.audio_processor = AudioProcessor(config_path)
        
    def synthesize(self, text):
        """Generate mel spectrogram from text."""
        text_seq = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(self.device)
        text_length = torch.LongTensor([len(text_seq)])  # CPU for pack_padded_sequence
        
        with torch.no_grad():
            mel_output, stop_output, attention = self.model(text_tensor, text_length)
        
        mel = mel_output.squeeze(0).cpu().numpy()
        attn = attention.squeeze(0).cpu().numpy() if attention is not None else None
        
        return mel, attn
    
    def evaluate_sample(self, text, target_mel):
        """Evaluate a single sample."""
        pred_mel, attention = self.synthesize(text)
        
        metrics = {
            'mcd': mel_cepstral_distortion(pred_mel, target_mel),
            'rmse': compute_rmse(pred_mel, target_mel),
            'dtw': compute_dtw_distance(pred_mel, target_mel),
            'duration_ratio': compute_duration_ratio(pred_mel, target_mel),
            'spectral_convergence': compute_spectral_convergence(pred_mel, target_mel),
            'log_spectral_distance': compute_log_spectral_distance(pred_mel, target_mel),
            'attention_diagonal_score': compute_attention_diagonal_score(attention),
            'pred_length': len(pred_mel),
            'target_length': len(target_mel),
        }
        
        return metrics, pred_mel, attention
    
    def evaluate_dataset(self, test_dataset, num_samples=None, output_dir="evaluation_results"):
        """Evaluate on test dataset."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Subfolders
        (output_dir / "spectrograms").mkdir(exist_ok=True)
        (output_dir / "attention").mkdir(exist_ok=True)
        
        all_metrics = []
        n = min(num_samples, len(test_dataset)) if num_samples else len(test_dataset)
        
        print(f"\nEvaluating {n} samples...")
        
        for i in tqdm(range(n), desc="Evaluating"):
            sample = test_dataset[i]
            text_seq = sample['text'].numpy()
            target_mel = sample['mel'].numpy()
            
            # Get original text
            text = self.text_processor.sequence_to_text(text_seq.tolist())
            
            try:
                metrics, pred_mel, attention = self.evaluate_sample(text, target_mel)
                metrics['sample_id'] = i
                metrics['text'] = text[:50]  # Truncate for display
                all_metrics.append(metrics)
                
                # Save comparison plots for first 10 samples
                if i < 10:
                    self.plot_comparison(
                        pred_mel, target_mel, attention, text,
                        output_dir / "spectrograms" / f"sample_{i}.png",
                        output_dir / "attention" / f"attention_{i}.png"
                    )
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        # Aggregate results
        df = pd.DataFrame(all_metrics)
        
        # Summary statistics
        summary = {
            'num_samples': len(df),
            'mcd_mean': float(df['mcd'].mean()),
            'mcd_std': float(df['mcd'].std()),
            'rmse_mean': float(df['rmse'].mean()),
            'rmse_std': float(df['rmse'].std()),
            'dtw_mean': float(df['dtw'].mean()),
            'dtw_std': float(df['dtw'].std()),
            'duration_ratio_mean': float(df['duration_ratio'].mean()),
            'duration_ratio_std': float(df['duration_ratio'].std()),
            'spectral_convergence_mean': float(df['spectral_convergence'].mean()),
            'log_spectral_distance_mean': float(df['log_spectral_distance'].mean()),
            'attention_score_mean': float(df['attention_diagonal_score'].mean()),
        }
        
        # Save results
        df.to_csv(output_dir / "detailed_metrics.csv", index=False)
        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Generate summary charts
        self.plot_metrics_summary(df, output_dir)
        self.plot_metric_distributions(df, output_dir)
        
        return df, summary
    
    def plot_comparison(self, pred_mel, target_mel, attention, text, spec_path, attn_path):
        """Plot predicted vs target mel spectrogram comparison."""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Target mel
        im1 = axes[0].imshow(target_mel.T, aspect='auto', origin='lower', cmap='magma')
        axes[0].set_title('Ground Truth Mel-Spectrogram')
        axes[0].set_ylabel('Mel Bands')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted mel
        im2 = axes[1].imshow(pred_mel.T, aspect='auto', origin='lower', cmap='magma')
        axes[1].set_title('Predicted Mel-Spectrogram')
        axes[1].set_xlabel('Time Frames')
        axes[1].set_ylabel('Mel Bands')
        plt.colorbar(im2, ax=axes[1])
        
        plt.suptitle(f'Text: "{text[:60]}..."' if len(text) > 60 else f'Text: "{text}"')
        plt.tight_layout()
        plt.savefig(spec_path, dpi=150)
        plt.close()
        
        # Attention plot
        if attention is not None:
            plt.figure(figsize=(10, 8))
            plt.imshow(attention.T, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Attention Alignment')
            plt.xlabel('Decoder Steps (Output)')
            plt.ylabel('Encoder Steps (Input)')
            plt.colorbar(label='Attention Weight')
            plt.tight_layout()
            plt.savefig(attn_path, dpi=150)
            plt.close()
    
    def plot_metrics_summary(self, df, output_dir):
        """Create summary bar chart of all metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics_info = [
            ('mcd', 'Mel Cepstral Distortion (dB)', 'Lower is better'),
            ('rmse', 'RMSE', 'Lower is better'),
            ('dtw', 'DTW Distance (normalized)', 'Lower is better'),
            ('duration_ratio', 'Duration Ratio', 'Closer to 1.0 is better'),
            ('spectral_convergence', 'Spectral Convergence', 'Lower is better'),
            ('attention_diagonal_score', 'Attention Alignment Score', 'Higher is better'),
        ]
        
        for ax, (metric, title, note) in zip(axes.flat, metrics_info):
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            
            ax.bar(['Mean'], [mean_val], yerr=[std_val], capsize=10, color='steelblue', alpha=0.8)
            ax.set_title(f'{title}\n({note})')
            ax.set_ylabel('Value')
            ax.text(0, mean_val + std_val + 0.05 * abs(mean_val), 
                   f'{mean_val:.3f} ± {std_val:.3f}', ha='center', fontsize=10)
        
        plt.suptitle('TTS Evaluation Metrics Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_summary.png", dpi=150)
        plt.close()
        print(f"Saved metrics summary chart")
    
    def plot_metric_distributions(self, df, output_dir):
        """Plot histograms of metric distributions."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        metrics = ['mcd', 'rmse', 'dtw', 'duration_ratio', 'spectral_convergence', 'attention_diagonal_score']
        titles = ['MCD (dB)', 'RMSE', 'DTW Distance', 'Duration Ratio', 'Spectral Convergence', 'Attention Score']
        
        for ax, metric, title in zip(axes.flat, metrics, titles):
            ax.hist(df[metric], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(df[metric].mean(), color='red', linestyle='--', label=f'Mean: {df[metric].mean():.3f}')
            ax.set_xlabel(title)
            ax.set_ylabel('Count')
            ax.legend()
        
        plt.suptitle('Distribution of Evaluation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "metric_distributions.png", dpi=150)
        plt.close()
        print(f"Saved metric distributions chart")
    
    def print_report(self, summary):
        """Print formatted evaluation report."""
        print("\n" + "=" * 60)
        print("EVALUATION REPORT")
        print("=" * 60)
        print(f"\nSamples evaluated: {summary['num_samples']}")
        print("\n--- Quality Metrics (lower is better) ---")
        print(f"  Mel Cepstral Distortion:  {summary['mcd_mean']:.3f} ± {summary['mcd_std']:.3f} dB")
        print(f"  RMSE:                     {summary['rmse_mean']:.4f} ± {summary['rmse_std']:.4f}")
        print(f"  DTW Distance:             {summary['dtw_mean']:.4f} ± {summary['dtw_std']:.4f}")
        print(f"  Spectral Convergence:     {summary['spectral_convergence_mean']:.4f}")
        print(f"  Log Spectral Distance:    {summary['log_spectral_distance_mean']:.4f}")
        print("\n--- Temporal Metrics ---")
        print(f"  Duration Ratio:           {summary['duration_ratio_mean']:.3f} ± {summary['duration_ratio_std']:.3f}")
        print(f"  (1.0 = perfect duration match)")
        print("\n--- Alignment Quality (higher is better) ---")
        print(f"  Attention Diagonal Score: {summary['attention_score_mean']:.3f}")
        print("\n" + "=" * 60)
        
        # Interpretation guide
        print("\nINTERPRETATION GUIDE:")
        print("-" * 40)
        print("MCD < 5 dB:  Excellent quality")
        print("MCD 5-7 dB:  Good quality")
        print("MCD > 8 dB:  Needs improvement")
        print("-" * 40)
        print("Duration Ratio 0.9-1.1:  Good timing")
        print("Attention Score > 0.7:   Good alignment")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Nupe TTS Model')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--num_samples', type=int, default=50,
                       help='Number of samples to evaluate')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = TTSEvaluator(args.checkpoint, args.config)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = NupeTTSDataset("data/splits/test.txt", args.config)
    print(f"Test samples available: {len(test_dataset)}")
    
    # Run evaluation
    df, summary = evaluator.evaluate_dataset(
        test_dataset,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # Print report
    evaluator.print_report(summary)
    
    print(f"\nDetailed results saved to: {args.output_dir}/")
    print(f"  - detailed_metrics.csv: Per-sample metrics")
    print(f"  - summary.json: Aggregated statistics")
    print(f"  - metrics_summary.png: Summary bar chart")
    print(f"  - metric_distributions.png: Histograms")
    print(f"  - spectrograms/: Predicted vs actual comparisons")
    print(f"  - attention/: Attention alignment plots")


if __name__ == "__main__":
    main()
