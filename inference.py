import torch
import numpy as np
import yaml
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from model.lstm_tts import LSTMTTS
    from utils.text_processor import NupeTextProcessor
    from utils.audio_processor import AudioProcessor
except ImportError as e:
    print(f"Import Error: {e}")
    print("Trying alternative import paths...")
    sys.path.append('.')
    try:
        from model.lstm_tts import LSTMTTS
        from utils.text_processor import NupeTextProcessor
        from utils.audio_processor import AudioProcessor
    except ImportError:
        print("Could not import modules. Please check your project structure.")
        sys.exit(1)


class NupeTTSInference:
    def __init__(self, model_path="models/best_model.pth", config_path="config/config.yaml"):
        print(f"Initializing Nupe TTS with model: {model_path}")
        config_path = self.find_config(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"CUDA GPU: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        self.model = LSTMTTS(config_path).to(self.device)
        self.load_model(model_path)
        self.model.eval()
        print("Model loaded and set to evaluation mode")
        self.text_processor = NupeTextProcessor(config_path)
        self.audio_processor = AudioProcessor(config_path)
    
    def find_config(self, config_path):
        if os.path.exists(config_path):
            return config_path
        config_candidates = [
            "config.yaml", "config/config.yaml", "../config/config.yaml",
            "./config.yaml", "hparams.yaml", "params.yaml"
        ]
        for cand in config_candidates:
            if os.path.exists(cand):
                print(f"Found config at: {cand}")
                return cand
        raise FileNotFoundError(f"Config file not found. Tried: {config_candidates}")
    
    def load_model(self, model_path):
        model_path = self.find_model(model_path)
        checkpoint = torch.load(model_path, map_location=self.device)
        print(f"Loaded checkpoint from {model_path}")
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        state_dict_keys = ['model_state_dict', 'state_dict', 'model', 'generator']
        for key in state_dict_keys:
            if key in checkpoint:
                try:
                    self.model.load_state_dict(checkpoint[key])
                    print(f"✓ Loaded model from '{key}'")
                    return
                except Exception as e:
                    print(f"Failed to load from '{key}': {e}")
        try:
            self.model.load_state_dict(checkpoint)
            print("✓ Loaded checkpoint as state dict")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            print("Attempting partial load...")
            self.partial_load_state_dict(checkpoint)
    
    def find_model(self, model_path):
        if os.path.exists(model_path):
            return model_path
        model_candidates = [
            "best_model.pth", "models/best_model.pth", "../models/best_model.pth",
            "./models/best_model.pth", "checkpoints/best_model.pth", "model.pth", "checkpoint.pth"
        ]
        for cand in model_candidates:
            if os.path.exists(cand):
                print(f"Found model at: {cand}")
                return cand
        raise FileNotFoundError(f"Model file not found. Tried: {model_candidates}")
    
    def partial_load_state_dict(self, checkpoint):
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        if len(pretrained_dict) == 0:
            print("No matching keys found in checkpoint!")
            return
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print(f"✓ Partially loaded {len(pretrained_dict)}/{len(state_dict)} parameters")

    def text_to_speech(self, text, output_path="output.wav", plot_attention=False):
        print(f"\nProcessing text: '{text}'")
        output_path = str(output_path)  # Ensure string for .replace()
        try:
            text_sequence = self.text_processor.text_to_sequence(text)
            text_sequence = [int(x) for x in text_sequence]
            print(f"Text sequence length: {len(text_sequence)}")

            # Clamp indices safely to embedding size
            embedding = getattr(self.model, 'embedding', getattr(getattr(self.model, 'encoder', None), 'embedding', None))
            if embedding is not None and hasattr(embedding, 'num_embeddings'):
                vocab_size = int(embedding.num_embeddings)
                padding_idx = getattr(embedding, 'padding_idx', None)
                safe_idx = int(padding_idx) if padding_idx is not None else (vocab_size - 1)
                replaced = 0
                for i, idx in enumerate(text_sequence):
                    if idx < 0 or idx >= vocab_size:
                        text_sequence[i] = safe_idx
                        replaced += 1
                if replaced > 0:
                    print(f"Warning: {replaced} token indices were out-of-range and replaced with safe index {safe_idx} (vocab_size={vocab_size})")
            else:
                text_sequence = [max(0, int(x)) for x in text_sequence]

            text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
            text_length = torch.LongTensor([len(text_sequence)])  # Keep on CPU for pack_padded_sequence

            print("Generating mel-spectrogram...")
            with torch.no_grad():
                mel_output, stop_output, attention = self.model(text_tensor, text_length)

            if isinstance(mel_output, tuple):
                mel_output = mel_output[0]
            mel_output = mel_output.squeeze(0).cpu().numpy()
            if attention is not None:
                attention = attention.squeeze(0).cpu().numpy()

            print(f"Mel shape: {mel_output.shape}")
            print("Converting mel to audio...")

            if hasattr(self.audio_processor, 'inv_melspectrogram'):
                # Correct np.complex usage inside audio_processor
                audio = self.audio_processor.inv_melspectrogram(mel_output)
            else:
                audio = self.griffin_lim(mel_output)

            sf.write(output_path, audio, self.audio_processor.sr)
            print(f"✓ Audio saved to {output_path}")
            print(f"  Duration: {len(audio)/self.audio_processor.sr:.2f} seconds")

            if plot_attention and attention is not None:
                self.plot_attention(attention, text, output_path)
            self.plot_mel(mel_output, output_path)

            return audio, mel_output, attention
        except Exception as e:
            print(f"Error during synthesis: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def griffin_lim(self, mel, n_iter=50):
        try:
            import librosa
            mel_db = librosa.power_to_db(mel)
            linear_spec = np.exp(mel_db / 20)
            angles = np.exp(2j * np.pi * np.random.rand(*linear_spec.shape))
            for i in range(n_iter):
                full = linear_spec * angles
                inverse = librosa.istft(full)
                rebuilt = librosa.stft(inverse)
                angles = np.exp(1j * np.angle(rebuilt))
            audio = librosa.istft(linear_spec * angles)
            return audio
        except ImportError:
            print("Warning: librosa not installed. Cannot use Griffin-Lim.")
            return np.zeros(22050)

    def plot_attention(self, attention, text, output_path):
        try:
            save_path = output_path.replace('.wav', '_attention.png')
            plt.figure(figsize=(10, 8))
            plt.imshow(attention.T, aspect='auto', origin='lower', cmap='viridis')
            plt.title(f'Attention: "{text}"')
            plt.xlabel('Decoder Steps')
            plt.ylabel('Encoder Steps')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"✓ Attention plot saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save attention plot: {e}")

    def plot_mel(self, mel, output_path):
        try:
            save_path = output_path.replace('.wav', '_mel.png')
            plt.figure(figsize=(12, 4))
            plt.imshow(mel.T, aspect='auto', origin='lower', cmap='magma')
            plt.title('Mel-Spectrogram')
            plt.xlabel('Time Frames')
            plt.ylabel('Mel Bands')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"✓ Mel spectrogram saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save mel plot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Nupe TTS Inference')
    parser.add_argument('--text', type=str, required=True, help='Nupe text to synthesize')
    parser.add_argument('--checkpoint', type=str, default='models/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='output.wav', help='Output audio filename')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--plot', action='store_true', help='Generate attention plots')
    args = parser.parse_args()

    print("=" * 60)
    print("Nupe TTS Inference Engine")
    print("=" * 60)

    try:
        tts = NupeTTSInference(model_path=args.checkpoint, config_path=args.config)
        audio, mel, attention = tts.text_to_speech(text=args.text, output_path=args.output, plot_attention=args.plot)

        if audio is not None:
            print(f"\n" + "=" * 60)
            print("✓ SUCCESS: Audio generated!")
            print(f"  Output: {args.output}")
            print(f"  Text: {args.text}")
            print("=" * 60)
        else:
            print("\n✗ FAILED: Could not generate audio")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()