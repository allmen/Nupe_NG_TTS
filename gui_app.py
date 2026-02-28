"""
Requirements: pip install tkinter sounddevice soundfile pillow
"""

import os
import sys
import threading
import queue
import time
import json
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import numpy as np

# Audio libraries
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice/soundfile not installed. Audio features limited.")

# Image display
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not installed. Image display limited.")

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import yaml
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from model.lstm_tts import LSTMTTS
from utils.text_processor import NupeTextProcessor
from utils.audio_processor import AudioProcessor


class NupeTTSApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nupe TTS Research Interface")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # State variables
        self.model = None
        self.text_processor = None
        self.audio_processor = None
        self.device = None
        self.config = None
        self.current_audio = None
        self.sample_rate = 22050
        self.is_recording = False
        self.recorded_audio = []
        self.recording_thread = None
        
        # Data collection state
        self.collection_texts = []
        self.current_text_index = 0
        self.collection_output_dir = Path("data/collected")
        self.speaker_id = "speaker_new"
        
        # Create UI
        self.create_menu()
        self.create_notebook()
        self.create_status_bar()
        
        # Load model in background
        self.root.after(100, self.load_model_async)
    
    def create_menu(self):
        """Create menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Checkpoint...", command=self.load_checkpoint_dialog)
        file_menu.add_command(label="Load Text File for Collection...", command=self.load_collection_texts)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_notebook(self):
        """Create tabbed interface."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Tab 1: TTS Synthesis
        self.create_synthesis_tab()
        
        # Tab 2: Data Collection
        self.create_collection_tab()
        
        # Tab 3: Test with Dataset
        self.create_testing_tab()
        
        # Tab 4: Evaluation Results
        self.create_evaluation_tab()
    
    def create_synthesis_tab(self):
        """Create TTS synthesis tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="TTS Synthesis")
        
        # Left panel: Input
        left_frame = ttk.LabelFrame(tab, text="Input", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text input
        ttk.Label(left_frame, text="Enter Nupe text:").pack(anchor=tk.W)
        self.synth_text_input = scrolledtext.ScrolledText(left_frame, height=5, width=40, font=('Arial', 12))
        self.synth_text_input.pack(fill=tk.X, pady=5)
        self.synth_text_input.insert(tk.END, "tswako tswa mi na")
        
        # Synthesize button
        btn_frame = ttk.Frame(left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        self.synth_btn = ttk.Button(btn_frame, text="🔊 Synthesize", command=self.synthesize)
        self.synth_btn.pack(side=tk.LEFT, padx=5)
        
        self.play_btn = ttk.Button(btn_frame, text="▶ Play", command=self.play_audio, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_btn = ttk.Button(btn_frame, text="💾 Save WAV", command=self.save_audio, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)
        
        # Info display
        self.synth_info = ttk.Label(left_frame, text="Model not loaded", foreground="gray")
        self.synth_info.pack(anchor=tk.W, pady=5)
        
        # Right panel: Visualizations
        right_frame = ttk.LabelFrame(tab, text="Output Visualization", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Mel spectrogram display
        self.mel_canvas = tk.Canvas(right_frame, bg='white', height=200)
        self.mel_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Attention display
        self.attn_canvas = tk.Canvas(right_frame, bg='white', height=200)
        self.attn_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_collection_tab(self):
        """Create data collection tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Data Collection")
        
        # Top: Settings
        settings_frame = ttk.LabelFrame(tab, text="Collection Settings", padding=10)
        settings_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Speaker ID
        ttk.Label(settings_frame, text="Speaker ID:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.speaker_id_entry = ttk.Entry(settings_frame, width=20)
        self.speaker_id_entry.grid(row=0, column=1, padx=5)
        self.speaker_id_entry.insert(0, "speaker_new")
        
        # Output directory
        ttk.Label(settings_frame, text="Output Dir:").grid(row=0, column=2, sticky=tk.W, padx=5)
        self.output_dir_entry = ttk.Entry(settings_frame, width=30)
        self.output_dir_entry.grid(row=0, column=3, padx=5)
        self.output_dir_entry.insert(0, "data/collected")
        
        ttk.Button(settings_frame, text="Browse...", command=self.browse_output_dir).grid(row=0, column=4, padx=5)
        
        # Middle: Text to record
        text_frame = ttk.LabelFrame(tab, text="Text to Record", padding=10)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Progress
        progress_frame = ttk.Frame(text_frame)
        progress_frame.pack(fill=tk.X)
        
        self.collection_progress = ttk.Label(progress_frame, text="0 / 0 texts", font=('Arial', 12))
        self.collection_progress.pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, length=200, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        
        # Current text display
        self.current_text_display = tk.Label(
            text_frame, 
            text="Load a text file to begin collection",
            font=('Arial', 18, 'bold'),
            wraplength=600,
            height=3,
            bg='lightyellow',
            relief=tk.RIDGE,
            padx=20, pady=20
        )
        self.current_text_display.pack(fill=tk.X, pady=20)
        
        # Navigation buttons
        nav_frame = ttk.Frame(text_frame)
        nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(nav_frame, text="⏮ Previous", command=self.prev_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Next ⏭", command=self.next_text).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Skip", command=self.skip_text).pack(side=tk.LEFT, padx=5)
        
        # Bottom: Recording controls
        record_frame = ttk.LabelFrame(tab, text="Recording", padding=10)
        record_frame.pack(fill=tk.X, padx=10, pady=5)
        
        btn_record_frame = ttk.Frame(record_frame)
        btn_record_frame.pack()
        
        self.record_btn = ttk.Button(btn_record_frame, text="🎤 Start Recording", command=self.toggle_recording)
        self.record_btn.pack(side=tk.LEFT, padx=10)
        
        self.playback_btn = ttk.Button(btn_record_frame, text="▶ Play Recording", command=self.play_recording, state=tk.DISABLED)
        self.playback_btn.pack(side=tk.LEFT, padx=10)
        
        self.save_recording_btn = ttk.Button(btn_record_frame, text="💾 Save & Next", command=self.save_recording, state=tk.DISABLED)
        self.save_recording_btn.pack(side=tk.LEFT, padx=10)
        
        # Recording status
        self.recording_status = ttk.Label(record_frame, text="Ready to record", font=('Arial', 10))
        self.recording_status.pack(pady=5)
        
        # Waveform display
        self.waveform_canvas = tk.Canvas(record_frame, bg='black', height=100)
        self.waveform_canvas.pack(fill=tk.X, pady=5)
    
    def create_testing_tab(self):
        """Create testing with dataset tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Test Dataset")
        
        # Left: Sample list
        left_frame = ttk.LabelFrame(tab, text="Test Samples", padding=10)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Load button
        ttk.Button(left_frame, text="Load Test Data", command=self.load_test_data).pack(fill=tk.X, pady=5)
        
        # Sample listbox
        self.sample_listbox = tk.Listbox(left_frame, height=20, font=('Courier', 10))
        self.sample_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.sample_listbox.bind('<<ListboxSelect>>', self.on_sample_select)
        
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.sample_listbox.yview)
        self.sample_listbox.config(yscrollcommand=scrollbar.set)
        
        # Right: Comparison view
        right_frame = ttk.LabelFrame(tab, text="Comparison", padding=10)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Text display
        self.test_text_label = ttk.Label(right_frame, text="Select a sample", font=('Arial', 12))
        self.test_text_label.pack(anchor=tk.W, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(right_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="▶ Play Original", command=self.play_original).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🔊 Synthesize", command=self.synthesize_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="▶ Play Synthesized", command=self.play_synthesized).pack(side=tk.LEFT, padx=5)
        
        # Metrics display
        self.metrics_text = scrolledtext.ScrolledText(right_frame, height=8, width=50, font=('Courier', 10))
        self.metrics_text.pack(fill=tk.X, pady=5)
        
        # Comparison visualization
        self.comparison_canvas = tk.Canvas(right_frame, bg='white', height=300)
        self.comparison_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_evaluation_tab(self):
        """Create evaluation results tab."""
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Evaluation")
        
        # Controls
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Button(ctrl_frame, text="Run Full Evaluation", command=self.run_evaluation).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="Load Results", command=self.load_evaluation_results).pack(side=tk.LEFT, padx=5)
        
        # Results display
        results_frame = ttk.LabelFrame(tab, text="Evaluation Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Summary text
        self.eval_summary = scrolledtext.ScrolledText(results_frame, height=10, font=('Courier', 11))
        self.eval_summary.pack(fill=tk.X, pady=5)
        
        # Charts display
        self.eval_canvas = tk.Canvas(results_frame, bg='white', height=400)
        self.eval_canvas.pack(fill=tk.BOTH, expand=True, pady=5)
    
    def create_status_bar(self):
        """Create status bar."""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar."""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
    
    def load_model_async(self):
        """Load model in background."""
        self.update_status("Loading model...")
        threading.Thread(target=self._load_model, daemon=True).start()
    
    def _load_model(self):
        """Actually load the model."""
        try:
            config_path = "config/config.yaml"
            checkpoint_path = "models/best_model.pth"
            
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
            
            # Load model
            self.model = LSTMTTS(config_path).to(self.device)
            
            if os.path.exists(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                epoch = checkpoint.get('epoch', 'N/A')
                self.root.after(0, lambda: self.synth_info.config(
                    text=f"Model loaded (epoch {epoch}) | Device: {self.device}",
                    foreground="green"
                ))
            else:
                self.root.after(0, lambda: self.synth_info.config(
                    text="No checkpoint found - using untrained model",
                    foreground="orange"
                ))
            
            self.model.eval()
            
            # Processors
            self.text_processor = NupeTextProcessor(config_path)
            self.audio_processor = AudioProcessor(config_path)
            self.sample_rate = self.config['audio']['sample_rate']
            
            self.root.after(0, lambda: self.update_status("Model loaded successfully"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to load model: {e}"))
            self.root.after(0, lambda: self.update_status(f"Error: {e}"))
    
    def synthesize(self):
        """Synthesize speech from text."""
        if self.model is None:
            messagebox.showwarning("Warning", "Model not loaded yet")
            return
        
        text = self.synth_text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter text to synthesize")
            return
        
        self.update_status(f"Synthesizing: {text[:50]}...")
        self.synth_btn.config(state=tk.DISABLED)
        
        threading.Thread(target=self._synthesize, args=(text,), daemon=True).start()
    
    def _synthesize(self, text):
        """Perform synthesis in background."""
        try:
            # Process text
            text_seq = self.text_processor.text_to_sequence(text)
            text_tensor = torch.LongTensor(text_seq).unsqueeze(0).to(self.device)
            text_length = torch.LongTensor([len(text_seq)])
            
            # Generate
            with torch.no_grad():
                mel_output, stop_output, attention = self.model(text_tensor, text_length)
            
            mel = mel_output.squeeze(0).cpu().numpy()
            attn = attention.squeeze(0).cpu().numpy() if attention is not None else None
            
            # Convert to audio
            audio = self.audio_processor.inv_melspectrogram(mel)
            self.current_audio = audio
            
            # Save temp visualizations
            self._save_temp_plots(mel, attn, text)
            
            # Update UI
            self.root.after(0, self._update_synthesis_ui)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Synthesis failed: {e}"))
            self.root.after(0, lambda: self.synth_btn.config(state=tk.NORMAL))
    
    def _save_temp_plots(self, mel, attn, text):
        """Save temporary visualization plots."""
        temp_dir = Path("temp_plots")
        temp_dir.mkdir(exist_ok=True)
        
        # Mel spectrogram
        plt.figure(figsize=(10, 3))
        plt.imshow(mel.T, aspect='auto', origin='lower', cmap='magma')
        plt.title(f'Mel: "{text[:40]}..."' if len(text) > 40 else f'Mel: "{text}"')
        plt.xlabel('Time')
        plt.ylabel('Mel Bands')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(temp_dir / "mel_temp.png", dpi=100)
        plt.close()
        
        # Attention
        if attn is not None:
            plt.figure(figsize=(8, 6))
            plt.imshow(attn.T, aspect='auto', origin='lower', cmap='viridis')
            plt.title('Attention Alignment')
            plt.xlabel('Decoder Steps')
            plt.ylabel('Encoder Steps')
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(temp_dir / "attn_temp.png", dpi=100)
            plt.close()
    
    def _update_synthesis_ui(self):
        """Update UI after synthesis."""
        self.synth_btn.config(state=tk.NORMAL)
        self.play_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.update_status("Synthesis complete")
        
        # Load and display images
        if PIL_AVAILABLE:
            try:
                # Mel
                mel_img = Image.open("temp_plots/mel_temp.png")
                mel_img = mel_img.resize((500, 180), Image.Resampling.LANCZOS)
                self.mel_photo = ImageTk.PhotoImage(mel_img)
                self.mel_canvas.delete("all")
                self.mel_canvas.create_image(250, 90, image=self.mel_photo)
                
                # Attention
                if os.path.exists("temp_plots/attn_temp.png"):
                    attn_img = Image.open("temp_plots/attn_temp.png")
                    attn_img = attn_img.resize((400, 180), Image.Resampling.LANCZOS)
                    self.attn_photo = ImageTk.PhotoImage(attn_img)
                    self.attn_canvas.delete("all")
                    self.attn_canvas.create_image(200, 90, image=self.attn_photo)
            except Exception as e:
                print(f"Display error: {e}")
    
    def play_audio(self):
        """Play synthesized audio."""
        if self.current_audio is None or not AUDIO_AVAILABLE:
            return
        try:
            sd.play(self.current_audio, self.sample_rate)
            self.update_status("Playing audio...")
        except Exception as e:
            messagebox.showerror("Error", f"Playback failed: {e}")
    
    def save_audio(self):
        """Save synthesized audio."""
        if self.current_audio is None:
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filename:
            sf.write(filename, self.current_audio, self.sample_rate)
            self.update_status(f"Saved to {filename}")
    
    # ==================== Data Collection ====================
    
    def browse_output_dir(self):
        """Browse for output directory."""
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir_entry.delete(0, tk.END)
            self.output_dir_entry.insert(0, dir_path)
    
    def load_collection_texts(self):
        """Load text file for data collection."""
        filename = filedialog.askopenfilename(
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r', encoding='utf-8') as f:
                self.collection_texts = [line.strip() for line in f if line.strip()]
            
            self.current_text_index = 0
            self.update_collection_display()
            self.update_status(f"Loaded {len(self.collection_texts)} texts for collection")
    
    def update_collection_display(self):
        """Update collection text display."""
        if self.collection_texts:
            self.current_text_display.config(text=self.collection_texts[self.current_text_index])
            self.collection_progress.config(
                text=f"{self.current_text_index + 1} / {len(self.collection_texts)} texts"
            )
            self.progress_bar['value'] = (self.current_text_index / len(self.collection_texts)) * 100
    
    def prev_text(self):
        """Go to previous text."""
        if self.collection_texts and self.current_text_index > 0:
            self.current_text_index -= 1
            self.update_collection_display()
    
    def next_text(self):
        """Go to next text."""
        if self.collection_texts and self.current_text_index < len(self.collection_texts) - 1:
            self.current_text_index += 1
            self.update_collection_display()
    
    def skip_text(self):
        """Skip current text."""
        self.next_text()
    
    def toggle_recording(self):
        """Toggle audio recording."""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "sounddevice not installed")
            return
        
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.recorded_audio = []
        self.record_btn.config(text="⏹ Stop Recording")
        self.recording_status.config(text="🔴 Recording...", foreground="red")
        self.playback_btn.config(state=tk.DISABLED)
        self.save_recording_btn.config(state=tk.DISABLED)
        
        def record_callback(indata, frames, time_info, status):
            if self.is_recording:
                self.recorded_audio.append(indata.copy())
        
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=record_callback
        )
        self.stream.start()
    
    def stop_recording(self):
        """Stop recording audio."""
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        self.record_btn.config(text="🎤 Start Recording")
        self.recording_status.config(text="Recording complete", foreground="green")
        
        if self.recorded_audio:
            self.recorded_audio = np.concatenate(self.recorded_audio, axis=0)
            self.playback_btn.config(state=tk.NORMAL)
            self.save_recording_btn.config(state=tk.NORMAL)
            
            # Draw waveform
            self.draw_waveform(self.recorded_audio)
    
    def draw_waveform(self, audio):
        """Draw waveform on canvas."""
        self.waveform_canvas.delete("all")
        width = self.waveform_canvas.winfo_width()
        height = self.waveform_canvas.winfo_height()
        
        if width < 10:
            width = 600
        if height < 10:
            height = 100
        
        # Downsample for display
        step = max(1, len(audio) // width)
        samples = audio[::step, 0] if len(audio.shape) > 1 else audio[::step]
        
        # Normalize
        max_val = np.max(np.abs(samples)) + 1e-8
        samples = samples / max_val
        
        # Draw
        mid = height // 2
        for i in range(len(samples) - 1):
            x1 = i * width // len(samples)
            x2 = (i + 1) * width // len(samples)
            y1 = mid - int(samples[i] * mid * 0.9)
            y2 = mid - int(samples[i + 1] * mid * 0.9)
            self.waveform_canvas.create_line(x1, y1, x2, y2, fill='lime')
    
    def play_recording(self):
        """Play recorded audio."""
        if self.recorded_audio is not None and len(self.recorded_audio) > 0 and AUDIO_AVAILABLE:
            sd.play(self.recorded_audio, self.sample_rate)
    
    def save_recording(self):
        """Save recording and move to next text."""
        if self.recorded_audio is None or len(self.recorded_audio) == 0:
            return
        
        # Create output directory
        speaker_id = self.speaker_id_entry.get()
        output_dir = Path(self.output_dir_entry.get()) / speaker_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"{speaker_id}_{self.current_text_index:04d}_{timestamp}.wav"
        audio_path = output_dir / audio_filename
        sf.write(str(audio_path), self.recorded_audio, self.sample_rate)
        
        # Save text
        if self.collection_texts:
            text_path = audio_path.with_suffix('.txt')
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(self.collection_texts[self.current_text_index])
        
        self.update_status(f"Saved: {audio_filename}")
        
        # Reset and move to next
        self.recorded_audio = []
        self.playback_btn.config(state=tk.DISABLED)
        self.save_recording_btn.config(state=tk.DISABLED)
        self.waveform_canvas.delete("all")
        self.next_text()
    
    # ==================== Testing Tab ====================
    
    def load_test_data(self):
        """Load test dataset."""
        try:
            import pandas as pd
            test_path = "data/splits/test.txt"
            
            if os.path.exists(test_path):
                df = pd.read_csv(test_path)
                self.test_samples = df.to_dict('records')
                
                self.sample_listbox.delete(0, tk.END)
                for i, row in enumerate(self.test_samples):
                    text = row.get('text', '')[:40]
                    self.sample_listbox.insert(tk.END, f"{i}: {text}")
                
                self.update_status(f"Loaded {len(self.test_samples)} test samples")
            else:
                messagebox.showwarning("Warning", "Test file not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load test data: {e}")
    
    def on_sample_select(self, event):
        """Handle sample selection."""
        selection = self.sample_listbox.curselection()
        if selection:
            idx = selection[0]
            sample = self.test_samples[idx]
            self.selected_sample = sample
            self.test_text_label.config(text=f"Text: {sample.get('text', 'N/A')}")
    
    def play_original(self):
        """Play original audio."""
        if hasattr(self, 'selected_sample') and AUDIO_AVAILABLE:
            audio_path = self.selected_sample.get('audio_path', '')
            if os.path.exists(audio_path):
                audio, sr = sf.read(audio_path)
                sd.play(audio, sr)
    
    def synthesize_selected(self):
        """Synthesize selected sample."""
        if hasattr(self, 'selected_sample') and self.model is not None:
            text = self.selected_sample.get('text', '')
            self.synth_text_input.delete("1.0", tk.END)
            self.synth_text_input.insert(tk.END, text)
            self.synthesize()
    
    def play_synthesized(self):
        """Play synthesized audio."""
        self.play_audio()
    
    # ==================== Evaluation Tab ====================
    
    def run_evaluation(self):
        """Run full evaluation."""
        self.update_status("Running evaluation...")
        threading.Thread(target=self._run_evaluation, daemon=True).start()
    
    def _run_evaluation(self):
        """Run evaluation in background."""
        try:
            import subprocess
            result = subprocess.run(
                ['python', 'evaluate.py', '--num_samples', '20'],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            self.root.after(0, self.load_evaluation_results)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Evaluation failed: {e}"))
    
    def load_evaluation_results(self):
        """Load and display evaluation results."""
        try:
            summary_path = "evaluation_results/summary.json"
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                
                # Display summary
                self.eval_summary.delete("1.0", tk.END)
                self.eval_summary.insert(tk.END, "=" * 50 + "\n")
                self.eval_summary.insert(tk.END, "EVALUATION SUMMARY\n")
                self.eval_summary.insert(tk.END, "=" * 50 + "\n\n")
                
                self.eval_summary.insert(tk.END, f"Samples: {summary.get('num_samples', 'N/A')}\n\n")
                self.eval_summary.insert(tk.END, f"MCD:              {summary.get('mcd_mean', 0):.3f} ± {summary.get('mcd_std', 0):.3f} dB\n")
                self.eval_summary.insert(tk.END, f"RMSE:             {summary.get('rmse_mean', 0):.4f}\n")
                self.eval_summary.insert(tk.END, f"DTW Distance:     {summary.get('dtw_mean', 0):.4f}\n")
                self.eval_summary.insert(tk.END, f"Duration Ratio:   {summary.get('duration_ratio_mean', 0):.3f}\n")
                self.eval_summary.insert(tk.END, f"Attention Score:  {summary.get('attention_score_mean', 0):.3f}\n")
                
                # Display chart
                if PIL_AVAILABLE and os.path.exists("evaluation_results/metrics_summary.png"):
                    img = Image.open("evaluation_results/metrics_summary.png")
                    img = img.resize((700, 400), Image.Resampling.LANCZOS)
                    self.eval_photo = ImageTk.PhotoImage(img)
                    self.eval_canvas.delete("all")
                    self.eval_canvas.create_image(350, 200, image=self.eval_photo)
                
                self.update_status("Evaluation results loaded")
            else:
                self.eval_summary.insert(tk.END, "No evaluation results found.\nClick 'Run Full Evaluation' first.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {e}")
    
    def load_checkpoint_dialog(self):
        """Dialog to load a different checkpoint."""
        filename = filedialog.askopenfilename(
            filetypes=[("PyTorch checkpoints", "*.pth"), ("All files", "*.*")]
        )
        if filename:
            self.update_status(f"Loading checkpoint: {filename}")
            # Would need to reload model here
    
    def show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Nupe TTS",
            "Nupe TTS Research Interface\n\n"
            "A GUI for:\n"
            "• Text-to-Speech synthesis\n"
            "• Speech data collection\n"
            "• Model evaluation\n\n"
            "Built for Nupe language TTS research."
        )


def main():
    # Check dependencies
    missing = []
    if not AUDIO_AVAILABLE:
        missing.append("sounddevice/soundfile (pip install sounddevice soundfile)")
    if not PIL_AVAILABLE:
        missing.append("Pillow (pip install Pillow)")
    
    if missing:
        print("Warning: Some features disabled due to missing packages:")
        for m in missing:
            print(f"  - {m}")
    
    root = tk.Tk()
    app = NupeTTSApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
