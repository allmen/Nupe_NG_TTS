# Nupe TTS Project

This project implements a neural text-to-speech (TTS) system for the Nupe language using a sequence-to-sequence LSTM-based model with attention. The codebase is organized for research and reproducibility, supporting training, inference, and data preprocessing.

## Project Structure

- `train.py` — Main training script. Handles model training, checkpointing, and logging.
- `inference.py` — Script for synthesizing speech from text using a trained model checkpoint.
- `preprocess.py` — Prepares and splits the dataset, cleans text, and analyzes statistics.
- `config/config.yaml` — All hyperparameters for audio, model, training, and data splits.
- `model/lstm_tts.py` — Model definition: LSTM encoder, attention, LSTM decoder.
- `utils/` — Helper modules for audio processing, text processing, and data loading.
- `data/` — Contains raw and processed data, splits, and feature caches.
- `models/` — Stores model checkpoints and the best model.
- `logs/` — TensorBoard logs for training visualization.

## Model Used

- **Architecture:**
  - LSTM encoder (bidirectional)
  - Additive attention
  - LSTM decoder with prenet
  - Mel-spectrogram regression (80 bands)
  - Stop token prediction (end-of-speech)
- **Input:** Cleaned Nupe text (character-level)
- **Output:** Mel-spectrogram, then waveform via Griffin-Lim

## Training

- Run with: `python train.py` (preferably in a GPU-enabled environment)
- Checkpoints are saved in `models/checkpoints/` every 10 epochs
- The best model (lowest validation loss) is saved as `models/best_model.pth`
- Training logs are available in `logs/` for TensorBoard
- Hyperparameters (batch size, learning rate, max sequence length, etc.) are set in `config/config.yaml`

## Inference

- Run with: `python inference.py --text "your text here" --checkpoint models/best_model.pth --output output.wav --plot`
- Produces:
  - `output.wav`: Synthesized speech audio
  - `output_attention.png`: Attention alignment plot (shows how text aligns to audio frames)
  - `output_mel.png`: Mel-spectrogram plot

## GUI Interface

Launch the graphical interface:

```bash
python gui_app.py
```

### Features:

**1. TTS Synthesis Tab**

- Enter Nupe text and synthesize speech
- Play and save generated audio
- View mel-spectrogram and attention visualizations

**2. Data Collection Tab**

- Load a text file with sentences to record
- Record user speech with microphone
- Playback and verify recordings
- Auto-save with speaker ID and text metadata

**3. Test Dataset Tab**

- Load existing test samples
- Compare original vs synthesized audio
- View per-sample quality metrics

**4. Evaluation Tab**

- Run full model evaluation
- View summary statistics and charts
- Load saved evaluation results

## Output Interpretation

- **output.wav**: The generated speech. Quality depends on training progress and data.
- **output_attention.png**: Visualizes attention weights. Diagonal patterns indicate good alignment between input text and output audio.
- **output_mel.png**: Shows the predicted mel-spectrogram. Should resemble real speech patterns as training improves.

## Research Notes

- The model is LSTM-based, suitable for low-resource and interpretable TTS research.
- All code is modular and configurable for experimentation (change model size, attention, etc. in config).
- Training on a small GPU (e.g., 4GB) is supported by truncating long sequences and using batch size 1.
- For best results, train for more epochs and use high-quality, well-aligned data.
- The code is designed for easy extension (e.g., add speaker embeddings, phoneme input, or replace Griffin-Lim with a neural vocoder).

## Quantitative Evaluation

Run the evaluation script to compute objective metrics:

```bash
python evaluate.py --checkpoint models/best_model.pth --num_samples 50 --output_dir evaluation_results
```

### Metrics Computed

| Metric                                  | Description                                                                 | Interpretation                                      |
| --------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------- |
| **MCD (Mel Cepstral Distortion)** | Measures cepstral coefficient difference between predicted and ground truth | < 5 dB: Excellent, 5-7 dB: Good, > 8 dB: Needs work |
| **RMSE**                          | Root mean square error on mel spectrograms                                  | Lower is better                                     |
| **DTW Distance**                  | Dynamic time warping distance (handles temporal misalignment)               | Lower is better                                     |
| **Duration Ratio**                | Predicted length / target length                                            | 0.9-1.1: Good timing                                |
| **Spectral Convergence**          | Relative Frobenius norm                                                     | Lower is better                                     |
| **Log Spectral Distance**         | Per-frame spectral envelope difference                                      | Lower is better                                     |
| **Attention Diagonal Score**      | How well attention aligns diagonally                                        | > 0.7: Good alignment                               |

### Output Files

- `evaluation_results/detailed_metrics.csv` — Per-sample metrics
- `evaluation_results/summary.json` — Aggregated statistics
- `evaluation_results/metrics_summary.png` — Bar chart summary
- `evaluation_results/metric_distributions.png` — Histogram distributions
- `evaluation_results/spectrograms/` — Side-by-side predicted vs ground truth comparisons
- `evaluation_results/attention/` — Attention alignment visualizations

### Interpreting Results for Research

1. **MCD** is the primary quality metric for TTS research. Values below 5 dB indicate publishable quality.
2. **Duration Ratio** near 1.0 indicates the model has learned proper duration prediction.
3. **Attention Diagonal Score** measures alignment quality — a diagonal attention pattern indicates the model is attending to the correct input positions sequentially.
4. Compare metrics across training checkpoints to track improvement.
5. Use the spectrogram comparisons to visually inspect formant structure and harmonic patterns.
