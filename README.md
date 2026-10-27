# EmotionSense: Noise-Aware Speech Emotion Recognition Using CNN-LSTM

A comprehensive, academic mini-project demonstrating Speech Emotion Recognition (SER) using deep learning (CNN-LSTM), classical baselines (SVM), multiple acoustic feature front-ends (MFCC vs. Log-Mel spectrograms), and controlled noise robustness evaluations across diverse noise spectra on the RAVDESS dataset.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![TensorFlow 2.15–2.18](https://img.shields.io/badge/tensorflow-2.15%E2%80%932.18-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%E2%80%931.44-red)](https://streamlit.io/)
[![Pytest 38/38 Passing](https://img.shields.io/badge/pytest-38%2F38%20passing-brightgreen)](https://docs.pytest.org/)

---

## 📌 Project Overview

EmotionSense classifies human emotion from spoken English utterances into 8 affective states: *Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised*.

### Key Highlights
- **Strict Actor-Independent Splits:** 24 actors are partitioned into mutually exclusive subsets (14 train, 5 validation, 5 test actors), guaranteeing **zero speaker leakage**.
- **Architectural Benchmark:** Evaluates four distinct acoustic models:
  1. **Original CNN-LSTM:** 40-dimensional MFCC sequences (shape `200 × 40 × 1`).
  2. **Improved CNN-LSTM:** MFCC + Delta + Delta-Delta features (120-dim) with early stopping, weight decay, and dropout regularisation.
  3. **SVM Baseline:** RBF kernel SVM operating on 160-dimensional summary statistics (mean, std, min, max across time).
  4. **Log-Mel CNN-LSTM:** 64-channel Log-Mel spectrograms (shape `200 × 64 × 1`) with learning rate scheduling and early stopping.
- **Multi-Noise Robustness Evaluation:** Models are evaluated across clean speech and 3 simulated noise profiles:
  - **AWGN (Additive White Gaussian Noise):** Clean, 20 dB, 10 dB, 0 dB SNR.
  - **Pink Noise (1/f spectral roll-off):** 20 dB, 10 dB, 0 dB SNR.
  - **Mains Hum (50 Hz fundamental + harmonics):** 20 dB, 10 dB, 0 dB SNR.
- **Interactive Streamlit Studio:** Multi-page dashboard with real-time waveform & spectrogram views, confidence estimation, audio quality checks, low-confidence warnings, ethical disclaimers, and interactive noise demos.
- **Comprehensive Academic Report:** Includes `EmotionSense_Improved_Mini_Project_Report.docx` with 19 detailed sections, tables, and inserted empirical figures.

---

## 📊 Summary of Empirical Results (Held-Out Test Set: 300 Files)

| Model | Feature | Clean Acc / Macro-F1 | 20 dB SNR Acc / F1 | 10 dB SNR Acc / F1 | 0 dB SNR Acc / F1 |
|---|---|---|---|---|---|
| **SVM Baseline** | MFCC-stats (160) | **47.0%** / **0.4556** | 23.0% / 0.1751 | 14.3% / 0.0477 | 13.3% / 0.0294 |
| **Log-Mel CNN-LSTM** | LogMel-64 | 30.0% / 0.2608 | 20.3% / 0.1519 | 21.3% / 0.1649 | 18.7% / 0.1228 |
| **Original CNN-LSTM** | MFCC-40 | 28.3% / 0.2658 | **29.7%** / **0.2590** | **28.0%** / 0.2331 | **28.7%** / **0.2145** |
| **Improved CNN-LSTM** | MFCC+Delta | 16.0% / 0.1371 | 26.0% / 0.2490 | 27.3% / **0.2557** | 24.7% / 0.2301 |

> **Crucial Finding:** The SVM baseline achieves high accuracy in pristine clean conditions (47.0%) but catastrophically degrades under noise (13.3% at 0 dB SNR). Conversely, the CNN-LSTM exhibits remarkable noise invariance, maintaining ~28.7% accuracy at 0 dB AWGN and reaching 31.0% under Pink noise.

---

## 🚀 Quickstart & Setup

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12** (Python 3.12 confirmed working).
- Local CPU Windows / Linux environment (no GPU or cloud accounts required).

### 2. Environment Installation
```powershell
# Clone or navigate to the repository
cd C:\Users\shamm\Downloads\EmotionSense

# Optional: Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Dataset Setup
1. Download **Audio_Speech_Actors_01-24.zip** (speech-only, ~215 MB) from Zenodo: [https://zenodo.org/records/1188976](https://zenodo.org/records/1188976).
2. Extract the archive so actor folders sit under `data/` or `data/ravdess/`:
```
data/
└── Actor_01/
    ├── 03-01-01-01-01-01-01.wav
    └── ...
```
3. Inspect and verify dataset discovery:
```powershell
python scripts/inspect_dataset.py
```

---

## 🛠️ Reproducibility Pipeline

All models and evaluations are fully reproducible using explicit random seeds:

```powershell
# Step 1: Train Models
python train.py             # Trains Original CNN-LSTM (models/emotionsense_cnn_lstm.keras)
python train_svm.py         # Trains SVM Baseline (models/svm_model.pkl)
python train_improved.py    # Trains Improved CNN-LSTM (models/improved_cnn_lstm.keras)
python train_logmel.py      # Trains Log-Mel CNN-LSTM (models/logmel_cnn_lstm.keras)

# Step 2: Run Evaluations
python evaluate.py          # Standard AWGN evaluation
python evaluate_all.py      # Cross-model comparative evaluation
python evaluate_extended.py # Multi-noise evaluation (AWGN, Pink, Hum)
python evaluate_logmel.py   # Log-Mel evaluation across noise conditions

# Step 3: Run Automated Tests
pytest tests/ -v            # 38 passing unit tests

# Step 4: Generate Word Report
python generate_report.py   # Builds EmotionSense_Improved_Mini_Project_Report.docx

# Step 5: Single-Clip Prediction
python predict.py "data\Actor_01\03-01-01-01-01-01-01.wav"

# Step 6: Launch Web Dashboard
streamlit run app.py
```

---

## 🌐 Streamlit Studio Dashboard

Launch the interactive interface with:
```powershell
streamlit run app.py
```
Pages available:
1. **🎤 Analyze Audio:** File uploader supporting WAV, MP3, FLAC, OGG with waveform display, Mel spectrogram, top emotion prediction, softmax confidence score, low-confidence warning (<30%), ethical disclaimer, and 2.5-second segmentation timeline.
2. **🔬 Noise Demo:** Live audio noise generator injecting synthetic AWGN at adjustable SNR levels (clean, 20 dB, 10 dB, 0 dB).
3. **📊 Noise Robustness:** Metric cards, comparative plots, full confusion matrices, and the extended multi-noise evaluation tab (AWGN, Pink Noise, 50 Hz Hum).
4. **📈 Model Comparison:** Head-to-head comparison table and bar charts across all trained models (Original CNN-LSTM, SVM, Improved CNN-LSTM, Log-Mel CNN-LSTM).
5. **ℹ️ Setup Guide:** Step-by-step setup walkthrough and active project status checklist.

---

## 🧪 Automated Test Suite (38/38 Passing)

The test suite validates signal processing, mathematical invariants, and inference reliability:
- `tests/test_model.py`: Verifies CNN-LSTM output tensor dimensions `(batch_size, 8)` and probability normalization.
- `tests/test_preprocessing.py`: Validates MFCC extraction shape `(200, 40, 1)`, finite value checks, and silent audio handling.
- `tests/test_noise.py`: Validates AWGN noise generation, SNR accuracy, and silence preservation.
- `tests/test_new_features.py`: Tests delta MFCC features, SVM statistical features, and model pickle deserialization.
- `tests/test_extended.py`: Tests Log-Mel feature shapes, Pink noise & 50 Hz Hum generation, reproducibility across seeds, metric calculations, and prediction error handling.

Run with:
```powershell
pytest tests/ -v
```

---

## 📂 Project Structure

```
EmotionSense/
├── app.py                                            # Streamlit Dashboard (EmotionSense Studio)
├── train.py                                          # Original CNN-LSTM training pipeline
├── train_svm.py                                      # SVM baseline training pipeline
├── train_improved.py                                 # Improved CNN-LSTM training pipeline
├── train_logmel.py                                   # Log-Mel CNN-LSTM training pipeline
├── evaluate.py                                       # Clean + AWGN evaluation
├── evaluate_all.py                                   # Cross-model evaluation
├── evaluate_extended.py                              # Multi-noise (AWGN, Pink, Hum) evaluation
├── evaluate_logmel.py                                # Log-Mel noise evaluation
├── predict.py                                        # Command-line prediction utility
├── generate_report.py                                # Generates Word report (.docx)
├── EmotionSense_Improved_Mini_Project_Report.docx    # Complete 19-section academic report
├── config.py                                         # Global configuration and hyperparameters
├── requirements.txt                                  # Environment dependencies
├── README.md                                         # Project documentation
├── data/                                             # RAVDESS dataset (Actor_01 to Actor_24)
├── models/                                           # Saved model weights & label mappings
│   ├── emotionsense_cnn_lstm.keras                   # Original CNN-LSTM model
│   ├── improved_cnn_lstm.keras                       # Improved CNN-LSTM model
│   ├── logmel_cnn_lstm.keras                         # Log-Mel CNN-LSTM model
│   ├── svm_model.pkl                                 # Trained SVM baseline
│   ├── svm_scaler.pkl                                # Fitted StandardScaler
│   ├── split_actors.json                             # Disjoint actor split configuration
│   └── *_label_mapping.json                          # Class label mappings
├── outputs/                                          # Evaluation metrics, reports & plots
│   ├── all_models_comparison.csv                     # Master comparison table across models
│   ├── extended_noise_comparison.csv                 # Multi-noise comparison table
│   ├── snr_accuracy_extended.png                     # Accuracy vs SNR across noise types
│   ├── snr_f1_extended.png                           # Macro-F1 vs SNR across noise types
│   ├── logmel_learning_curves.png                    # Log-Mel training curves
│   └── confusion_matrix_*.png                        # Confusion matrices
├── src/                                              # Source modules
│   ├── dataset.py                                    # RAVDESS filename parser & discovery
│   ├── preprocessing.py                              # MFCC, Log-Mel, and SVM feature extraction
│   ├── model.py                                      # CNN-LSTM model definition
│   ├── noise.py                                      # AWGN, Pink, and 50 Hz Hum generators
│   └── metrics.py                                    # Metric calculation utilities
└── tests/                                            # Pytest test suite (38 unit tests)
```

---

## 📄 Academic References
1. S. R. Livingstone and F. A. Russo, "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English," *PLoS ONE*, vol. 13, no. 5, p. e0196391, 2018.
2. M. B. Akçay and K. Oğuz, "Speech emotion recognition: Emotional models, databases, features, preprocessing methods, supporting modalities, and classifiers," *Speech Communication*, vol. 116, pp. 56–76, 2020.
3. B. Schuller et al., "The INTERSPEECH 2009 Emotion Challenge," in *Interspeech 2009*, Brighton, UK, 2009, pp. 312–315.
