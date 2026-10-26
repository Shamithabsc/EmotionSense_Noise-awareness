# EmotionSense: Noise-Aware Speech Emotion Recognition Using CNN-LSTM

A polished academic mini-project demonstrating speech emotion recognition with a CNN-LSTM model, MFCC features, and synthetic noise robustness evaluation on the RAVDESS dataset.

[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)](https://www.python.org/)
[![TensorFlow 2.15–2.18](https://img.shields.io/badge/tensorflow-2.15%E2%80%932.18-orange)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.32%E2%80%931.44-red)](https://streamlit.io/)

---

## Project overview

EmotionSense trains a compact CNN-LSTM on 40-coefficient MFCC sequences extracted from RAVDESS speech recordings, using an actor-independent train/validation/test split to avoid speaker leakage. The unique feature is a **noise robustness evaluation**: the same held-out test set is evaluated four times — clean, 20 dB SNR, 10 dB SNR, and 0 dB SNR — using synthetic additive white Gaussian noise. All results are displayed in the **EmotionSense Studio** Streamlit dashboard.

---

## Requirements

- **Python 3.10, 3.11, or 3.12** (3.12 confirmed working)
- CPU-only Windows environment (no GPU required)
- ~2 GB disk space for the RAVDESS dataset

---

## Step 1 — Virtual environment and dependencies

Open PowerShell in the project folder:

```powershell
# Create venv (use your installed Python)
py -3.12 -m venv .venv

# Activate  (if blocked, first run: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass)
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

> **Note:** Streamlit is pinned to `<1.45` — version 1.60 has a starlette import bug. The pin is already in `requirements.txt`.

---

## Step 2 — Download the RAVDESS dataset

1. Open **https://zenodo.org/records/1188976** in a browser.
2. Download **Audio_Speech_Actors_01-24.zip** (speech-only, ~215 MB).
3. Extract so the actor folders are directly under `data/ravdess/`:

```
data/
└── ravdess/
    ├── Actor_01/
    │   ├── 03-01-01-01-01-01-01.wav
    │   └── ...
    ├── Actor_02/
    └── ...
```

4. Verify the discovery:

```powershell
python scripts/inspect_dataset.py
```

Expected output: 1440 valid audio files across 24 actors and up to 8 emotion classes.

---

## Step 3 — Train the model

```powershell
python train.py
```

- Uses a **speaker-independent split**: 20% test actors, 20% validation actors, 60% training actors (grouped by actor ID, not individual files).
- Trains for up to 60 epochs with early stopping (patience 8).
- Saves to `models/emotionsense_cnn_lstm.keras` and `models/label_mapping.json`.
- Saves `outputs/test_manifest.csv` for evaluation.
- Typical CPU training time: **10–25 minutes**.

The script **stops with a clear message** if the dataset is missing.

---

## Step 4 — Evaluate clean and noisy conditions

```powershell
python evaluate.py
```

Evaluates the held-out test recordings under:
- **Clean** — original audio
- **20 dB SNR** — light synthetic white noise
- **10 dB SNR** — moderate noise
- **0 dB SNR** — severe noise (signal power equals noise power)

Produces in `outputs/`:
- `noise_evaluation_metrics.csv` — accuracy, precision, recall, F1, deltas
- `classification_report_{condition}.csv` — per-class scores for each condition
- `confusion_matrix_{condition}.png` — heatmaps
- `accuracy_by_noise.png`, `macro_f1_by_noise.png` — bar charts

> Synthetic white noise is a controlled approximation; it does not reproduce real babble, traffic, or music noise.

---

## Step 5 — Launch the dashboard

```powershell
streamlit run app.py
```

The **EmotionSense Studio** dashboard opens in your browser with three pages:
1. **Analyze Audio** — upload a WAV/MP3/FLAC/OGG file, see waveform, Mel spectrogram, predicted emotion, class probability bars, and an emotion timeline for longer recordings.
2. **Noise Robustness** — view accuracy/F1 charts, full metrics table, and confusion matrices from the evaluation.
3. **Setup Guide** — step-by-step instructions and current project status checklist.

The model **does not retrain when the dashboard starts**. It loads the saved `.keras` file.

---

## Step 6 — Run unit tests

```powershell
pytest -q
```

Tests cover preprocessing, noise injection, and a synthetic model round-trip.

---

## Predict from command line

```powershell
python predict.py path/to/speech.wav
```

Returns JSON with the predicted emotion, softmax confidence, and all class probabilities.

---

## Project structure

```
EmotionSense/
├── app.py                        # Streamlit dashboard (EmotionSense Studio)
├── train.py                      # Dataset loading, actor split, model training
├── evaluate.py                   # Clean + noisy evaluation, metrics, plots
├── predict.py                    # Single-file prediction utility
├── config.py                     # Paths and hyperparameters
├── requirements.txt
├── README.md
├── data/
│   └── ravdess/                  # Place RAVDESS Actor_* folders here
├── models/
│   ├── emotionsense_cnn_lstm.keras
│   ├── label_mapping.json
│   └── split_actors.json
├── outputs/
│   ├── noise_evaluation_metrics.csv
│   ├── confusion_matrix_*.png
│   ├── classification_report_*.csv
│   ├── accuracy_by_noise.png
│   ├── macro_f1_by_noise.png
│   └── training_history.csv
├── reports/
│   └── project_report.md
├── scripts/
│   └── inspect_dataset.py
├── src/
│   ├── dataset.py                # RAVDESS filename parser and discoverer
│   ├── preprocessing.py          # Audio loading, MFCC extraction, quality checks
│   ├── model.py                  # CNN-LSTM architecture
│   ├── noise.py                  # White noise injection at target SNR
│   └── metrics.py                # Accuracy, precision, recall, F1, confusion matrix
└── tests/
    ├── test_preprocessing.py
    ├── test_noise.py
    └── test_model.py
```

---

## CNN-LSTM architecture summary

```
Input (200, 40, 1)  →  Reshape (200, 40)
Conv1D(64, k=5)     →  BatchNorm  →  MaxPool(2)
Conv1D(96, k=3)     →  BatchNorm  →  MaxPool(2)
LSTM(64, dropout=0.2)
Dense(64, relu)     →  Dropout(0.35)
Dense(N, softmax)                    # N = number of discovered emotion classes
```

Total parameters: ~78 K (trains in minutes on CPU).

---

## RAVDESS emotion codes

```
01 neutral  02 calm  03 happy  04 sad
05 angry    06 fearful  07 disgust  08 surprised
```

The speech-only subset (modality code 03) contains all 8 classes. The actual classes used are derived from the discovered files and saved to `models/label_mapping.json`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No RAVDESS files found` | Check that `data/ravdess/Actor_01/` exists with `.wav` files |
| `Trained model not found` | Run `python train.py` first |
| `ImportError: starlette` | Run `pip install "streamlit>=1.32,<1.45"` |
| Streamlit page is blank | Ensure no syntax errors in `app.py`; restart with `streamlit run app.py` |
| `ml_dtypes` AttributeError on import | Benign TensorFlow/ml_dtypes version mismatch; does not affect model training |

---

## Academic references

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
- RAVDESS dataset: https://zenodo.org/records/1188976

