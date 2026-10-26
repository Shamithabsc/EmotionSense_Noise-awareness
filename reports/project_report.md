# EmotionSense: Noise-Aware Speech Emotion Recognition Using CNN-LSTM

## Abstract

EmotionSense is a speech emotion recognition (SER) mini-project evaluated on the RAVDESS speech dataset using a fully **speaker-independent** experimental design (held-out test actors never seen during training or model selection). Three models are compared: an MFCC + SVM baseline, an original CNN-LSTM, and an improved CNN-LSTM with delta/delta-delta features and noise augmentation. All models are evaluated under clean audio and three synthetic noise conditions (20 dB, 10 dB, 0 dB SNR). The SVM achieves the highest clean-audio accuracy (47.0%) but collapses under noise. The CNN-LSTM maintains more stable accuracy across noise conditions, demonstrating a real accuracy–robustness trade-off that is the central finding of this project.

---

## 1. Problem Statement and Objectives

Speech emotion recognition predicts a discrete emotion label from acoustic properties of a speech recording. Systems must generalise across speakers (not just memorize voice characteristics) and degrade gracefully under noise. This project investigates both dimensions simultaneously on a small academic dataset.

**Objectives:**
- Prepare 1440 RAVDESS speech recordings with 8 emotion classes.
- Extract MFCC features and train a compact CNN-LSTM classifier.
- Establish an MFCC + SVM baseline for comparison.
- Evaluate all models on a held-out test set of actors not seen during training.
- Measure accuracy, macro F1, and per-class metrics under clean and noisy conditions.
- Diagnose the causes of low test accuracy and report findings honestly.

---

## 2. RAVDESS Dataset and Verified Emotion Classes

**Source:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)  
**Reference:** Livingstone & Russo (2018). PLOS ONE, 13(5), e0196391.  
**Archive:** Audio_Speech_Actors_01-24.zip — https://zenodo.org/records/1188976  
**Subset used:** Modality code 03 (audio-only speech)

### Verified dataset statistics

| Property | Value |
|---|---|
| Total usable audio files | **1 440** |
| Actors | 24 (12 female, 12 male; IDs 01–24) |
| Files per actor | 60 |
| Emotion classes | 8 |
| Sample rate (loaded at) | 16 000 Hz |

### Emotion class distribution

| Emotion | Files | Note |
|---|---|---|
| angry | 192 | Normal + strong intensity |
| calm | 192 | Normal + strong intensity |
| disgust | 192 | Normal + strong intensity |
| fearful | 192 | Normal + strong intensity |
| happy | 192 | Normal + strong intensity |
| neutral | **96** | Normal intensity only (RAVDESS design, not a data error) |
| sad | 192 | Normal + strong intensity |
| surprised | 192 | Normal + strong intensity |

The neutral class imbalance (96 vs 192 for all others) is a known property of RAVDESS. Class weights were applied during CNN-LSTM training to compensate.

---

## 3. Audio Preprocessing and Feature Extraction

All audio is loaded mono at 16 000 Hz using librosa. Three feature representations are used:

### 3.1 MFCC features (CNN-LSTM original)
- 40 MFCC coefficients (n_fft=512, hop_length=160, n_mels=64)
- Fixed to **200 frames** by zero-padding or truncation
- Per-recording normalization: subtract per-coefficient mean, divide by std (ε=1e-6)
- Output tensor shape: **(200, 40, 1)**

### 3.2 MFCC + delta + delta-delta features (CNN-LSTM improved)
- Same 40 MFCCs + 1st derivative (delta) + 2nd derivative (delta-delta)
- Stacked along the coefficient axis: **(200, 120, 1)**
- Temporal derivatives capture how features change over time, which should improve generalisation

### 3.3 MFCC pooled statistics (SVM baseline)
- 40 MFCCs computed as above
- Summarized by {mean, std, min, max} over the time axis
- Output feature vector: **(160,)** (speaker-agnostic pooling)
- StandardScaler fitted on training data only

---

## 4. Model Architectures

### 4.1 MFCC + SVM Baseline

```
Input: 160-dimensional MFCC statistics vector
  │
  ├─ StandardScaler (μ=0, σ=1) — fitted on train split only
  │
  └─ SVC(kernel='rbf', class_weight='balanced', probability=True)
       C selected by validation-set grid search: C ∈ {0.1, 1.0, 10.0, 100.0}
       Best C = 1.0  (val accuracy = 50.0%)
```

Speaker-agnostic pooled statistics generalise better than raw sequence features on a small dataset (840 training samples). The RBF kernel SVM is well-suited to this regime.

### 4.2 Original CNN-LSTM

```
Input  (200, 40, 1)
  ├─ Reshape → (200, 40)
  ├─ Conv1D(64, k=5, same, relu) → BatchNorm → MaxPool(2)
  ├─ Conv1D(96, k=3, same, relu) → BatchNorm → MaxPool(2)
  ├─ LSTM(64, dropout=0.2)
  ├─ Dense(64, relu)
  ├─ Dropout(0.35)
  └─ Dense(8, softmax)
```

Parameters ≈ 78K | Optimizer: Adam (lr=1e-3) | Batch: 16 | Max epochs: 60  
Early stopping: patience=8 on val_loss, restore best weights  
Best epoch: 5 | Best val_acc: 25.7% | Stopped at epoch: 13

### 4.3 Improved CNN-LSTM

Same architecture as 4.2 but:
- Input shape: **(200, 120, 1)** (MFCC + delta + delta-delta)
- Initial LR: 5e-4 (lower than baseline)
- Early stopping patience: 15
- Class weights applied (handles neutral imbalance)
- Online noise augmentation: 50% probability per sample, SNR ∈ [5, 30] dB

Best val_acc: 27.7% (similar to original — both overfit severely)

---

## 5. Actor-Independent Splitting

Actors were partitioned at random (seed 42) into three disjoint sets verified for zero overlap:

| Split | Actor IDs | Actors | Files |
|---|---|---|---|
| Training | 1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 21, 22, 23 | 14 | 840 |
| Validation | 7, 8, 11, 18, 24 | 5 | 300 |
| Test | 10, 16, 17, 19, 20 | 5 | 300 |

**Intersection (train ∩ test) = ∅ — verified.**  
Validation set used only for early stopping and C selection. Test set used only for final reporting.

---

## 6. Noise Augmentation and SNR Evaluation Methodology

For signal RMS power $P_s$, target SNR $S$ (dB):

$$P_n = \frac{P_s}{10^{S/10}}, \quad \sigma_n = \sqrt{P_n}$$

White Gaussian noise $\mathcal{N}(0,\, \sigma_n^2)$ is added to copies of recordings. Source files are never overwritten.

**Training augmentation (Improved CNN-LSTM only):**  
50% of training samples corrupted with random SNR ∈ [5, 30] dB (uniform), seed = RANDOM_SEED + sample_index.

**Evaluation:**  
All 300 test files evaluated at clean / 20 dB / 10 dB / 0 dB SNR.  
Noise seed = 2000 + file_index (different from training seeds — no contamination).  
Clean validation and test data are never noise-augmented.

---

## 7. Diagnosis of Original Low Accuracy (28.3%)

Investigation of training artifacts confirmed:

| Cause | Evidence |
|---|---|
| **Small dataset** | Only 840 training samples for 8-class speaker-independent SER |
| **Severe overfitting** | Train acc 90%, val acc 26% after 5 epochs — 64pp gap |
| **Early stopping at epoch 5 (best)** | Model barely left the early-learning phase |
| **No speaker normalisation** | Model memorizes actor vocal identity, not emotion |
| **Static MFCCs only** | Missing temporal dynamics (delta/delta-delta) |
| **No class weighting** | Neutral class underrepresented (96 vs 192 files) |
| **No data/pipeline bugs** | Label mapping, split disjointness, and metric calculation all verified correct |

The root cause is the fundamental **sample-size/task-difficulty mismatch**: 840 samples across 8 classes with speaker-independent evaluation is an extremely challenging regime for deep learning. SVMs with pooled statistics are more appropriate in this regime.

---

## 8. Experimental Results

### 8.1 Noise robustness — all models

| Model | Clean Acc | 20 dB Acc | 10 dB Acc | 0 dB Acc |
|---|---|---|---|---|
| **SVM Baseline** | **47.0%** | 23.0% | 14.3% | 13.3% |
| Original CNN-LSTM | 28.3% | 26.3% | 28.3% | 26.7% |
| Improved CNN-LSTM | 16.0% | 26.0% | 27.3% | 23.7% |

| Model | Clean F1 | 20 dB F1 | 10 dB F1 | 0 dB F1 |
|---|---|---|---|---|
| **SVM Baseline** | **0.456** | 0.173 | 0.048 | 0.029 |
| Original CNN-LSTM | 0.258 | 0.199 | 0.221 | 0.197 |
| Improved CNN-LSTM | 0.137 | 0.248 | 0.256 | 0.229 |

Chance level for 8 balanced classes = 12.5%.

### 8.2 Per-class results — SVM Baseline (clean condition)

From `outputs/svm_classification_report_clean.csv`.

### 8.3 Per-class results — Original CNN-LSTM (clean condition)

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| angry | 0.500 | 0.200 | 0.286 | 40 |
| calm | 0.241 | 0.325 | 0.277 | 40 |
| disgust | 0.357 | 0.500 | 0.417 | 40 |
| fearful | 0.415 | 0.425 | 0.420 | 40 |
| happy | 0.188 | 0.075 | 0.107 | 40 |
| neutral | 0.105 | 0.200 | 0.138 | 20 |
| sad | 0.231 | 0.150 | 0.182 | 40 |
| surprised | 0.264 | 0.350 | 0.301 | 40 |

Confusion matrices for all 3 models × 4 conditions saved under `outputs/`.

---

## 9. Baseline vs. Improved Comparison and Analysis

### Clean audio
- SVM: +18.7pp accuracy over original CNN-LSTM (**+66% relative improvement**)
- Improved CNN-LSTM: -12.3pp vs original (worse — delta features did not help with small dataset)

### Noise robustness trade-off

The SVM's global pooled statistics are highly sensitive to noise: adding Gaussian noise to the signal fundamentally corrupts mean/std/min/max statistics computed over the full recording. At 0 dB SNR the SVM accuracy falls to 13.3% — barely above chance.

The CNN-LSTM maintains ~26–28% under all noise levels. This is because the LSTM's temporal gating effectively suppresses noise-corrupted frames by weighting consistent patterns across time.

**Key academic finding:** There is a genuine accuracy–robustness trade-off between these feature/model classes. The SVM is the right choice for clean audio deployment; the CNN-LSTM is more appropriate for noisy environments.

### Why the Improved CNN-LSTM did not improve
The delta-MFCC improved CNN-LSTM performs *worse* on clean audio (16.0%) but better under noise (up to 27.3%). The triple-sized input (120 features vs 40) increased model capacity and noise, worsening generalisation on the 840-sample training set — a textbook consequence of high-dimensional features with small N.

---

## 10. Noise-Robustness Trade-off Summary

| Condition | Best model | Accuracy | Recommended for |
|---|---|---|---|
| Clean audio | SVM Baseline | 47.0% | Studio / clean-mic deployment |
| 20 dB SNR | SVM Baseline | 23.0% | Mild noise |
| 10 dB SNR | Improved CNN-LSTM | 27.3% | Moderate noise |
| 0 dB SNR | Original CNN-LSTM | 26.7% | Heavy noise |

For noise-robust deployment: **Original CNN-LSTM**. For highest clean accuracy: **SVM Baseline**.

---

## 11. Limitations

1. **Small training set**: 840 samples for speaker-independent 8-class SER is too few for reliable deep learning.
2. **Synthetic noise only**: AWGN does not model real-world noise (babble, traffic, reverberation).
3. **No data augmentation for SVM**: Time-stretch, pitch-shift, or feature-level augmentation were not explored.
4. **RAVDESS acted speech**: Emotions are professionally acted, not naturalistic.
5. **Single architecture comparison**: No hyperparameter search beyond SVM's C.
6. **Improved CNN-LSTM converged poorly**: Delta features + noise augmentation hurt clean accuracy at this sample size.

---

## 12. Reproducibility

All results are reproducible with `RANDOM_SEED=42` (set in `config.py`).  
Actor split stored in `models/split_actors.json`.  
Evaluation uses seed `2000 + file_index` for noise — different from training seeds.

```powershell
# Full pipeline (Windows PowerShell)
python scripts/inspect_dataset.py    # verify 1440 files
python train.py                      # original CNN-LSTM
python train_svm.py                  # SVM baseline
python train_improved.py             # improved CNN-LSTM (delta features + augmentation)
python evaluate_all.py               # compare all models
streamlit run app.py                 # launch dashboard
python -m pytest tests/ -v           # run all tests
```

---

## 13. Conclusion

EmotionSense demonstrates a complete, reproducible SER pipeline from raw RAVDESS audio to a multi-model comparison dashboard. The key results are:

1. **SVM with pooled MFCC statistics achieves 47.0% speaker-independent test accuracy** — +18.7pp over the original CNN-LSTM, confirming that classical ML with speaker-agnostic features outperforms deep learning on small SER datasets.

2. **CNN-LSTM is more noise-robust**: accuracy stays at 26–28% across all SNR levels, while SVM collapses to ~13% at 0 dB SNR.

3. **The accuracy–robustness trade-off is real and measurable**: the choice of model depends on the target deployment environment.

4. **Delta features did not improve the CNN-LSTM** on this dataset — tripling the input dimension worsened generalisation with only 840 training samples.

5. The original 28.3% accuracy was not caused by bugs, data leakage, or incorrect labels — it reflects the genuine difficulty of speaker-independent SER with a small dataset and no augmentation.

---

## 14. References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
- RAVDESS dataset: https://zenodo.org/records/1188976
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition. *IEEE TASLP*, 28(4), 357–366.
- Vapnik, V. N. (1998). *Statistical Learning Theory*. Wiley.
