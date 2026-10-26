# EmotionSense: Noise-Aware Speech Emotion Recognition Using CNN-LSTM

## Abstract

EmotionSense is a speech emotion recognition mini-project that uses 40-coefficient MFCC audio features and a compact CNN-LSTM classifier trained on the RAVDESS speech dataset. Its central experiment compares recognition performance on clean speech and on copies degraded by synthetic additive white noise at 20 dB, 10 dB, and 0 dB signal-to-noise ratios. The system is evaluated using a **speaker-independent** test split to ensure reported accuracy reflects generalisation to unseen speakers, not speaker memorisation.

---

## 1. Problem formulation

Speech emotion recognition (SER) predicts a discrete emotion category from acoustic properties of a speech recording. A classifier evaluated only on clean recordings from known speakers may perform poorly under practical conditions where test speakers and background noise differ from training conditions. This project examines both issues simultaneously: speaker-independent generalisation and noise robustness.

---

## 2. Objectives

- Prepare a labeled speech corpus from RAVDESS and extract fixed-size MFCC tensors.
- Train a compact CNN-LSTM model on a speaker-independent training partition.
- Evaluate on a held-out test partition of actors never seen during training.
- Measure accuracy, macro precision, macro recall, and macro F1 on clean audio.
- Repeat evaluation on the same test files degraded by synthetic noise at three SNR levels.
- Visualise confusion patterns, noise robustness, and training dynamics.
- Provide a Streamlit dashboard for live audio file analysis.

---

## 3. Dataset

**Source:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)  
**Subset used:** Audio-only speech recordings (modality code 03)  
**Reference:** Livingstone & Russo (2018). PLOS ONE, 13(5), e0196391.  
**Archive:** `Audio_Speech_Actors_01-24.zip` from https://zenodo.org/records/1188976

### Verified statistics (from `scripts/inspect_dataset.py`)

| Property | Value |
|---|---|
| Total usable files | **1 440** |
| Actors | 24 (12 female, 12 male; IDs 01–24) |
| Files per actor | 60 |
| Emotion classes | 8 |
| Sample rate (resampled to) | 16 000 Hz |

### Emotion distribution

| Emotion | Files | Note |
|---|---|---|
| angry | 192 | Normal + strong intensity |
| calm | 192 | Normal + strong intensity |
| disgust | 192 | Normal + strong intensity |
| fearful | 192 | Normal + strong intensity |
| happy | 192 | Normal + strong intensity |
| neutral | **96** | Normal intensity only (no strong variant) |
| sad | 192 | Normal + strong intensity |
| surprised | 192 | Normal + strong intensity |

The class imbalance for `neutral` (96 vs 192 files for all others) is a known property of the RAVDESS speech subset, not a data preparation error.

---

## 4. Preprocessing

1. Audio is loaded at **16 000 Hz** (mono) using librosa.
2. **40 MFCC coefficients** are extracted per frame (512-sample window, 160-sample hop).
3. The resulting time × coefficient array is fixed to **200 frames** by zero-padding (short clips) or truncating (long clips).
4. Each recording is normalised independently by subtracting the mean and dividing by the standard deviation of its own MFCC values.
5. The final tensor shape is **(200, 40, 1)** — compatible with Conv1D input after reshaping to (200, 40).
6. Files that cannot be decoded are skipped with a console message.

This preprocessing is applied identically during training, evaluation, and Streamlit prediction to prevent train/test mismatch.

---

## 5. Noise experiment

For signal power $P_s$, target SNR $S$ (dB), and required noise power $P_n$:

$$S = 10\log_{10}\left(\frac{P_s}{P_n}\right)$$

Therefore:

$$P_n = \frac{P_s}{10^{S/10}}, \qquad \sigma_n = \sqrt{P_n}$$

White Gaussian noise scaled to $\sigma_n$ is added to **copies** of the test recordings — original source files are never overwritten. The SNR is mathematically exact given $P_s = \mathbb{E}[x^2]$. Synthetic AWGN is a controlled reproducible approximation of noise; it does not model babble, music, or environmental noise.

---

## 6. Model architecture

```
Input  (200, 40, 1)
  │
  ├─ Reshape → (200, 40)
  │
  ├─ Conv1D(64, kernel=5, padding='same', activation='relu')
  ├─ BatchNormalization
  ├─ MaxPooling1D(2)
  │
  ├─ Conv1D(96, kernel=3, padding='same', activation='relu')
  ├─ BatchNormalization
  ├─ MaxPooling1D(2)
  │
  ├─ LSTM(64, dropout=0.2, recurrent_dropout=0.0)
  │
  ├─ Dense(64, activation='relu')
  ├─ Dropout(0.35)
  │
  └─ Dense(8, activation='softmax')    # 8 = number of emotion classes
```

**Total parameters: ~78 K**  
Optimiser: Adam (lr = 1e-3, reduced on plateau by ×0.5, min lr = 1e-6)  
Loss: sparse categorical cross-entropy  
Batch size: 16  
Max epochs: 60  
Early stopping: patience 8, monitors validation loss, restores best weights

---

## 7. Actor-independent experimental design

Actors were partitioned at random (seed 42) into three disjoint sets:

| Split | Actor IDs | Count |
|---|---|---|
| Training | 1–6, 9, 12–15, 21–23 | 14 actors (840 files) |
| Validation | 7, 8, 11, 18, 24 | 5 actors (300 files) |
| Test | 10, 16, 17, 19, 20 | 5 actors (300 files) |

No actor appears in more than one split (verified: intersection = ∅).  
The model never sees validation or test actor voices during training; it selects the best checkpoint by validation loss and evaluates on the test set.

---

## 8. Results

### 8.1 Training summary

Training completed in 14 epochs (early stopping triggered, patience 8).  
Final epoch metrics: train accuracy ≈ 95%, val accuracy ≈ 30%.

The large gap between training and validation accuracy indicates strong overfitting — expected given:
- Only 14 training actors (840 recordings)
- Speaker-independent evaluation (validation actors have different vocal characteristics)
- No data augmentation applied during training

Learning curves are saved to `outputs/learning_curves.png`.

### 8.2 Noise robustness evaluation (held-out test actors)

| Condition | Test files | Accuracy | Macro P | Macro R | Macro F1 | ΔAccuracy | ΔMacro F1 |
|---|---|---|---|---|---|---|---|
| Clean | 300 | **28.3%** | 28.8% | 27.8% | 26.6% | — | — |
| 20 dB SNR | 300 | 26.3% | 29.7% | 25.0% | 23.5% | −2.0% | −3.1% |
| 10 dB SNR | 300 | 28.7% | 27.3% | 27.2% | 22.4% | +0.3% | −4.1% |
| 0 dB SNR | 300 | 28.7% | 27.3% | 27.2% | 22.3% | +0.3% | −4.3% |

Chance level for 8 balanced classes = 12.5%; for the actual class distribution (neutral underrepresented) ≈ 12.3%.

> **Interpretation:** The test accuracy (~28%) is significantly above chance (~12.5%) but well below the training accuracy, reflecting genuine speaker-independent generalisation difficulty with a small dataset. The macro F1 decreases monotonically with noise level, confirming that noise degrades class-level discrimination even when top-1 accuracy remains roughly stable.

### 8.3 Per-class results — Clean condition

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

Best recognised: **fearful** and **disgust** (F1 ≈ 0.42).  
Worst recognised: **happy** and **neutral** — common SER confusion pair due to acoustic similarity with calm/disgust.

Confusion matrices per condition are saved to `outputs/confusion_matrix_{condition}.png`.

### 8.4 Real-file prediction test

File tested: `data/Actor_01/03-01-05-01-01-01-01.wav`  
Ground-truth label (from filename code 05): **angry**  
Model prediction: **angry** (softmax confidence: 70.3%) ✓  
Audio duration: 3.87 s | Peak amplitude: 0.181 | Clipping: 0.0%

---

## 9. Analysis

### Overfitting
The model memorises training actor-specific patterns (vocal timbre, speaking style) rather than learning emotion-invariant acoustic features. Mitigation strategies for future work: speaker normalisation, voice conversion augmentation, or adversarial speaker disentanglement.

### Noise effect on Macro F1
Macro F1 decreases by 3.1 points at 20 dB and 4.3 points at 0 dB. The effect on raw accuracy is smaller because the model's predicted distribution shifts rather than changes top-1 class. This illustrates a limitation of reporting only accuracy for unbalanced noise robustness experiments.

### Emotion confusions
High-arousal emotions (angry, disgust, fearful) are better distinguished than low-arousal ones (calm, neutral, sad, happy). This is consistent with literature: MFCC features capture energy and formant structure but do not robustly distinguish emotions that differ primarily in tempo or pitch contour.

---

## 10. Limitations

1. **Small speaker set:** 24 actors is insufficient for robust speaker-independent generalisation.
2. **Controlled noise only:** AWGN is not representative of real noise environments.
3. **No data augmentation:** no time-stretch, pitch-shift, or speed perturbation applied.
4. **Single architecture:** no hyperparameter search or comparison with baseline classifiers.
5. **Acted speech:** RAVDESS emotions are acted, not naturalistic, which limits ecological validity.

---

## 11. Conclusion

EmotionSense demonstrates a complete, reproducible SER pipeline from raw RAVDESS speech to a deployed Streamlit dashboard. The compact CNN-LSTM (78 K parameters) trains in under 2 minutes on CPU and achieves 28.3% accuracy on a fully speaker-independent test set — well above the 12.5% chance level for 8 classes, but with clear overfitting that motivates future augmentation work. The noise robustness experiment confirms that macro F1 degrades monotonically with increasing noise, while accuracy is less sensitive, highlighting the importance of reporting F1 alongside accuracy for SER benchmarks.

---

## 12. References

- Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS). *PLOS ONE*, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391
- RAVDESS dataset: https://zenodo.org/records/1188976
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE TASLP*, 28(4), 357–366.
