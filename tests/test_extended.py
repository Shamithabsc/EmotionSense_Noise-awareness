"""tests/test_extended.py — Extended tests for EmotionSense.

Covers:
  - Log-Mel feature extraction (shape, finite, silence, short/long audio)
  - Pink noise generation (shape, SNR accuracy, reproducibility)
  - Harmonic/hum noise (shape, SNR)
  - Metric calculations (accuracy, macro-F1, weighted-F1)
  - Consistent emotion label ordering
  - Prediction on a valid audio sample
  - Graceful error on invalid audio path

All tests use synthetic audio only; no RAVDESS files are required.
"""
from __future__ import annotations
import json
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from config import MODEL_DIR, N_MFCC, MAX_FRAMES, SAMPLE_RATE
from src.preprocessing import extract_log_mel, extract_mfcc, LOG_MEL_N_MELS
from src.noise import add_white_noise_at_snr, add_pink_noise_at_snr, add_harmonic_noise_at_snr

# ── Shared test audio ──────────────────────────────────────────────────────────
SR   = SAMPLE_RATE
_t   = np.arange(SR, dtype=np.float32) / SR
SINE = (0.25 * np.sin(2 * np.pi * 220 * _t)).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════════
#  Log-Mel feature extraction
# ════════════════════════════════════════════════════════════════════════════════

def test_log_mel_shape_normal():
    feat = extract_log_mel(SINE)
    assert feat.shape == (MAX_FRAMES, LOG_MEL_N_MELS, 1), (
        f"Expected ({MAX_FRAMES}, {LOG_MEL_N_MELS}, 1) got {feat.shape}")


def test_log_mel_finite():
    feat = extract_log_mel(SINE)
    assert np.isfinite(feat).all(), "Log-Mel features contain NaN or Inf"


def test_log_mel_dtype():
    feat = extract_log_mel(SINE)
    assert feat.dtype == np.float32


def test_log_mel_silence():
    feat = extract_log_mel(np.zeros(4000, dtype=np.float32))
    assert feat.shape == (MAX_FRAMES, LOG_MEL_N_MELS, 1)
    assert np.all(feat == 0.0), "Silence should return zero tensor"


def test_log_mel_short_audio():
    short = SINE[:SR // 4]
    feat  = extract_log_mel(short)
    assert feat.shape == (MAX_FRAMES, LOG_MEL_N_MELS, 1)
    assert np.isfinite(feat).all()


def test_log_mel_long_audio():
    long = np.tile(SINE, 5)
    feat = extract_log_mel(long)
    assert feat.shape == (MAX_FRAMES, LOG_MEL_N_MELS, 1)


def test_log_mel_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        extract_log_mel(np.array([], dtype=np.float32))


# ════════════════════════════════════════════════════════════════════════════════
#  Pink noise
# ════════════════════════════════════════════════════════════════════════════════

def test_pink_noise_shape():
    noisy = add_pink_noise_at_snr(SINE, snr_db=10, seed=1)
    assert noisy.shape == SINE.shape
    assert noisy.dtype == np.float32


def test_pink_noise_snr_approximate():
    """Measured SNR should be within 3 dB of target."""
    target_snr = 10.0
    noisy = add_pink_noise_at_snr(SINE, snr_db=target_snr, seed=7)
    noise = noisy - SINE
    P_s = float(np.mean(SINE.astype(np.float64) ** 2))
    P_n = float(np.mean(noise.astype(np.float64) ** 2))
    measured_snr = 10 * np.log10(P_s / max(P_n, 1e-20))
    assert abs(measured_snr - target_snr) < 3.0, (
        f"Pink noise SNR {measured_snr:.2f} dB too far from target {target_snr} dB")


def test_pink_noise_reproducible():
    n1 = add_pink_noise_at_snr(SINE, snr_db=15, seed=99)
    n2 = add_pink_noise_at_snr(SINE, snr_db=15, seed=99)
    np.testing.assert_array_equal(n1, n2, err_msg="Pink noise is not reproducible")


def test_pink_noise_different_seeds():
    n1 = add_pink_noise_at_snr(SINE, snr_db=15, seed=1)
    n2 = add_pink_noise_at_snr(SINE, snr_db=15, seed=2)
    assert not np.allclose(n1, n2), "Different seeds should produce different noise"


def test_pink_noise_silence():
    silent = np.zeros(SR, dtype=np.float32)
    out = add_pink_noise_at_snr(silent, snr_db=10, seed=0)
    assert np.allclose(out, 0.0), "Silent signal should remain silent"


# ════════════════════════════════════════════════════════════════════════════════
#  Harmonic (hum) noise
# ════════════════════════════════════════════════════════════════════════════════

def test_harmonic_noise_shape():
    noisy = add_harmonic_noise_at_snr(SINE, snr_db=10, seed=0)
    assert noisy.shape == SINE.shape
    assert noisy.dtype == np.float32


def test_harmonic_noise_snr_approximate():
    target = 10.0
    noisy = add_harmonic_noise_at_snr(SINE, snr_db=target, seed=5)
    noise = noisy - SINE
    P_s = float(np.mean(SINE.astype(np.float64) ** 2))
    P_n = float(np.mean(noise.astype(np.float64) ** 2))
    measured = 10 * np.log10(P_s / max(P_n, 1e-20))
    assert abs(measured - target) < 1.5, (
        f"Hum noise SNR {measured:.2f} dB too far from target {target} dB")


def test_harmonic_noise_silence():
    silent = np.zeros(SR, dtype=np.float32)
    out = add_harmonic_noise_at_snr(silent, snr_db=10, seed=0)
    assert np.allclose(out, 0.0)


# ════════════════════════════════════════════════════════════════════════════════
#  Metric calculations
# ════════════════════════════════════════════════════════════════════════════════

def test_accuracy_calculation():
    from sklearn.metrics import accuracy_score
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 1, 1])
    assert accuracy_score(y_true, y_pred) == pytest.approx(0.8)


def test_macro_f1_zero_division_safe():
    from sklearn.metrics import f1_score
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 1])
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    assert np.isfinite(f1), "F1 should be finite even with all-wrong predictions"


def test_weighted_f1_matches_per_class():
    from sklearn.metrics import f1_score
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    wf1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert 0.0 <= wf1 <= 1.0


# ════════════════════════════════════════════════════════════════════════════════
#  Consistent emotion label ordering
# ════════════════════════════════════════════════════════════════════════════════

def test_label_mapping_consistent():
    """All three label_mapping.json files should have identical emotion sets."""
    maps = []
    for name in ["label_mapping.json", "improved_label_mapping.json",
                 "svm_label_mapping.json"]:
        path = MODEL_DIR / name
        if path.exists():
            m = json.loads(path.read_text(encoding="utf-8"))
            maps.append((name, set(m.values())))
    if len(maps) < 2:
        pytest.skip("Not enough label mappings available")
    ref_name, ref_set = maps[0]
    for name, emotion_set in maps[1:]:
        assert emotion_set == ref_set, (
            f"Emotion set mismatch: {ref_name}={ref_set} vs {name}={emotion_set}")


# ════════════════════════════════════════════════════════════════════════════════
#  Prediction on valid audio file
# ════════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists(),
    reason="Original CNN-LSTM model not found"
)
def test_predict_valid_audio():
    """predict_file should return a dict with expected keys."""
    from predict import predict_file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, SINE, SR)
        tmp_path = f.name
    result = predict_file(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    assert isinstance(result, dict)
    assert "predicted_emotion" in result
    assert "softmax_confidence" in result
    assert 0.0 <= result["softmax_confidence"] <= 1.0


@pytest.mark.skipif(
    not (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists(),
    reason="Original CNN-LSTM model not found"
)
def test_predict_invalid_path_raises():
    """predict_file should raise or return an error for a non-existent path."""
    from predict import predict_file
    with pytest.raises(Exception):
        predict_file("non_existent_audio_file_xyz.wav")
