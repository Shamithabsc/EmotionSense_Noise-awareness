from __future__ import annotations

import numpy as np
import librosa

from config import SAMPLE_RATE, N_MFCC, MAX_FRAMES

# Feature dimension for the delta-augmented CNN-LSTM: static + delta + delta-delta
N_MFCC_DELTA = N_MFCC * 3  # 120 when N_MFCC=40
# Feature dimension for the SVM baseline: mean + std + min + max per coefficient
SVM_FEATURE_DIM = N_MFCC * 4  # 160 when N_MFCC=40
# Log-Mel spectrogram number of mel bands
LOG_MEL_N_MELS = 64  # matches mel bank used in MFCCs (n_mels=64); input becomes (200, 64, 1)


def load_audio(path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 at the project's target sample rate."""
    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty or unreadable: {path}")
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return audio


def extract_mfcc(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """Return a fixed-size (max_frames, n_mfcc, 1) MFCC tensor."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot extract features from empty audio.")
    if np.max(np.abs(audio)) < 1e-8:
        # Keep silence valid and finite; model should not be expected to classify it well.
        features = np.zeros((n_mfcc, max_frames), dtype=np.float32)
    else:
        features = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=160,
            n_mels=64,
        )
        if features.shape[1] < max_frames:
            features = np.pad(
                features,
                ((0, 0), (0, max_frames - features.shape[1])),
                mode="constant",
            )
        else:
            features = features[:, :max_frames]

        # Per-recording normalization reduces scale variation. Apply identically at inference.
        mean = features.mean(axis=1, keepdims=True)
        std = features.std(axis=1, keepdims=True)
        features = (features - mean) / np.maximum(std, 1e-6)

    features = features.T  # (time, coefficients)
    return features[..., np.newaxis].astype(np.float32)


def extract_mfcc_with_deltas(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """Return a (max_frames, n_mfcc*3, 1) tensor of MFCC + delta + delta-delta.

    Stacking temporal derivatives captures how features change over time, which
    substantially improves cross-speaker generalisation in SER.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot extract features from empty audio.")
    if np.max(np.abs(audio)) < 1e-8:
        return np.zeros((max_frames, n_mfcc * 3, 1), dtype=np.float32)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=n_mfcc,
        n_fft=512, hop_length=160, n_mels=64,
    )  # shape: (n_mfcc, frames)
    delta1 = librosa.feature.delta(mfcc, order=1)
    delta2 = librosa.feature.delta(mfcc, order=2)
    combined = np.concatenate([mfcc, delta1, delta2], axis=0)  # (n_mfcc*3, frames)

    # Pad / truncate time axis
    if combined.shape[1] < max_frames:
        combined = np.pad(combined, ((0, 0), (0, max_frames - combined.shape[1])), mode="constant")
    else:
        combined = combined[:, :max_frames]

    # Per-recording normalisation per coefficient (same as extract_mfcc)
    mean = combined.mean(axis=1, keepdims=True)
    std  = combined.std(axis=1,  keepdims=True)
    combined = (combined - mean) / np.maximum(std, 1e-6)

    return combined.T[..., np.newaxis].astype(np.float32)  # (max_frames, n_mfcc*3, 1)


def extract_svm_features(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
) -> np.ndarray:
    """Return a flat feature vector for classical ML (SVM / RF).

    Computes mean, standard deviation, minimum, and maximum across the time axis
    for each of the 40 MFCC coefficients, giving a 160-dimensional vector.
    This pooling is speaker-agnostic: the same statistics are computed whether
    the speaker has fast or slow speech rate, so it generalises better than a
    raw MFCC sequence on a small dataset.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot extract SVM features from empty audio.")
    if np.max(np.abs(audio)) < 1e-8:
        return np.zeros(n_mfcc * 4, dtype=np.float32)
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=n_mfcc,
        n_fft=512, hop_length=160, n_mels=64,
    )  # (n_mfcc, frames)
    feats = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),
        mfcc.min(axis=1),
        mfcc.max(axis=1),
    ]).astype(np.float32)
    return feats


def audio_quality(audio: np.ndarray) -> dict:
    """Simple signal checks; these are indicators, not a trained noise detector."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {"duration_seconds": 0.0, "peak_amplitude": 0.0,
                "clipping_fraction": 0.0, "silence_like": True}
    peak = float(np.max(np.abs(audio)))
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.999))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    return {
        "duration_seconds": float(audio.size / SAMPLE_RATE),
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "clipping_fraction": clipping_fraction,
        "silence_like": bool(rms < 1e-4),
    }


def extract_log_mel(
    audio: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    n_mels: int = LOG_MEL_N_MELS,
    max_frames: int = MAX_FRAMES,
) -> np.ndarray:
    """Return a fixed-size (max_frames, n_mels, 1) log-Mel spectrogram tensor.

    Uses identical STFT parameters (n_fft=512, hop_length=160) and the same
    padding/truncation and per-recording normalisation as extract_mfcc(), ensuring
    a fair feature comparison. Input shape (200, 64, 1) is compatible with the
    existing CNN-LSTM architecture.

    Notes
    -----
    - n_mels=64 matches the mel filter bank used inside extract_mfcc(), so the
      spectral resolution is identical.
    - The model must be retrained for log-Mel features; do NOT load MFCC
      checkpoints for inference with this function (shape mismatch: 64 vs 40).
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("Cannot extract log-Mel features from empty audio.")
    if np.max(np.abs(audio)) < 1e-8:
        return np.zeros((max_frames, n_mels, 1), dtype=np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
        n_mels=n_mels,
        fmax=8000,
    )  # shape: (n_mels, frames)
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Pad or truncate to max_frames
    if log_mel.shape[1] < max_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, max_frames - log_mel.shape[1])), mode="constant")
    else:
        log_mel = log_mel[:, :max_frames]

    # Per-recording normalisation (mean=0, std=1 per mel band) — same convention as MFCC
    mean = log_mel.mean(axis=1, keepdims=True)
    std  = log_mel.std(axis=1,  keepdims=True)
    log_mel = (log_mel - mean) / np.maximum(std, 1e-6)

    return log_mel.T[..., np.newaxis].astype(np.float32)  # (max_frames, n_mels, 1)
