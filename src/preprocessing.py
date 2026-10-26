from __future__ import annotations

import numpy as np
import librosa

from config import SAMPLE_RATE, N_MFCC, MAX_FRAMES


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
