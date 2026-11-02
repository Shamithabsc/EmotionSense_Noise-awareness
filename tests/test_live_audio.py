"""Tests for live microphone audio processing, buffer conversion, and prediction pipeline compatibility."""
import io
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import tensorflow as tf

from config import SAMPLE_RATE, N_MFCC, MAX_FRAMES, MODEL_DIR
from src.preprocessing import (
    load_audio_from_buffer,
    extract_mfcc,
    extract_mfcc_with_deltas,
    audio_quality,
)
from predict import predict_file


def _generate_sine_wav_bytes(duration_sec: float = 1.0, sr: int = 16000, channels: int = 1) -> bytes:
    """Helper to generate in-memory WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
    if channels == 2:
        sig = np.column_stack([sig, sig])
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def test_load_audio_from_buffer_mono():
    """Verify loading mono audio bytes produces float32 array at project sample rate."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=1.0, sr=16000, channels=1)
    audio = load_audio_from_buffer(wav_bytes)
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE


def test_load_audio_from_buffer_stereo():
    """Verify stereo audio bytes are automatically downmixed to 1D mono."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=1.0, sr=16000, channels=2)
    audio = load_audio_from_buffer(wav_bytes)
    assert audio.ndim == 1
    assert len(audio) == SAMPLE_RATE


def test_load_audio_from_buffer_resampling():
    """Verify 44.1 kHz input (common for browser mics) is correctly resampled to 16 kHz."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=1.0, sr=44100, channels=1)
    audio = load_audio_from_buffer(wav_bytes, sample_rate=SAMPLE_RATE)
    assert len(audio) == SAMPLE_RATE
    assert np.isfinite(audio).all()


def test_load_audio_from_buffer_empty():
    """Verify empty buffer raises ValueError."""
    with pytest.raises(ValueError, match="empty"):
        load_audio_from_buffer(b"")


def test_load_audio_from_buffer_none():
    """Verify None input raises ValueError."""
    with pytest.raises(ValueError, match="None"):
        load_audio_from_buffer(None)


def test_live_audio_feature_shapes():
    """Verify extracted features match exact model dimensions."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=1.5, sr=16000)
    audio = load_audio_from_buffer(wav_bytes)

    # Original model MFCC
    mfcc = extract_mfcc(audio)
    assert mfcc.shape == (MAX_FRAMES, N_MFCC, 1)
    assert np.isfinite(mfcc).all()

    # Delta MFCC
    mfcc_deltas = extract_mfcc_with_deltas(audio)
    assert mfcc_deltas.shape == (MAX_FRAMES, N_MFCC * 3, 1)
    assert np.isfinite(mfcc_deltas).all()


def test_audio_quality_analysis():
    """Verify audio quality diagnostics return valid metrics."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=2.0, sr=16000)
    audio = load_audio_from_buffer(wav_bytes)
    metrics = audio_quality(audio)

    assert "duration_seconds" in metrics
    assert "rms_amplitude" in metrics
    assert "clipping_fraction" in metrics
    assert "estimated_snr_db" in metrics
    assert "silence_like" in metrics
    assert 1.9 <= metrics["duration_seconds"] <= 2.1
    assert not metrics["silence_like"]
    assert metrics["clipping_fraction"] == 0.0


def test_audio_quality_silence_detection():
    """Verify silence_like flags true on flat zero audio."""
    silent = np.zeros(16000, dtype=np.float32)
    metrics = audio_quality(silent)
    assert metrics["silence_like"] is True
    assert metrics["quality_status"] == "Silent or near-silent"


@pytest.mark.skipif(
    not (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists(),
    reason="CNN-LSTM model not present"
)
def test_live_audio_model_prediction(tmp_path):
    """Verify end-to-end forward pass on microphone audio file."""
    wav_bytes = _generate_sine_wav_bytes(duration_sec=2.0, sr=16000)
    temp_file = tmp_path / "mic_test.wav"
    temp_file.write_bytes(wav_bytes)

    res = predict_file(temp_file)
    assert "predicted_emotion" in res
    assert "softmax_confidence" in res
    assert "class_probabilities" in res
    assert len(res["class_probabilities"]) == 8
    assert 0.0 <= res["softmax_confidence"] <= 1.0
