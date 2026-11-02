import io
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


def load_audio_from_buffer(audio_source, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio from file path, raw bytes, or BytesIO buffer as mono float32.

    Handles:
      - Raw bytes or BytesIO stream (e.g. from Streamlit st.audio_input)
      - File paths / Path objects
      - Stereo-to-mono downmixing
      - Automatic resampling to target sample_rate (16,000 Hz)
      - Validation against empty or invalid content
    """
    if audio_source is None:
        raise ValueError("Audio source cannot be None.")

    if isinstance(audio_source, (bytes, bytearray)):
        if len(audio_source) == 0:
            raise ValueError("Audio buffer is empty.")
        audio_source = io.BytesIO(audio_source)
    elif hasattr(audio_source, "getvalue"):  # Streamlit UploadedFile
        data = audio_source.getvalue()
        if len(data) == 0:
            raise ValueError("Audio stream is empty.")
        audio_source = io.BytesIO(data)
    elif hasattr(audio_source, "read") and not hasattr(audio_source, "seek"):
        data = audio_source.read()
        if len(data) == 0:
            raise ValueError("Audio stream is empty.")
        audio_source = io.BytesIO(data)

    try:
        audio, _ = librosa.load(audio_source, sr=sample_rate, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio stream: {exc}") from exc

    if audio.size == 0:
        raise ValueError("Decoded audio stream contains zero samples.")

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

    # Per-recording normalisation on actual speech frames prior to padding
    mean = combined.mean(axis=1, keepdims=True)
    std  = combined.std(axis=1,  keepdims=True)
    combined = (combined - mean) / np.maximum(std, 1e-6)

    # Pad / truncate time axis with true zeros
    if combined.shape[1] < max_frames:
        combined = np.pad(combined, ((0, 0), (0, max_frames - combined.shape[1])), mode="constant")
    else:
        combined = combined[:, :max_frames]

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
    """Signal checks and heuristic quality metrics.

    Note: The estimated SNR is an acoustic energy-ratio heuristic based on frame energy
    percentiles (90th percentile speech vs 10th percentile background), not an instrument-grade
    SNR measurement.
    """
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return {
            "duration_seconds": 0.0,
            "peak_amplitude": 0.0,
            "rms_amplitude": 0.0,
            "clipping_fraction": 0.0,
            "silence_like": True,
            "estimated_noise_floor_rms": 0.0,
            "estimated_snr_db": 0.0,
            "quality_status": "Empty audio",
        }

    peak = float(np.max(np.abs(audio)))
    clipping_fraction = float(np.mean(np.abs(audio) >= 0.99))
    rms = float(np.sqrt(np.mean(np.square(audio))))
    silence_like = bool(rms < 1e-4 or peak < 1e-3)

    # Frame-level RMS energy for noise floor and SNR heuristic estimation
    frame_len = 1024
    hop_len = 256
    if audio.size >= frame_len:
        # Compute sliding frame energies
        n_frames = 1 + (audio.size - frame_len) // hop_len
        shape = (n_frames, frame_len)
        strides = (audio.strides[0] * hop_len, audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
        noise_floor_rms = float(np.percentile(frame_rms, 10))
        speech_level_rms = float(np.percentile(frame_rms, 90))
        estimated_snr = float(np.clip(20.0 * np.log10((speech_level_rms + 1e-6) / (noise_floor_rms + 1e-6)), 0.0, 50.0))
    else:
        noise_floor_rms = float(rms * 0.1)
        estimated_snr = 15.0

    if silence_like:
        quality_status = "Silent or near-silent"
    elif clipping_fraction > 0.02:
        quality_status = "Clipping distortion detected"
    elif rms < 0.01:
        quality_status = "Very quiet recording"
    elif estimated_snr < 10.0:
        quality_status = "High background noise"
    else:
        quality_status = "Good recording quality"

    return {
        "duration_seconds": float(audio.size / SAMPLE_RATE),
        "peak_amplitude": peak,
        "rms_amplitude": rms,
        "clipping_fraction": clipping_fraction,
        "silence_like": silence_like,
        "estimated_noise_floor_rms": noise_floor_rms,
        "estimated_snr_db": round(estimated_snr, 1),
        "quality_status": quality_status,
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
