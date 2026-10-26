from __future__ import annotations
import numpy as np


def add_white_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise at a target SNR. Returns float32 audio."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot add noise to empty audio.")
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        # SNR is undefined for silent signals; preserve silence rather than fabricate signal.
        return x.copy()
    noise = np.random.default_rng(seed).normal(0.0, 1.0, size=x.shape)
    noise_power = float(np.mean(noise ** 2))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return (x + noise.astype(np.float32)).astype(np.float32)
