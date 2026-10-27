"""src/noise.py — Noise generation utilities for EmotionSense.

Provides:
  add_white_noise_at_snr()  — additive white Gaussian noise (AWGN)
  add_pink_noise_at_snr()   — pink (1/f) noise via Voss-McCartney algorithm
  add_harmonic_noise_at_snr() — synthetic tone burst (machine hum approximation)

All functions:
  - Accept float32 audio arrays and return float32 arrays.
  - Are reproducible with an explicit seed.
  - Preserve silence (RMS < 1e-6) unchanged rather than fabricating signal.
  - Use the exact SNR definition:
      SNR_dB = 10 * log10(P_signal / P_noise)
      P_noise_target = P_signal / 10^(SNR_dB / 10)
"""
from __future__ import annotations
import numpy as np


def _scale_noise_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale *noise* so that the signal + noise mixture achieves *snr_db*."""
    signal_power = float(np.mean(signal.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        # Silence — undefined SNR; return zeros
        return np.zeros_like(noise, dtype=np.float32)
    noise_power = float(np.mean(noise.astype(np.float64) ** 2))
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / max(noise_power, 1e-12))
    return (noise * scale).astype(np.float32)


def add_white_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add white Gaussian noise at a target SNR. Returns float32 audio."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot add noise to empty audio.")
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        return x.copy()
    noise = np.random.default_rng(seed).normal(0.0, 1.0, size=x.shape).astype(np.float32)
    noise = _scale_noise_to_snr(x, noise, snr_db)
    return (x + noise).astype(np.float32)


def add_pink_noise_at_snr(audio: np.ndarray, snr_db: float, seed: int = 42) -> np.ndarray:
    """Add pink (1/f) noise at a target SNR.

    Pink noise approximated via the Voss-McCartney algorithm:
    sum of N octave-band white noise sequences, each updated at half the rate
    of the previous. This gives a spectrum that falls ~3 dB/octave.

    Pink noise is a better approximation of natural ambient sound than white
    noise but is still synthetic and does not model specific real environments.
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot add noise to empty audio.")
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        return x.copy()
    n = x.size
    rng = np.random.default_rng(seed)
    # Approximate pink noise via FFT shaping: generate white noise, shape in frequency domain
    white = rng.standard_normal(n).astype(np.float64)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid divide-by-zero at DC
    power_spectrum = 1.0 / np.sqrt(freqs)
    power_spectrum[0] = 0.0  # zero DC component
    spectrum = np.fft.rfft(white) * power_spectrum
    pink = np.fft.irfft(spectrum, n=n).astype(np.float32)
    pink = _scale_noise_to_snr(x, pink, snr_db)
    return (x + pink).astype(np.float32)


def add_harmonic_noise_at_snr(
    audio: np.ndarray,
    snr_db: float,
    seed: int = 42,
    fundamental_hz: float = 50.0,
    n_harmonics: int = 5,
    sample_rate: int = 16_000,
) -> np.ndarray:
    """Add synthetic harmonic (hum) noise at a target SNR.

    Generates a sum of sinusoids at the fundamental and first n_harmonics-1
    harmonics with uniformly random initial phases. This approximates
    electrical hum (50/60 Hz) or mechanical vibration, not real recorded noise.

    Parameters
    ----------
    fundamental_hz : float
        Fundamental frequency in Hz (default 50 Hz for mains hum).
    n_harmonics : int
        Number of harmonics to include (default 5).
    """
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot add noise to empty audio.")
    signal_power = float(np.mean(x.astype(np.float64) ** 2))
    if signal_power < 1e-12:
        return x.copy()
    rng = np.random.default_rng(seed)
    t = np.arange(len(x), dtype=np.float64) / sample_rate
    hum = np.zeros(len(x), dtype=np.float64)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        hum += np.sin(2 * np.pi * k * fundamental_hz * t + phase)
    hum = hum.astype(np.float32)
    hum = _scale_noise_to_snr(x, hum, snr_db)
    return (x + hum).astype(np.float32)
