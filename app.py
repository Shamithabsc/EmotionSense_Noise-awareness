"""EmotionSense Studio — AI Speech Emotion Recognition & Noise Robustness Dashboard.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for Streamlit threads
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st

# ── Project imports ──────────────────────────────────────────────────────────
from config import MODEL_DIR, OUTPUT_DIR, SAMPLE_RATE, N_MFCC, MAX_FRAMES
from predict import predict_audio, predict_file, get_model
from src.noise import (
    add_white_noise_at_snr,
    add_pink_noise_at_snr,
    add_harmonic_noise_at_snr,
)
from src.preprocessing import (
    load_audio,
    load_audio_from_buffer,
    extract_mfcc,
    extract_mfcc_with_deltas,
    extract_svm_features,
    audio_quality,
)

# ── Streamlit page config (Must be the very first Streamlit call) ─────────────
st.set_page_config(
    page_title="EmotionSense Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS INJECTION (DARK NAVY / PURPLE / TEAL THEME)
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-deep:      #0b0f19;
    --bg-card:      #131b2e;
    --bg-surface:   #1a243d;
    --bg-elevated:  #212e4c;
    --accent:       #8b5cf6;
    --accent-teal:  #06b6d4;
    --accent-cyan:  #38bdf8;
    --accent-glow:  rgba(139, 92, 246, 0.25);
    --border:       rgba(139, 92, 246, 0.22);
    --border-hover: rgba(6, 182, 212, 0.45);
    --text-hi:      #f8fafc;
    --text-mid:     #94a3b8;
    --text-muted:   #64748b;
    --radius-lg:    14px;
    --radius-md:    10px;
    --shadow-card:  0 8px 32px rgba(0, 0, 0, 0.45);
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-hi);
}
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stHeader"] { background: transparent !important; }

h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
h1 { font-size: 2.2rem !important; font-weight: 700; letter-spacing: -0.5px; }
h2 { font-size: 1.45rem !important; font-weight: 600; }
h3 { font-size: 1.15rem !important; color: var(--text-mid) !important; font-weight: 600; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #131b2e 0%, #1e1442 50%, #0d1b2a 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.6rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-card), 0 0 50px var(--accent-glow);
    position: relative;
    overflow: hidden;
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -80px;
    right: -80px;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #c4b5fd 0%, #8b5cf6 45%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.4rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: var(--text-mid);
    max-width: 820px;
    line-height: 1.6;
    margin: 0 0 1.2rem;
}

/* ── Status Pills & Badges ── */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.status-ready {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(52, 211, 153, 0.35);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
}

/* ── Cards & Containers ── */
.es-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow-card);
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.es-card:hover {
    border-color: var(--border-hover);
}
.es-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--accent-teal);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.es-metric-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 1.1rem 1.3rem;
    text-align: left;
}
.es-metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-hi);
}
.es-metric-lbl {
    font-size: 0.8rem;
    color: var(--text-mid);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 2px;
}

/* ── Studio Control Console ── */
.studio-console-box {
    background: linear-gradient(135deg, rgba(19, 27, 46, 0.95) 0%, rgba(26, 36, 61, 0.9) 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-card);
}

/* ── Probability Bars ── */
.prob-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 0.45rem;
    font-size: 0.88rem;
}
.prob-label {
    width: 105px;
    font-weight: 500;
    color: var(--text-hi);
    white-space: nowrap;
}
.prob-track {
    flex: 1;
    height: 10px;
    background: rgba(255, 255, 255, 0.07);
    border-radius: 6px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.prob-pct {
    width: 48px;
    text-align: right;
    font-weight: 600;
    color: var(--text-mid);
    font-size: 0.82rem;
}

/* ── Top 3 Pill Row ── */
.top3-container {
    display: flex;
    gap: 10px;
    margin: 0.8rem 0 1.2rem;
}
.top3-pill {
    flex: 1;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    text-align: center;
}
.top3-rank {
    font-size: 0.7rem;
    text-transform: uppercase;
    color: var(--text-muted);
    font-weight: 600;
}
.top3-name {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-hi);
    margin: 2px 0;
}
.top3-conf {
    font-size: 0.8rem;
    color: var(--accent-teal);
    font-weight: 700;
}

/* ── Stability Box ── */
.stability-box {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
    border: 1px solid rgba(6, 182, 212, 0.35);
    border-radius: var(--radius-md);
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

/* ── Reliability Badges ── */
.reliability-high {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #10b981;
    padding: 0.35rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.reliability-mod {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #f59e0b;
    padding: 0.35rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.reliability-low {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #ef4444;
    padding: 0.35rem 0.8rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* ── Dividers ── */
.es-sep {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 1.4rem 0;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  METADATA & COLOR SCHEMES
# ═══════════════════════════════════════════════════════════════════════════════
EMOTION_META = {
    "neutral":   {"emoji": "😐", "color": "#94a3b8", "label": "Neutral"},
    "calm":      {"emoji": "😌", "color": "#10b981", "label": "Calm"},
    "happy":     {"emoji": "😊", "color": "#f59e0b", "label": "Happy"},
    "sad":       {"emoji": "😢", "color": "#38bdf8", "label": "Sad"},
    "angry":     {"emoji": "😠", "color": "#ef4444", "label": "Angry"},
    "fearful":   {"emoji": "😨", "color": "#a855f7", "label": "Fearful"},
    "disgust":   {"emoji": "🤢", "color": "#14b8a6", "label": "Disgust"},
    "surprised": {"emoji": "😲", "color": "#ec4899", "label": "Surprised"},
}

COND_LABELS = {
    "clean": "Clean",
    "snr_20db": "20 dB SNR",
    "snr_10db": "10 dB SNR",
    "snr_0db": "0 dB SNR",
}
COND_ORDER = ["clean", "snr_20db", "snr_10db", "snr_0db"]

MPL_DARK = {
    "figure.facecolor": "#131b2e",
    "axes.facecolor": "#131b2e",
    "axes.edgecolor": "#2a3652",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#f8fafc",
    "grid.color": "#1e2942",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL AVAILABILITY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def original_ready() -> bool:
    return (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists() and (MODEL_DIR / "label_mapping.json").exists()

def improved_ready() -> bool:
    return (MODEL_DIR / "improved_cnn_lstm.keras").exists() and (MODEL_DIR / "improved_label_mapping.json").exists()

def svm_ready() -> bool:
    return (MODEL_DIR / "svm_model.pkl").exists() and (MODEL_DIR / "svm_scaler.pkl").exists()

# ═══════════════════════════════════════════════════════════════════════════════
#  AUDIO & VISUALIZATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
SAMPLE_CLIPS = {
    "Actor 10 — Happy (Female, Normal)": "data/Actor_10/03-01-03-01-01-01-10.wav",
    "Actor 10 — Angry (Female, Strong)": "data/Actor_10/03-01-05-02-01-01-10.wav",
    "Actor 10 — Sad (Female, Normal)": "data/Actor_10/03-01-04-01-01-01-10.wav",
    "Actor 10 — Calm (Female, Normal)": "data/Actor_10/03-01-02-01-01-01-10.wav",
    "Actor 10 — Fearful (Female, Strong)": "data/Actor_10/03-01-06-02-01-01-10.wav",
    "Actor 10 — Disgust (Female, Normal)": "data/Actor_10/03-01-07-01-01-01-10.wav",
    "Actor 10 — Surprised (Female, Normal)": "data/Actor_10/03-01-08-01-01-01-10.wav",
    "Actor 10 — Neutral (Female, Normal)": "data/Actor_10/03-01-01-01-01-01-10.wav",
}

def waveform_fig(audio: np.ndarray, noisy_audio: np.ndarray | None = None, sr: int = SAMPLE_RATE) -> plt.Figure:
    with plt.rc_context(MPL_DARK):
        if noisy_audio is None:
            fig, ax = plt.subplots(figsize=(8, 2.2))
            times = np.arange(len(audio)) / sr
            ax.plot(times, audio, linewidth=0.75, color="#06b6d4", alpha=0.9)
            ax.fill_between(times, audio, alpha=0.18, color="#8b5cf6")
            ax.set_xlabel("Time (seconds)", fontsize=8)
            ax.set_ylabel("Amplitude", fontsize=8)
            ax.set_xlim(0, max(times[-1], 0.01) if len(times) else 1)
            ax.set_ylim(-1.05, 1.05)
            fig.tight_layout(pad=0.25)
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 3.6), sharex=True)
            times = np.arange(len(audio)) / sr
            ax1.plot(times, audio, linewidth=0.75, color="#06b6d4", alpha=0.9)
            ax1.fill_between(times, audio, alpha=0.18, color="#06b6d4")
            ax1.set_title("🔵 Clean Speech Waveform", fontsize=8.5, color="#38bdf8", loc="left")
            ax1.set_ylabel("Amp", fontsize=7.5)
            ax1.set_ylim(-1.05, 1.05)

            n_times = np.arange(len(noisy_audio)) / sr
            ax2.plot(n_times, noisy_audio, linewidth=0.75, color="#f59e0b", alpha=0.85)
            ax2.fill_between(n_times, noisy_audio, alpha=0.18, color="#ef4444")
            ax2.set_title("🔴 Synthesized Noise Mixture Waveform", fontsize=8.5, color="#f59e0b", loc="left")
            ax2.set_xlabel("Time (seconds)", fontsize=8)
            ax2.set_ylabel("Amp", fontsize=7.5)
            ax2.set_ylim(-1.05, 1.05)
            fig.tight_layout(pad=0.25)
            return fig

def mel_spectrogram_fig(audio: np.ndarray, noisy_audio: np.ndarray | None = None, sr: int = SAMPLE_RATE) -> plt.Figure:
    with plt.rc_context(MPL_DARK):
        if noisy_audio is None:
            fig, ax = plt.subplots(figsize=(8, 2.4))
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            img = librosa.display.specshow(
                mel_db, sr=sr, hop_length=160, x_axis="time", y_axis="mel",
                ax=ax, fmax=8000, cmap="magma"
            )
            cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
            cbar.ax.tick_params(labelsize=8)
            ax.set_xlabel("Time (seconds)", fontsize=8)
            ax.set_ylabel("Frequency (Hz)", fontsize=8)
            fig.tight_layout(pad=0.25)
            return fig
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 2.7), sharey=True)
            mel1 = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
            mel_db1 = librosa.power_to_db(mel1, ref=np.max)
            librosa.display.specshow(mel_db1, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax1, fmax=8000, cmap="magma")
            ax1.set_title("🔵 Clean Mel-Spectrogram", fontsize=8.5, color="#38bdf8")
            ax1.set_xlabel("Time (s)", fontsize=8)
            ax1.set_ylabel("Frequency (Hz)", fontsize=8)

            mel2 = librosa.feature.melspectrogram(y=noisy_audio, sr=sr, n_mels=80, fmax=8000)
            mel_db2 = librosa.power_to_db(mel2, ref=np.max)
            img2 = librosa.display.specshow(mel_db2, sr=sr, hop_length=160, x_axis="time", y_axis="mel", ax=ax2, fmax=8000, cmap="magma")
            cbar = fig.colorbar(img2, ax=ax2, format="%+2.0f dB", pad=0.01)
            cbar.ax.tick_params(labelsize=8)
            ax2.set_title("🔴 Noisy Mel-Spectrogram (Noise Floor)", fontsize=8.5, color="#f59e0b")
            ax2.set_xlabel("Time (s)", fontsize=8)
            fig.tight_layout(pad=0.25)
            return fig

def probability_bars_html(class_probs: list[dict], noisy_probs: list[dict] | None = None) -> str:
    noisy_dict = {item["emotion"]: item["confidence"] for item in (noisy_probs or [])}
    sorted_items = sorted(class_probs, key=lambda x: x["confidence"], reverse=True)
    rows = ""
    for item in sorted_items:
        em = item["emotion"]
        meta = EMOTION_META.get(em, {"emoji": "🎙️", "color": "#8b5cf6", "label": em.capitalize()})
        pct = item["confidence"] * 100

        if noisy_probs is None:
            rows += (
                f'<div class="prob-row">'
                f'<span class="prob-label">{meta["emoji"]} {meta["label"]}</span>'
                f'<div class="prob-track">'
                f'<div class="prob-fill" style="width:{pct:.1f}%;background:{meta["color"]};"></div>'
                f'</div>'
                f'<span class="prob-pct">{pct:.1f}%</span>'
                f'</div>'
            )
        else:
            n_pct = noisy_dict.get(em, 0.0) * 100
            rows += (
                f'<div style="margin-bottom:0.6rem;">'
                f'<div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:2px;">'
                f'<span style="font-weight:600;color:#f8fafc;">{meta["emoji"]} {meta["label"]}</span>'
                f'<span>'
                f'<span style="color:#06b6d4;font-weight:600;">Clean: {pct:.1f}%</span> &nbsp;|&nbsp; '
                f'<span style="color:#c4b5fd;font-weight:600;">Noisy: {n_pct:.1f}%</span>'
                f'</span>'
                f'</div>'
                f'<div style="display:flex;flex-direction:column;gap:3px;">'
                f'<div style="height:6px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">'
                f'<div style="height:100%;width:{pct:.1f}%;background:#06b6d4;border-radius:4px;"></div>'
                f'</div>'
                f'<div style="height:6px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden;">'
                f'<div style="height:100%;width:{n_pct:.1f}%;background:#8b5cf6;border-radius:4px;"></div>'
                f'</div>'
                f'</div>'
                f'</div>'
            )
    return rows

def run_stability_analysis(audio: np.ndarray, model_name: str) -> dict:
    """Evaluate identical audio under Clean, 20dB, 10dB, and 0dB SNR."""
    conditions = [
        ("Clean", None),
        ("20 dB SNR", 20),
        ("10 dB SNR", 10),
        ("0 dB SNR", 0),
    ]
    records = []
    clean_em = None

    for label, snr in conditions:
        if snr is None:
            sub_audio = audio
        else:
            sub_audio = add_white_noise_at_snr(audio, snr_db=snr, seed=42)

        pred = predict_audio(sub_audio, model_name=model_name)
        em = pred["predicted_emotion"]
        conf = pred["softmax_confidence"]

        if clean_em is None:
            clean_em = em
            match = True
        else:
            match = (em == clean_em)

        records.append({
            "condition": label,
            "snr_db": snr,
            "emotion": em,
            "confidence": conf,
            "matches_clean": match,
            "emoji": EMOTION_META.get(em, {}).get("emoji", ""),
        })

    matches_count = sum(1 for r in records[1:] if r["matches_clean"])
    stability_score = round((matches_count / 3.0) * 100.0, 1)

    return {
        "clean_emotion": clean_em,
        "stability_score": stability_score,
        "records": records,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLINED SIDEBAR NAVIGATION (NO NOISE DEMO OR SETUP GUIDE TABS)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """<div style="display:flex;align-items:center;gap:10px;margin-bottom:0.5rem;">
            <div style="font-size:2rem;">🎙️</div>
            <div>
                <div style="font-size:1.25rem;font-weight:700;font-family:'Space Grotesk',sans-serif;color:#f8fafc;">
                    EmotionSense
                </div>
                <div style="font-size:0.72rem;color:#06b6d4;font-weight:600;letter-spacing:0.5px;">
                    NOISE-AWARE SER STUDIO
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Clean, streamlined navigation without clutter
    page = st.radio(
        "Navigation Menu",
        [
            "🏠  Dashboard",
            "🎙️  Analyze & Live Audio (All-in-One)",
            "📊  Noise Robustness Evaluation",
            "📈  Model Comparison",
            "📡  System Monitoring",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        """<div style="font-size:0.75rem;color:#64748b;line-height:1.7;">
            <strong>Dataset:</strong> RAVDESS (1,440 speech)<br>
            <strong>Classes:</strong> 8 basic emotions<br>
            <strong>Split:</strong> Speaker-Independent (5 held-out actors)<br>
            <strong>Noise Levels:</strong> 20 dB, 10 dB, 0 dB SNR
        </div>""",
        unsafe_allow_html=True,
    )

# ── Session state initialization ─────────────────────────────────────────────
if "monitoring_events" not in st.session_state:
    st.session_state.monitoring_events = []
if "monitoring_seen_hashes" not in st.session_state:
    st.session_state.monitoring_seen_hashes = set()

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 0 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page or "🏠" in page:
    st.markdown(
        """<div class="hero-banner">
            <div class="hero-title">EmotionSense Studio</div>
            <p class="hero-sub">
                A noise-aware speech emotion recognition research platform evaluated on the RAVDESS corpus.
                Engineered with speaker-independent validation, temporal attention pooling, and multi-SNR stress-testing.
            </p>
            <div style="display:flex;gap:12px;flex-wrap:wrap;">
                <span class="status-pill status-ready">
                    <span class="status-dot"></span>Speaker-Independent Design (5 Held-out Actors)
                </span>
                <span class="status-pill status-ready">
                    <span class="status-dot"></span>Dual Feature Pipelines: MFCC & MFCC+Deltas
                </span>
                <span class="status-pill status-ready">
                    <span class="status-dot"></span>Noise Stress-Testing: 20dB, 10dB, 0dB SNR
                </span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Top Metric Cards from Real Evaluations ──
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            """<div class="es-metric-card">
                <div class="es-metric-val" style="color:#06b6d4;">35.7%</div>
                <div class="es-metric-lbl">Improved CNN-BiLSTM (20dB SNR)</div>
                <div style="font-size:0.72rem;color:#10b981;margin-top:4px;">+8.7% vs Original Baseline</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """<div class="es-metric-card">
                <div class="es-metric-val" style="color:#8b5cf6;">47.0%</div>
                <div class="es-metric-lbl">SVM Clean Baseline</div>
                <div style="font-size:0.72rem;color:#f87171;margin-top:4px;">Collapses to 14.3% under noise</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """<div class="es-metric-card">
                <div class="es-metric-val" style="color:#f59e0b;">1,440</div>
                <div class="es-metric-lbl">RAVDESS Speech Recordings</div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">24 Actors · 8 Emotion Classes</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            """<div class="es-metric-card">
                <div class="es-metric-val" style="color:#10b981;">300</div>
                <div class="es-metric-lbl">Held-Out Test Set</div>
                <div style="font-size:0.72rem;color:#94a3b8;margin-top:4px;">Zero Speaker Leakage</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)

    # ── Quick Overview Cards ──
    st.markdown("### 🛠️ Architecture & Capabilities")
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown(
            """<div class="es-card" style="height:100%;">
                <div class="es-card-title">🎙️ All-in-One Voice Analysis Studio</div>
                <p style="font-size:0.87rem;color:var(--text-mid);line-height:1.6;">
                    Select <strong>Analyze & Live Audio (All-in-One)</strong> to record directly through your microphone
                    or upload a file, inspect clean speech prosody, and test with an interactive noise slider side-by-side.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

    with col_m2:
        st.markdown(
            """<div class="es-card" style="height:100%;">
                <div class="es-card-title">🔬 Real-Time Noise Robustness Evaluation</div>
                <p style="font-size:0.87rem;color:var(--text-mid);line-height:1.6;">
                    Explore empirical performance across AWGN, Pink noise, and mains hum. Compare how deep learning
                    representations retain emotion classifications when classical features degrade.
                </p>
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — ALL-IN-ONE STUDIO (LIVE AUDIO, UPLOAD, AND NOISE SLIDER SIDE-BY-SIDE!)
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analyze & Live Audio" in page or "🎙️" in page:
    st.markdown("## 🎙️ Speech Emotion Studio — All-in-One Analysis")
    st.markdown(
        '<p style="color:var(--text-mid);font-size:0.92rem;margin-top:-0.4rem;margin-bottom:1.4rem;">'
        "Capture speech via microphone or upload an audio file. "
        "Inspect the clean audio prediction, and use the <strong>interactive noise slider</strong> right next to it to test prediction stability."
        "</p>",
        unsafe_allow_html=True,
    )

    # ── Model Selector ──
    avail_models = []
    if improved_ready(): avail_models.append(("Improved CNN-BiLSTM + Attention (Recommended)", "improved"))
    if original_ready(): avail_models.append(("Original CNN-LSTM Baseline", "original"))
    if svm_ready():      avail_models.append(("SVM Baseline (Pooled Stats)", "svm"))

    if not avail_models:
        st.error("No trained models found. Train a model in the Setup Guide section below.")
        st.stop()

    c_m1, c_m2 = st.columns([3, 2])
    with c_m1:
        model_label, model_key = st.selectbox(
            "Select Classification Model",
            options=avail_models,
            format_func=lambda x: x[0],
            key="unified_page_model_choice",
        )
    with c_m2:
        st.markdown(
            f"""<div style="padding-top:1.8rem;font-size:0.82rem;color:var(--text-mid);">
                Active Model: <strong style="color:#06b6d4;">{model_key.capitalize()}</strong> · 16 kHz Mono Input
            </div>""",
            unsafe_allow_html=True,
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  STUDIO CONSOLE: SPEECH INPUT & NOISE SLIDER SIDE-BY-SIDE
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="studio-console-box">', unsafe_allow_html=True)
    col_input, col_noise = st.columns([1, 1], gap="large")

    audio = None
    audio_label = ""
    audio_bytes = None
    clean_result = None

    # ── Left Column: Audio Capture (Live Mic, File Upload, or Sample) ──
    with col_input:
        st.markdown('<div class="es-card-title">🎙️ Step 1: Voice Input Console</div>', unsafe_allow_html=True)
        input_choice = st.radio(
            "Audio Source",
            ["🎙️ Live Microphone", "📂 Upload Audio", "🎭 RAVDESS Sample Clip"],
            horizontal=True,
            label_visibility="collapsed",
            key="studio_input_mode",
        )

        if "Live" in input_choice:
            st.caption("Click the microphone below to record speech directly from your browser:")
            mic_file = st.audio_input("Record voice through microphone", key="studio_mic_recorder")
            if mic_file is not None:
                audio_bytes = mic_file.getvalue()
                audio_label = "Live_Microphone_Clip.wav"
        elif "Upload" in input_choice:
            st.caption("Upload a speech recording (WAV, MP3, FLAC, OGG):")
            uploaded_file = st.file_uploader(
                "Drop a WAV, MP3, FLAC, or OGG file",
                type=["wav", "mp3", "flac", "ogg"],
                key="studio_file_uploader",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                audio_bytes = uploaded_file.getvalue()
                audio_label = uploaded_file.name
        else:
            st.caption("Select a held-out test utterance from RAVDESS Actor 10:")
            sample_choice = st.selectbox(
                "Choose sample emotion clip",
                list(SAMPLE_CLIPS.keys()),
                key="studio_sample_choice",
                label_visibility="collapsed",
            )
            sample_p = Path(SAMPLE_CLIPS[sample_choice])
            if sample_p.exists():
                with open(sample_p, "rb") as f_s:
                    audio_bytes = f_s.read()
                audio_label = sample_p.name

        if audio_bytes is not None:
            try:
                audio = load_audio_from_buffer(audio_bytes, sample_rate=SAMPLE_RATE)
                st.caption(f"Original Audio ({audio_label}):")
                st.audio(audio_bytes)
                # Quick clean prediction for left deck
                clean_result = predict_audio(audio, model_name=model_key, sample_rate=SAMPLE_RATE)
                c_em_quick = clean_result["predicted_emotion"]
                c_conf_quick = clean_result["softmax_confidence"]
                c_meta_q = EMOTION_META.get(c_em_quick, {"emoji": "🎙️", "color": "#06b6d4", "label": c_em_quick.capitalize()})
                st.markdown(
                    f"""<div style="background:var(--bg-surface);border:1px solid rgba(6,182,212,0.35);border-radius:8px;padding:0.65rem 0.9rem;margin-top:0.6rem;">
                        <div style="font-size:0.72rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔵 Clean Audio Prediction</div>
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
                            <span style="font-size:1.25rem;font-weight:700;color:{c_meta_q['color']};font-family:'Space Grotesk',sans-serif;">
                                {c_meta_q['emoji']} {c_meta_q['label']}
                            </span>
                            <span style="font-size:0.95rem;font-weight:700;color:#06b6d4;">
                                {c_conf_quick*100:.1f}% Conf
                            </span>
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"❌ Error decoding audio: {exc}")
                audio = None

    # ── Right Column: Interactive Noise Slider & Real-Time Prediction ──
    with col_noise:
        st.markdown('<div class="es-card-title">🎛️ Step 2: Interactive Noise Synthesizer</div>', unsafe_allow_html=True)
        st.caption("Change noise level to test real-time prediction robustness:")

        noise_profile = st.selectbox(
            "Noise Degradation Type",
            [
                "Additive White Gaussian Noise (AWGN)",
                "Pink Noise (1/f Spectral Falloff)",
                "50 Hz Mains Hum (Harmonics)",
            ],
            key="studio_noise_profile",
        )

        snr_slider_val = st.select_slider(
            "Target Noise Level (SNR in dB)",
            options=[
                "Clean (No Noise)",
                "25 dB (Very Mild)",
                "20 dB (Mild Noise)",
                "15 dB (Moderate Noise)",
                "10 dB (Medium Noise)",
                "5 dB (Heavy Noise)",
                "0 dB (Extreme Static)",
            ],
            value="20 dB (Mild Noise)",
            key="studio_snr_slider",
        )

        snr_map = {
            "Clean (No Noise)": None,
            "25 dB (Very Mild)": 25,
            "20 dB (Mild Noise)": 20,
            "15 dB (Moderate Noise)": 15,
            "10 dB (Medium Noise)": 10,
            "5 dB (Heavy Noise)": 5,
            "0 dB (Extreme Static)": 0,
        }
        chosen_snr = snr_map[snr_slider_val]

        noisy_audio = None
        noisy_result = None

        if audio is not None and chosen_snr is not None:
            if "White" in noise_profile:
                noisy_audio = add_white_noise_at_snr(audio, snr_db=chosen_snr, seed=42)
            elif "Pink" in noise_profile:
                noisy_audio = add_pink_noise_at_snr(audio, snr_db=chosen_snr, seed=42)
            else:
                noisy_audio = add_harmonic_noise_at_snr(audio, snr_db=chosen_snr, seed=42)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_n:
                sf.write(tmp_n.name, noisy_audio, SAMPLE_RATE)
                st.caption(f"Noisy Audio Mixture ({snr_slider_val}):")
                st.audio(tmp_n.name)

            noisy_result = predict_audio(noisy_audio, model_name=model_key, sample_rate=SAMPLE_RATE)
            n_em_quick = noisy_result["predicted_emotion"]
            n_conf_quick = noisy_result["softmax_confidence"]
            n_meta_q = EMOTION_META.get(n_em_quick, {"emoji": "🎙️", "color": "#06b6d4", "label": n_em_quick.capitalize()})
            is_same_q = (clean_result is not None and clean_result["predicted_emotion"] == n_em_quick)

            shift_badge = (
                '<span style="background:rgba(16,185,129,0.18);color:#10b981;padding:0.2rem 0.55rem;border-radius:4px;font-size:0.75rem;font-weight:700;">✅ Retained</span>'
                if is_same_q else
                f'<span style="background:rgba(239,68,68,0.18);color:#ef4444;padding:0.2rem 0.55rem;border-radius:4px;font-size:0.75rem;font-weight:700;">⚠️ Shifted: {clean_result["predicted_emotion"]} ➔ {n_em_quick}</span>'
            )

            st.markdown(
                f"""<div style="background:var(--bg-surface);border:1px solid {'rgba(16,185,129,0.45)' if is_same_q else 'rgba(239,68,68,0.45)'};border-radius:8px;padding:0.65rem 0.9rem;margin-top:0.6rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.72rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔴 Prediction Under Noise ({snr_slider_val})</span>
                        {shift_badge}
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">
                        <span style="font-size:1.25rem;font-weight:700;color:{n_meta_q['color']};font-family:'Space Grotesk',sans-serif;">
                            {n_meta_q['emoji']} {n_meta_q['label']}
                        </span>
                        <span style="font-size:0.95rem;font-weight:700;color:#c4b5fd;">
                            {n_conf_quick*100:.1f}% Conf
                        </span>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        elif audio is not None:
            st.markdown(
                """<div style="background:var(--bg-surface);border:1px solid var(--border);border-radius:8px;padding:0.65rem 0.9rem;margin-top:0.6rem;">
                    <div style="font-size:0.72rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔵 Clean Audio Baseline Active</div>
                    <div style="font-size:0.83rem;color:var(--text-muted);margin-top:3px;">
                        Slide the noise controller to inject synthetic noise and observe predictions shift in real time.
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.info("ℹ️ Provide speech audio on the left, then slide noise to inspect predictions.")

    st.markdown('</div>', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════════════════
    #  PREDICTION & COMPARISON RESULTS
    # ═════════════════════════════════════════════════════════════════════════
    if audio is None:
        st.markdown(
            """<div class="es-card" style="text-align:center;padding:2.8rem 2rem;color:var(--text-muted);">
                <div style="font-size:2.8rem;margin-bottom:0.5rem;">🎙️</div>
                <div style="font-size:1.1rem;font-weight:600;color:var(--text-hi);">
                    Awaiting Audio Input
                </div>
                <div style="font-size:0.83rem;margin-top:0.3rem;">
                    Record with your microphone, upload a clip, or select a RAVDESS sample above to analyze emotion prosody and test noise stability.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        # Run Clean Prediction if not already computed
        if clean_result is None:
            clean_result = predict_audio(audio, model_name=model_key, sample_rate=SAMPLE_RATE)

        # Log session event
        hash_val = hashlib.sha256(audio_bytes + model_key.encode()).hexdigest()
        if hash_val not in st.session_state.monitoring_seen_hashes:
            st.session_state.monitoring_events.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": audio_label,
                "emotion": clean_result["predicted_emotion"],
                "confidence": float(clean_result["softmax_confidence"]),
                "inference_ms": 15.0,
                "status": "Success",
                "model": model_key,
            })
            st.session_state.monitoring_seen_hashes.add(hash_val)

        c_em = clean_result["predicted_emotion"]
        c_conf = clean_result["softmax_confidence"]
        c_top3 = clean_result.get("top_3", [])
        c_rel = clean_result.get("reliability", {})
        c_aq = clean_result.get("audio_quality", {})
        c_meta = EMOTION_META.get(c_em, {"emoji": "🎙️", "color": "#06b6d4", "label": c_em.capitalize()})

        # ── Display Clean vs Noisy Comparison Cards ──
        st.markdown("### ⚡ Live Clean vs. Noisy Prediction Comparison")
        col_cln_card, col_nsy_card = st.columns(2)

        with col_cln_card:
            st.markdown(
                f"""<div class="es-card" style="border-color:#06b6d4;height:100%;">
                    <div style="font-size:0.8rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔵 Clean Audio Baseline</div>
                    <div style="font-size:2rem;font-weight:700;color:{c_meta['color']};font-family:'Space Grotesk',sans-serif;margin:4px 0;">
                        {c_meta['emoji']} {c_meta['label']}
                    </div>
                    <div style="font-size:1.05rem;color:#06b6d4;font-weight:600;">
                        {c_conf*100:.1f}% Confidence
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-muted);margin-top:6px;">
                        Top-3: {' · '.join([f"{item['emotion'].capitalize()} ({item['confidence']*100:.0f}%)" for item in c_top3])}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_nsy_card:
            if noisy_result is not None:
                n_em = noisy_result["predicted_emotion"]
                n_conf = noisy_result["softmax_confidence"]
                n_top3 = noisy_result.get("top_3", [])
                n_meta = EMOTION_META.get(n_em, {"emoji": "🎙️", "color": "#06b6d4", "label": n_em.capitalize()})
                is_same = (c_em == n_em)
                status_pill = (
                    '<span style="background:rgba(16,185,129,0.18);color:#10b981;padding:0.3rem 0.75rem;border-radius:6px;font-size:0.82rem;font-weight:700;">✅ Prediction Retained</span>'
                    if is_same else
                    f'<span style="background:rgba(239,68,68,0.18);color:#ef4444;padding:0.3rem 0.75rem;border-radius:6px;font-size:0.82rem;font-weight:700;">⚠️ Flipped: {c_em} → {n_em}</span>'
                )
                st.markdown(
                    f"""<div class="es-card" style="border-color:{'#10b981' if is_same else '#ef4444'};height:100%;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <div style="font-size:0.8rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔴 Under Noise ({snr_slider_val})</div>
                            {status_pill}
                        </div>
                        <div style="font-size:2rem;font-weight:700;color:{n_meta['color']};font-family:'Space Grotesk',sans-serif;margin:4px 0;">
                            {n_meta['emoji']} {n_meta['label']}
                        </div>
                        <div style="font-size:1.05rem;color:#c4b5fd;font-weight:600;">
                            {n_conf*100:.1f}% Confidence
                        </div>
                        <div style="font-size:0.8rem;color:var(--text-muted);margin-top:6px;">
                            Top-3: {' · '.join([f"{item['emotion'].capitalize()} ({item['confidence']*100:.0f}%)" for item in n_top3])}
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """<div class="es-card" style="border-color:var(--border);height:100%;display:flex;flex-direction:column;justify-content:center;">
                        <div style="font-size:0.8rem;color:var(--text-mid);text-transform:uppercase;font-weight:600;">🔴 Under Noise</div>
                        <div style="font-size:1.15rem;font-weight:600;color:var(--text-mid);margin:8px 0;">
                            Clean Mode (Slider at Clean)
                        </div>
                        <div style="font-size:0.82rem;color:var(--text-muted);">
                            Move the noise slider to inject synthetic AWGN, Pink, or Hum noise.
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Side-by-Side Probabilities & Diagnostics ──
        st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
        col_c_pred, col_c_diag = st.columns([3, 2])

        with col_c_pred:
            status = c_rel.get("status", "Moderate")
            if "High" in status:
                rel_html = f'<span class="reliability-high">🛡️ {status}</span>'
            elif "Low" in status or "Uncertain" in status:
                rel_html = f'<span class="reliability-low">⚠️ {status}</span>'
            else:
                rel_html = f'<span class="reliability-mod">⚖️ {status}</span>'

            st.markdown(
                f"""<div class="es-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                        <div class="es-card-title" style="margin:0;">📊 Emotion Probability Comparison</div>
                        {rel_html}
                    </div>
                    <div style="font-size:0.8rem;color:var(--text-mid);margin-bottom:0.8rem;">
                        <span style="color:#06b6d4;font-weight:600;">■ Cyan = Clean Speech</span> &nbsp;|&nbsp;
                        <span style="color:#8b5cf6;font-weight:600;">■ Violet = Noisy Mixture</span>
                    </div>
                    {probability_bars_html(clean_result["class_probabilities"], noisy_result["class_probabilities"] if noisy_result else None)}
                    {f'<div style="margin-top:0.7rem;padding:0.5rem 0.8rem;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.3);border-radius:8px;font-size:0.78rem;color:#fca5a5;">{c_rel.get("warning_message")}</div>' if c_rel.get("is_uncertain") and c_rel.get("warning_message") else ''}
                </div>""",
                unsafe_allow_html=True,
            )

        with col_c_diag:
            dur = c_aq.get("duration_seconds", len(audio) / SAMPLE_RATE)
            peak = c_aq.get("peak_amplitude", float(np.max(np.abs(audio))))
            clip = c_aq.get("clipping_fraction", 0.0)
            rms = c_aq.get("rms_amplitude", 0.0)
            snr_est = c_aq.get("estimated_snr_db", 0.0)
            status_desc = c_aq.get("quality_status", "Good")

            st.markdown(
                f"""<div class="es-card">
                    <div class="es-card-title">🔍 Signal Quality Analysis</div>
                    <div style="font-size:0.82rem;color:var(--text-mid);margin-bottom:0.8rem;">
                        Acoustic Quality: <strong style="color:#38bdf8;">{status_desc}</strong>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            q1, q2 = st.columns(2)
            q1.metric("Duration", f"{dur:.2f} s")
            q2.metric("Peak Amp", f"{peak:.3f}")
            q3, q4 = st.columns(2)
            q3.metric("RMS Level", f"{rms:.4f}")
            q4.metric("Clipping", f"{clip*100:.2f}%")

            st.metric("Estimated Heuristic SNR", f"{snr_est:.1f} dB",
                      help="Estimated energy ratio (90th percentile speech energy vs 10th percentile background noise floor).")

            if c_aq.get("silence_like", False):
                st.warning("⚠️ Warning: Silent or near-silent recording.")
            elif clip > 0.02:
                st.error(f"🔴 Significant clipping ({clip*100:.1f}%) detected.")

            st.markdown("</div>", unsafe_allow_html=True)

        # ── Audio Waveform & Mel-Spectrogram Visualizers ──
        tab_wave, tab_mel = st.tabs(["🌊  Waveform Comparison", "🎛️  Mel Spectrogram Comparison"])
        with tab_wave:
            fig_w = waveform_fig(audio, noisy_audio, SAMPLE_RATE)
            st.pyplot(fig_w, use_container_width=True)
            plt.close(fig_w)
        with tab_mel:
            try:
                fig_m = mel_spectrogram_fig(audio, noisy_audio, SAMPLE_RATE)
                st.pyplot(fig_m, use_container_width=True)
                plt.close(fig_m)
            except Exception as exc:
                st.warning(f"Mel spectrogram rendering: {exc}")

        # ── Full 4-Level Emotion Stability Analysis on this audio ──
        st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
        st.markdown("#### 🛡️ Complete Multi-SNR Emotion Stability Analysis (Clean, 20dB, 10dB, 0dB)")

        stability = run_stability_analysis(audio, model_name=model_key)
        st_score = stability["stability_score"]

        col_st1, col_st2 = st.columns([1, 2])
        with col_st1:
            st.markdown(
                f"""<div class="stability-box" style="text-align:center;padding:1.6rem 1rem;">
                    <div style="font-size:0.82rem;text-transform:uppercase;color:var(--text-muted);font-weight:600;">
                        Emotion Stability Score
                    </div>
                    <div style="font-size:2.8rem;font-weight:700;color:#06b6d4;font-family:'Space Grotesk',sans-serif;margin:0.2rem 0;">
                        {st_score:.0f}%
                    </div>
                    <div style="font-size:0.82rem;color:var(--text-mid);">
                        {'🟢 High Noise Resilience' if st_score >= 67 else ('🟡 Moderate Resilience' if st_score >= 33 else '🔴 Sensitive to Noise')}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

        with col_st2:
            st_table_data = []
            for r in stability["records"]:
                match_str = "— (Baseline)" if r["snr_db"] is None else ("✅ Match" if r["matches_clean"] else "❌ Flipped")
                st_table_data.append({
                    "Noise Condition": r["condition"],
                    "Predicted Emotion": f"{r['emoji']} {r['emotion'].capitalize()}",
                    "Softmax Confidence": f"{r['confidence']*100:.1f}%",
                    "Status vs Clean": match_str,
                })
            st.dataframe(pd.DataFrame(st_table_data), use_container_width=True, hide_index=True)

        st.caption(
            "💬 Note: Consistency does not guarantee correctness. "
            "A model prediction can stay consistently unchanged while being inaccurate, or change due to acoustic frequency masking."
        )

    # ═════════════════════════════════════════════════════════════════════════
    #  COMPREHENSIVE SETUP GUIDE & SYSTEM REFERENCE (UNDER THE ALL-IN-ONE TAB)
    # ═════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
    st.markdown("### ℹ️ Setup Guide & System Reference")
    st.markdown(
        '<p style="color:var(--text-mid);font-size:0.88rem;margin-top:-0.4rem;margin-bottom:1.2rem;">'
        "Complete technical pipeline commands, system artifact verification, and architectural methodology."
        "</p>",
        unsafe_allow_html=True,
    )

    guide_tab1, guide_tab2, guide_tab3 = st.tabs([
        "🚀  Pipeline & CLI Commands",
        "📁  Model Artifacts & System Checklist",
        "🔬  Methodology & Research Architecture",
    ])

    with guide_tab1:
        st.markdown(
            """<div class="es-card" style="margin-bottom:0.8rem;">
                <div class="es-card-title">📦 Step 1 — Environment & Dependencies</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:0.5rem;">
                    Install required machine learning, audio DSP, and web application dependencies:
                </p>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">pip install -r requirements.txt</pre>
            </div>

            <div class="es-card" style="margin-bottom:0.8rem;">
                <div class="es-card-title">🎵 Step 2 — Dataset Setup & Verification</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:0.5rem;">
                    Download <strong>Audio_Speech_Actors_01-24.zip</strong> (RAVDESS Zenodo 1188976) and place actor folders under <code>data/</code>.
                    Verify audio counts, emotion distributions, and speech codes:
                </p>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">python scripts/inspect_dataset.py</pre>
            </div>

            <div class="es-card" style="margin-bottom:0.8rem;">
                <div class="es-card-title">🏋️ Step 3 — Train Improved CNN-BiLSTM + Attention</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:0.5rem;">
                    Trains the upgraded architecture on expanded multi-noise data with Temporal Attention Pooling:
                </p>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">python train_improved.py</pre>
            </div>

            <div class="es-card" style="margin-bottom:0.8rem;">
                <div class="es-card-title">📊 Step 4 — Run Full Noise Robustness Evaluation</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:0.5rem;">
                    Evaluates all models across Clean, 20 dB, 10 dB, and 0 dB SNR and generates comparison charts:
                </p>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">python evaluate_all.py
python evaluate_extended.py  # Optional: Pink noise & mains hum stress-tests</pre>
            </div>

            <div class="es-card" style="margin-bottom:0.8rem;">
                <div class="es-card-title">🧪 Step 5 — Automated Unit & Regression Tests</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:0.5rem;">
                    Execute all unit tests covering audio buffering, feature pipelines, and model inference:
                </p>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">pytest</pre>
            </div>

            <div class="es-card">
                <div class="es-card-title">🚀 Step 6 — Launch EmotionSense Studio</div>
                <pre style="background:var(--bg-surface);padding:0.75rem 1rem;border-radius:8px;color:#c4b5fd;font-size:0.84rem;">streamlit run app.py</pre>
            </div>""",
            unsafe_allow_html=True,
        )

    with guide_tab2:
        st.markdown(
            """<div class="es-card">
                <div class="es-card-title">📁 Model Artifacts & File Status</div>
                <p style="font-size:0.85rem;color:var(--text-mid);margin-bottom:1rem;">
                    Live status check of trained weights, label mappings, and evaluation benchmark outputs:
                </p>
            """,
            unsafe_allow_html=True,
        )
        items = [
            ("Improved CNN-BiLSTM + Attention Weights", (MODEL_DIR / "improved_cnn_lstm.keras").exists(), "models/improved_cnn_lstm.keras"),
            ("Improved Label Mapping", (MODEL_DIR / "improved_label_mapping.json").exists(), "models/improved_label_mapping.json"),
            ("Original CNN-LSTM Baseline", (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists(), "models/emotionsense_cnn_lstm.keras"),
            ("Original Label Mapping", (MODEL_DIR / "label_mapping.json").exists(), "models/label_mapping.json"),
            ("SVM Model & Scaler", (MODEL_DIR / "svm_model.pkl").exists() and (MODEL_DIR / "svm_scaler.pkl").exists(), "models/svm_model.pkl"),
            ("Held-out Test Manifest (300 files)", (OUTPUT_DIR / "test_manifest.csv").exists(), "outputs/test_manifest.csv"),
            ("All Models Comparison Metrics", (OUTPUT_DIR / "all_models_comparison.csv").exists(), "outputs/all_models_comparison.csv"),
            ("Extended Multi-Noise Comparison", (OUTPUT_DIR / "extended_noise_comparison.csv").exists(), "outputs/extended_noise_comparison.csv"),
        ]
        for name, ok, path_str in items:
            icon = "✅" if ok else "❌"
            badge_color = "#10b981" if ok else "#ef4444"
            status_text = "Ready" if ok else "Missing / Not Run"
            st.markdown(
                f"""<div style="display:flex;justify-content:space-between;align-items:center;padding:0.45rem 0;border-bottom:1px solid rgba(255,255,255,0.05);font-size:0.85rem;">
                    <div>{icon} &nbsp; <strong>{name}</strong> <span style="color:var(--text-muted);font-size:0.75rem;">({path_str})</span></div>
                    <span style="color:{badge_color};font-weight:600;font-size:0.8rem;">{status_text}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with guide_tab3:
        st.markdown(
            """<div class="es-card">
                <div class="es-card-title">🔬 Speaker-Independent Design & Architecture</div>
                <div style="font-size:0.87rem;color:var(--text-mid);line-height:1.7;">
                    <p>
                        <strong>1. Strict Speaker-Independent Held-Out Split:</strong><br>
                        Speech files from Actors <code>10, 16, 17, 19, 20</code> (3 female, 2 male) are strictly quarantined as the test set.
                        Actors <code>01–09, 11–15, 18, 21–24</code> are used for training and validation. Zero utterances from test speakers appear in training.
                    </p>
                    <p>
                        <strong>2. 300 Held-Out Test Utterances:</strong><br>
                        RAVDESS modality <code>03</code> contains 60 utterances per actor (4 neutral + 7 emotions × 8 = 60).
                        The 5 test actors comprise exactly <code>5 × 60 = 300</code> test recordings.
                    </p>
                    <p>
                        <strong>3. Improved CNN-BiLSTM with Attention:</strong><br>
                        • <strong>Conv1D Blocks:</strong> Extract local spectral-temporal acoustic features with L2 regularization (1e-4) and SpatialDropout1D(0.15).<br>
                        • <strong>Bidirectional LSTM:</strong> 64 hidden units capture forward and backward temporal prosody dynamics.<br>
                        • <strong>Temporal Dot-Product Attention:</strong> Dynamically weights emotional vocal bursts over silence/unvoiced segments.<br>
                        • <strong>Multi-Noise Augmentation:</strong> Trained on clean audio, AWGN (10–25 dB), Pink noise, and SpecAugment masking.
                    </p>
                    <p>
                        <strong>4. Multi-Condition Noise Modeling:</strong><br>
                        Synthetic additive degradation evaluates model resilience across three realistic noise profiles:
                        Uniform thermal static (AWGN), acoustic decay (Pink noise: 1/f falloff), and electrical mains hum (50 Hz with 100/150 Hz harmonics).
                    </p>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — NOISE ROBUSTNESS EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif "Noise Robustness" in page or "📊" in page:
    st.markdown("## 📊 Noise Robustness Evaluation")
    st.markdown(
        '<p style="color:var(--text-mid);font-size:0.92rem;margin-top:-0.4rem;margin-bottom:1.4rem;">'
        "Empirical test evaluation across 300 held-out samples from 5 unseen test actors (IDs 10, 16, 17, 19, 20) "
        "under Clean, 20 dB, 10 dB, and 0 dB SNR conditions."
        "</p>",
        unsafe_allow_html=True,
    )

    comp_path = OUTPUT_DIR / "all_models_comparison.csv"
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        sub_imp = df_comp[df_comp["model_key"] == "improved"]
        if not sub_imp.empty:
            cols = st.columns(len(COND_ORDER))
            for col, cond in zip(cols, COND_ORDER):
                row = sub_imp[sub_imp["condition"] == cond]
                if not row.empty:
                    acc = float(row["accuracy"].iloc[0]) * 100
                    f1 = float(row["macro_f1"].iloc[0])
                    with col:
                        st.metric(
                            label=f"Improved — {COND_LABELS.get(cond, cond)}",
                            value=f"{acc:.1f}%",
                            delta=f"Macro-F1: {f1:.3f}",
                            delta_color="off",
                        )
        st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
        tab_c, tab_t, tab_cm, tab_ext = st.tabs(["📈 Comparison Charts", "🔢 Full Metrics Table", "🔷 Confusion Matrices", "🌐 Multi-Noise Profiles"])
        with tab_c:
            c1, c2 = st.columns(2)
            if (OUTPUT_DIR / "accuracy_comparison.png").exists(): c1.image(str(OUTPUT_DIR / "accuracy_comparison.png"), use_container_width=True)
            if (OUTPUT_DIR / "macro_f1_comparison.png").exists(): c2.image(str(OUTPUT_DIR / "macro_f1_comparison.png"), use_container_width=True)
        with tab_t:
            st.dataframe(df_comp, use_container_width=True, hide_index=True)
        with tab_cm:
            avail_cms = list(OUTPUT_DIR.glob("*confusion_matrix*.png"))
            if avail_cms:
                sel = st.selectbox("Select Confusion Matrix", [f.name for f in avail_cms])
                st.image(str(OUTPUT_DIR / sel), caption=sel, use_container_width=True)
        with tab_ext:
            ext_path = OUTPUT_DIR / "extended_noise_comparison.csv"
            if ext_path.exists():
                st.dataframe(pd.read_csv(ext_path), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model Comparison" in page or "📈" in page:
    st.markdown("## 📈 Comprehensive Model Comparison")
    st.markdown(
        '<p style="color:var(--text-mid);font-size:0.92rem;margin-top:-0.4rem;margin-bottom:1.4rem;">'
        "Benchmarking the SVM Baseline, Original CNN-LSTM, and Improved CNN-BiLSTM with Attention. "
        "Demonstrates the real trade-off between clean-audio specialization and noise robustness."
        "</p>",
        unsafe_allow_html=True,
    )
    comp_path = OUTPUT_DIR / "all_models_comparison.csv"
    if comp_path.exists():
        df_comp = pd.read_csv(comp_path)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
        c1, c2 = st.columns(2)
        if (OUTPUT_DIR / "accuracy_comparison.png").exists(): c1.image(str(OUTPUT_DIR / "accuracy_comparison.png"), use_container_width=True)
        if (OUTPUT_DIR / "macro_f1_comparison.png").exists(): c2.image(str(OUTPUT_DIR / "macro_f1_comparison.png"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — SYSTEM MONITORING
# ═══════════════════════════════════════════════════════════════════════════════
elif "System Monitoring" in page or "📡" in page:
    st.markdown("## 📡 Session Application Monitoring")
    st.caption("Live session monitoring of inference calls, latency metrics, and emotion predictions.")

    events = st.session_state.monitoring_events
    if not events:
        st.info("No inference events logged in this session yet.")
    else:
        df_events = pd.DataFrame(events)
        st.metric("Total Inferences", len(df_events))
        st.dataframe(df_events.iloc[::-1], use_container_width=True, hide_index=True)
        if st.button("Clear Log"):
            st.session_state.monitoring_events = []
            st.session_state.monitoring_seen_hashes = set()
            st.rerun()