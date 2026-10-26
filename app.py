"""EmotionSense Studio — Streamlit dashboard.

Run with:   streamlit run app.py
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # must be before pyplot import; prevents GUI thread issues in Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# ── project imports ──────────────────────────────────────────────────────────
from config import MODEL_DIR, OUTPUT_DIR, SAMPLE_RATE
from predict import predict_file

# ── page config must be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="EmotionSense Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS INJECTION
# ═══════════════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg-deep:    #0d1117;
    --bg-card:    #161b27;
    --bg-surface: #1e2535;
    --accent:     #7c6af7;
    --accent2:    #5ba3f5;
    --accent-glow:rgba(124,106,247,0.25);
    --success:    #22c55e;
    --warning:    #f59e0b;
    --danger:     #ef4444;
    --text-hi:    #f0f4ff;
    --text-mid:   #9ba6c2;
    --text-lo:    #4a5568;
    --border:     rgba(124,106,247,0.18);
    --radius-lg:  14px;
    --radius-md:  10px;
    --shadow-card:0 4px 32px rgba(0,0,0,0.45);
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
h1 { font-size: 2.4rem !important; letter-spacing: -0.5px; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.1rem !important; color: var(--text-mid) !important; font-weight: 500; }

.hero-banner {
    background: linear-gradient(135deg, #161b27 0%, #1a1040 50%, #0f1a2e 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem;
    box-shadow: var(--shadow-card), 0 0 60px var(--accent-glow);
    position: relative; overflow: hidden;
}
.hero-banner::before {
    content: ''; position: absolute; top: -60px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(124,106,247,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem; font-weight: 700;
    background: linear-gradient(90deg, #a89cff 0%, #7c6af7 40%, #5ba3f5 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin: 0 0 0.5rem;
}
.hero-sub { color: var(--text-mid); font-size: 1rem; font-weight: 400; line-height: 1.6; margin: 0; }
.status-pill {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.4px; margin-top: 1.2rem;
}
.status-ready   { background: rgba(34,197,94,0.12);  color: var(--success);  border: 1px solid rgba(34,197,94,0.3);  }
.status-missing { background: rgba(239,68,68,0.10);  color: var(--danger);   border: 1px solid rgba(239,68,68,0.28); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.status-dot-pulse { animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

.es-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-card);
}
.es-card-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.75rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: var(--accent); margin-bottom: 0.9rem;
}

[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stMetricLabel"] { color: var(--text-mid) !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"] { color: var(--text-hi) !important; font-size: 1.4rem !important; font-weight: 700 !important; }

[data-testid="stFileUploader"] {
    background: var(--bg-surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-lg) !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] { color: var(--text-mid) !important; font-weight: 500; }
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}

.es-sep { border-top: 1px solid var(--border); margin: 1.2rem 0; }

.prob-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 0.85rem; }
.prob-label  { width: 90px; color: var(--text-mid); text-align: right; text-transform: capitalize; }
.prob-track  { flex: 1; height: 8px; background: var(--bg-deep); border-radius: 4px; overflow: hidden; }
.prob-fill   { height: 100%; border-radius: 4px;
               background: linear-gradient(90deg, var(--accent), var(--accent2)); }
.prob-pct    { width: 44px; color: var(--text-hi); font-weight: 600; font-size: 0.82rem; }

.seg-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; font-size: 0.83rem; }
.seg-time  { width: 80px; color: var(--text-mid); font-variant-numeric: tabular-nums; }
.seg-label { padding: 3px 10px; border-radius: 4px;
             background: var(--bg-surface); color: var(--text-hi);
             font-weight: 600; border: 1px solid var(--border); }
.seg-conf  { color: var(--text-mid); font-size: 0.78rem; }

.setup-box {
    background: linear-gradient(135deg, rgba(124,106,247,0.07), rgba(91,163,245,0.07));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 1.8rem 2rem;
}
.setup-box h3 { color: var(--accent) !important; margin-top: 1.2rem; }
.setup-box h3:first-child { margin-top: 0; }
.setup-box code { background: var(--bg-surface); padding: 2px 6px; border-radius: 4px;
                  font-size: 0.85rem; color: var(--accent2); }
.setup-box pre  { background: var(--bg-surface) !important; border: 1px solid var(--border);
                  border-radius: 8px; padding: 0.9rem; overflow-x: auto; font-size: 0.82rem; }

.sidebar-logo {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(90deg, #a89cff, #5ba3f5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; padding: 1rem 0 0.4rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS / HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

MPL_STYLE = {
    "figure.facecolor": "#161b27", "axes.facecolor": "#1e2535",
    "axes.edgecolor": "#2a3450",   "axes.labelcolor": "#9ba6c2",
    "axes.grid": True,             "grid.color": "#2a3450",
    "grid.linewidth": 0.6,         "xtick.color": "#9ba6c2",
    "ytick.color": "#9ba6c2",      "text.color": "#f0f4ff",
    "font.family": "sans-serif",   "font.size": 10,
}
EMOTION_EMOJIS = {
    "neutral": "😐", "calm": "😌", "happy": "😊", "sad": "😢",
    "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲",
}
ACCENT = "#7c6af7"
ACCENT2 = "#5ba3f5"
COND_LABELS = {"clean": "Clean", "snr_20db": "20 dB SNR",
               "snr_10db": "10 dB SNR", "snr_0db": "0 dB SNR"}
COND_ORDER = ["clean", "snr_20db", "snr_10db", "snr_0db"]


def model_ready() -> bool:
    return ((MODEL_DIR / "emotionsense_cnn_lstm.keras").exists()
            and (MODEL_DIR / "label_mapping.json").exists())


def evaluation_ready() -> bool:
    return (OUTPUT_DIR / "noise_evaluation_metrics.csv").exists()


def waveform_fig(audio: np.ndarray, sr: int) -> plt.Figure:
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 2.4))
        times = np.arange(len(audio)) / sr
        ax.plot(times, audio, linewidth=0.55, color=ACCENT, alpha=0.9)
        ax.fill_between(times, audio, alpha=0.15, color=ACCENT2)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Amplitude", fontsize=9)
        ax.set_xlim(0, max(times[-1], 0.01))
        ax.set_ylim(-1.05, 1.05)
        fig.tight_layout(pad=0.4)
    return fig


def mel_spectrogram_fig(audio: np.ndarray, sr: int) -> plt.Figure:
    with plt.rc_context(MPL_STYLE):
        fig, ax = plt.subplots(figsize=(8, 2.8))
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=160,
                                       x_axis="time", y_axis="mel", ax=ax,
                                       fmax=8000, cmap="magma")
        fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)
        fig.tight_layout(pad=0.4)
    return fig


def probability_bars_html(class_probs: list[dict]) -> str:
    rows = ""
    for item in sorted(class_probs, key=lambda x: x["confidence"], reverse=True):
        pct = item["confidence"] * 100
        emoji = EMOTION_EMOJIS.get(item["emotion"], "")
        rows += (
            f'<div class="prob-row">'
            f'<span class="prob-label">{emoji} {item["emotion"]}</span>'
            f'<div class="prob-track"><div class="prob-fill" style="width:{pct:.1f}%"></div></div>'
            f'<span class="prob-pct">{pct:.1f}%</span></div>'
        )
    return rows


def segment_timeline_html(audio: np.ndarray, sr: int, seg_sec: float = 2.5) -> str:
    import soundfile as sf
    seg_len = int(seg_sec * sr)
    segments = []
    for start in range(0, len(audio) - seg_len + 1, seg_len):
        segments.append((start / sr, (start + seg_len) / sr, audio[start:start + seg_len]))
    if not segments:
        return ""
    rows_html = ""
    for t0, t1, seg in segments:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, seg, sr)
            tmp_path = tmp.name
        try:
            res = predict_file(tmp_path)
            em = res["predicted_emotion"]
            conf = res["softmax_confidence"] * 100
            emoji = EMOTION_EMOJIS.get(em, "")
            rows_html += (
                f'<div class="seg-row">'
                f'<span class="seg-time">{t0:.1f}–{t1:.1f}s</span>'
                f'<span class="seg-label">{emoji} {em.capitalize()}</span>'
                f'<span class="seg-conf">{conf:.0f}%</span></div>'
            )
        except Exception:
            rows_html += (
                f'<div class="seg-row">'
                f'<span class="seg-time">{t0:.1f}–{t1:.1f}s</span>'
                f'<span class="seg-label">—</span></div>'
            )
        finally:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
    return rows_html


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎙️ EmotionSense</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🎤  Analyze Audio", "📊  Noise Robustness", "ℹ️  Setup Guide"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.75rem;color:#4a5568;line-height:1.5;">'
        "EmotionSense Studio<br>CNN-LSTM · RAVDESS · Local CPU<br>Academic mini-project</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

_ready = model_ready()
_cls = "status-ready" if _ready else "status-missing"
_dot_extra = "" if _ready else " status-dot-pulse"
_msg = "Model ready — predictions enabled" if _ready else "No trained model — run python train.py"

st.markdown(
    f"""<div class="hero-banner">
        <div class="hero-title">EmotionSense Studio</div>
        <p class="hero-sub">
            Noise-aware speech emotion recognition using CNN-LSTM · RAVDESS dataset · Local CPU inference
        </p>
        <span class="status-pill {_cls}">
            <span class="status-dot{_dot_extra}"></span>{_msg}
        </span>
    </div>""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — ANALYZE AUDIO
# ═══════════════════════════════════════════════════════════════════════════════

if "🎤" in page:
    if not _ready:
        st.markdown(
            """<div class="setup-box">
            <h3>🔧 Model not found</h3>
            <p>The trained CNN-LSTM model is missing. Follow the Setup Guide (left sidebar) to train it.</p>
            <pre>python train.py   # then:
python evaluate.py
streamlit run app.py</pre>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("### Upload a speech recording")
        uploaded = st.file_uploader(
            "Drop a WAV, MP3, FLAC, or OGG file here",
            type=["wav", "mp3", "flac", "ogg"],
            label_visibility="collapsed",
        )

        if not uploaded:
            st.markdown(
                """<div class="es-card" style="text-align:center;padding:3rem;color:#4a5568;">
                    <div style="font-size:3rem;margin-bottom:0.8rem;">🎵</div>
                    <div style="font-size:1rem;font-weight:500;color:#9ba6c2;">
                        Upload a speech recording to see the prediction</div>
                    <div style="font-size:0.8rem;margin-top:0.4rem;">
                        Supported: WAV · MP3 · FLAC · OGG</div>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            suffix = Path(uploaded.name).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                temp_path = tmp.name
            try:
                audio, sr = librosa.load(temp_path, sr=SAMPLE_RATE, mono=True)
                st.audio(uploaded)

                tab_wave, tab_mel = st.tabs(["🌊  Waveform", "🎛️  Mel Spectrogram"])
                with tab_wave:
                    fig_w = waveform_fig(audio, sr)
                    st.pyplot(fig_w, use_container_width=True)
                    plt.close(fig_w)
                with tab_mel:
                    try:
                        fig_m = mel_spectrogram_fig(audio, sr)
                        st.pyplot(fig_m, use_container_width=True)
                        plt.close(fig_m)
                    except Exception as exc:
                        st.warning(f"Could not render Mel spectrogram: {exc}")

                st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)

                with st.spinner("Running inference…"):
                    result = predict_file(temp_path)

                emotion = result["predicted_emotion"]
                conf    = result["softmax_confidence"]
                emoji   = EMOTION_EMOJIS.get(emotion, "")
                aq      = result.get("audio_quality", {})

                col_pred, col_quality = st.columns([3, 2])

                with col_pred:
                    st.markdown(
                        f"""<div class="es-card">
                            <div class="es-card-title">🧠 Prediction</div>
                            <div style="display:flex;align-items:center;gap:14px;margin-bottom:1.2rem;">
                                <div style="font-size:3.2rem;">{emoji}</div>
                                <div>
                                    <div style="font-size:1.8rem;font-weight:700;
                                         font-family:'Space Grotesk',sans-serif;color:#f0f4ff;">
                                        {emotion.capitalize()}</div>
                                    <div style="font-size:1.1rem;color:#7c6af7;font-weight:600;">
                                        {conf*100:.1f}% softmax confidence</div>
                                </div>
                            </div>
                            <div class="es-card-title" style="margin-top:0.8rem;">Class probabilities</div>
                            {probability_bars_html(result["class_probabilities"])}
                        </div>""",
                        unsafe_allow_html=True,
                    )

                with col_quality:
                    dur   = aq.get("duration_seconds", len(audio) / sr)
                    peak  = aq.get("peak_amplitude", float(np.max(np.abs(audio))))
                    clip  = aq.get("clipping_fraction", 0.0)
                    rms   = aq.get("rms_amplitude", 0.0)
                    silent = aq.get("silence_like", False)

                    st.markdown('<div class="es-card"><div class="es-card-title">📊 Audio Quality</div>',
                                unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    c1.metric("Duration", f"{dur:.2f} s")
                    c2.metric("Peak Amplitude", f"{peak:.3f}")
                    c3, c4 = st.columns(2)
                    c3.metric("RMS Level", f"{rms:.4f}")
                    c4.metric("Clipped", f"{clip*100:.2f}%")

                    if silent:
                        st.warning("⚠️ Near-silent or extremely quiet recording.")
                    if clip > 0.01:
                        st.error(f"🔴 {clip*100:.1f}% of samples may be clipped.")
                    st.markdown(
                        '<p style="font-size:0.73rem;color:#4a5568;margin-top:0.8rem;line-height:1.5;">'
                        "These checks flag obvious recording issues. They do not measure environmental noise or SNR."
                        "</p></div>",
                        unsafe_allow_html=True,
                    )

                # Emotion timeline for longer recordings
                seg_sec = 2.5
                dur = len(audio) / sr
                if dur > seg_sec * 2:
                    st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="es-card"><div class="es-card-title">🕐 Emotion Timeline (2.5-second segments)</div>',
                        unsafe_allow_html=True,
                    )
                    with st.spinner("Segmenting…"):
                        timeline_html = segment_timeline_html(audio, sr, seg_sec)
                    if timeline_html:
                        st.markdown(timeline_html, unsafe_allow_html=True)
                    else:
                        st.info("Recording too short to segment into multiple windows.")
                    st.markdown("</div>", unsafe_allow_html=True)

            except Exception as exc:
                st.error(f"❌ Could not process this audio: {exc}")
            finally:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — NOISE ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════════════

elif "📊" in page:
    st.markdown("## 📊 Noise Robustness Evaluation")
    st.markdown(
        '<p style="color:#9ba6c2;font-size:0.9rem;margin-top:-0.5rem;margin-bottom:1.4rem;">'
        "Held-out test set evaluated under clean audio and synthetic additive white noise at 20, 10, and 0 dB SNR."
        "</p>",
        unsafe_allow_html=True,
    )

    metrics_path = OUTPUT_DIR / "noise_evaluation_metrics.csv"
    if not metrics_path.exists():
        st.markdown(
            """<div class="setup-box">
            <h3>⏳ Evaluation not yet run</h3>
            <p>No evaluation results found. Generate them with:</p>
            <pre>python evaluate.py</pre>
            <p>This reads the trained model and test manifest produced by <code>train.py</code>.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        df_metrics = pd.read_csv(metrics_path)
        ordered = df_metrics.set_index("condition").reindex(
            [c for c in COND_ORDER if c in df_metrics["condition"].values]
        ).reset_index()

        # Summary metric tiles
        cols = st.columns(len(ordered))
        for col, (_, row) in zip(cols, ordered.iterrows()):
            label = COND_LABELS.get(row["condition"], row["condition"])
            delta_acc = row.get("accuracy_change_vs_clean", None)
            delta_str = (f"{delta_acc*100:+.1f}%"
                         if (delta_acc is not None and not np.isnan(float(delta_acc))
                             and row["condition"] != "clean") else None)
            with col:
                st.metric(f"Accuracy — {label}", f"{row['accuracy']*100:.1f}%",
                          delta=delta_str, delta_color="normal")

        st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)

        t1, t2, t3 = st.tabs(["📈  Charts", "🔢  Full Metrics Table", "🔷  Confusion Matrices"])

        with t1:
            c_left, c_right = st.columns(2)
            for col, metric, label in [(c_left, "accuracy", "Accuracy"), (c_right, "macro_f1", "Macro F1")]:
                with col:
                    saved_img = OUTPUT_DIR / f"{metric}_by_noise.png"
                    if saved_img.exists():
                        st.image(str(saved_img), caption=f"{label} by noise condition",
                                 use_container_width=True)
                    else:
                        with plt.rc_context(MPL_STYLE):
                            fig, ax = plt.subplots(figsize=(5, 3.5))
                            xlabels = [COND_LABELS.get(c, c) for c in ordered["condition"]]
                            vals = ordered[metric].tolist()
                            bars = ax.bar(xlabels, vals,
                                          color=[ACCENT, ACCENT2, "#a89cff", "#4a5568"],
                                          width=0.55, edgecolor="none")
                            ax.set_ylim(0, 1.05)
                            ax.set_ylabel(label, fontsize=9)
                            ax.set_title(f"{label} by Noise Condition", fontsize=10, pad=8)
                            for bar, v in zip(bars, vals):
                                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                                        f"{v*100:.1f}%", ha="center", va="bottom",
                                        fontsize=8, color="#f0f4ff")
                            fig.tight_layout(pad=0.4)
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)

            lc_path = OUTPUT_DIR / "learning_curves.png"
            if lc_path.exists():
                st.markdown('<div class="es-sep"></div>', unsafe_allow_html=True)
                st.image(str(lc_path), caption="Training learning curves (loss & accuracy)",
                         use_container_width=True)

        with t2:
            display_df = ordered.copy()
            display_df["condition"] = display_df["condition"].map(lambda c: COND_LABELS.get(c, c))
            rename_map = {
                "condition": "Condition", "n_test_files": "Test Files",
                "accuracy": "Accuracy", "macro_precision": "Macro Precision",
                "macro_recall": "Macro Recall", "macro_f1": "Macro F1",
                "accuracy_change_vs_clean": "Δ Accuracy",
                "macro_f1_change_vs_clean": "Δ Macro F1",
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items()
                                                      if k in display_df.columns})
            for col in ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: f"{x*100:.2f}%")
            for col in ["Δ Accuracy", "Δ Macro F1"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].map(lambda x: f"{float(x)*100:+.2f}%")
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.markdown(
                '<p style="font-size:0.75rem;color:#4a5568;margin-top:0.4rem;">'
                "Synthetic white Gaussian noise is a controlled approximation; "
                "real-world noise (babble, traffic) is not modelled here.</p>",
                unsafe_allow_html=True,
            )

        with t3:
            available_conds = [c for c in COND_ORDER if c in df_metrics["condition"].values]
            cond_select = st.selectbox(
                "Select condition", options=available_conds,
                format_func=lambda c: COND_LABELS.get(c, c), key="cm_select",
            )
            cm_path = OUTPUT_DIR / f"confusion_matrix_{cond_select}.png"
            if cm_path.exists():
                st.image(str(cm_path),
                         caption=f"Confusion matrix — {COND_LABELS.get(cond_select, cond_select)}",
                         use_container_width=True)
            else:
                st.info("Confusion matrix image not found. Re-run `python evaluate.py`.")

            cr_path = OUTPUT_DIR / f"classification_report_{cond_select}.csv"
            if cr_path.exists():
                st.markdown("**Per-class classification report**")
                cr_df = pd.read_csv(cr_path, index_col=0)
                num_cols = [c for c in ["precision", "recall", "f1-score"]
                            if c in cr_df.columns]
                fmt_df = cr_df.copy()
                for nc in num_cols:
                    fmt_df[nc] = pd.to_numeric(fmt_df[nc], errors="coerce")
                st.dataframe(fmt_df.style.format(
                    {c: "{:.3f}" for c in num_cols}, na_rep="—"),
                    use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — SETUP GUIDE
# ═══════════════════════════════════════════════════════════════════════════════

elif "ℹ️" in page:
    st.markdown("## ℹ️ Setup Guide")
    st.markdown(
        """<div class="setup-box">
        <h3>📦 Step 1 — Install dependencies</h3>
        <pre>pip install -r requirements.txt</pre>

        <h3>🎵 Step 2 — Download RAVDESS</h3>
        <p>Open <code>https://zenodo.org/records/1188976</code><br>
        Download <strong>Audio_Speech_Actors_01-24.zip</strong> and extract so actor folders sit under:</p>
        <pre>data/ravdess/Actor_01/03-01-05-01-01-01-01.wav
data/ravdess/Actor_02/...</pre>

        <h3>🔍 Step 3 — Inspect dataset</h3>
        <pre>python scripts/inspect_dataset.py</pre>

        <h3>🏋️ Step 4 — Train the model</h3>
        <pre>python train.py</pre>
        <p>Saves <code>models/emotionsense_cnn_lstm.keras</code> and <code>models/label_mapping.json</code>.</p>

        <h3>📊 Step 5 — Evaluate</h3>
        <pre>python evaluate.py</pre>

        <h3>🚀 Step 6 — Launch dashboard</h3>
        <pre>streamlit run app.py</pre>

        <h3>🧪 Run tests</h3>
        <pre>pytest -q</pre>
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("### Current project status")
    status_items = [
        ("Model weights",    (MODEL_DIR / "emotionsense_cnn_lstm.keras").exists()),
        ("Label mapping",    (MODEL_DIR / "label_mapping.json").exists()),
        ("Test manifest",    (OUTPUT_DIR / "test_manifest.csv").exists()),
        ("Evaluation CSV",   (OUTPUT_DIR / "noise_evaluation_metrics.csv").exists()),
        ("Training history", (OUTPUT_DIR / "training_history.csv").exists()),
    ]
    for name, ok in status_items:
        icon = "✅" if ok else "❌"
        st.markdown(f"{icon} &nbsp; **{name}**", unsafe_allow_html=True)
