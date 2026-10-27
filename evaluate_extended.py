"""evaluate_extended.py — Extended evaluation for EmotionSense.

Covers Task D (multi-noise-type evaluation) and Task E (full metrics: accuracy,
macro-F1, weighted-F1, per-class P/R/F1, confusion matrices, learning curves).

Runs all available models under:
  - AWGN:       clean / 20 dB / 10 dB / 0 dB SNR
  - Pink noise: 20 dB / 10 dB / 0 dB SNR (locally generated, no downloads)
  - Hum noise:  20 dB / 10 dB / 0 dB SNR (50 Hz harmonics, locally generated)

Saves:
  outputs/extended_noise_comparison.csv   — machine-readable master table
  outputs/extended_noise_comparison.json  — JSON version
  outputs/[model]_[noise]_[cond]_cm.png  — confusion matrices
  outputs/[model]_[noise]_[cond]_report.csv — per-class classification reports
  outputs/snr_accuracy_extended.png       — accuracy vs SNR by noise type
  outputs/snr_f1_extended.png             — macro-F1 vs SNR by noise type

Reuses evaluate_all.py helper logic and avoids re-extracting features when
possible. Does NOT touch training splits or model selection.

Run: python evaluate_extended.py
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)

from config import OUTPUT_DIR, MODEL_DIR
from src.preprocessing import (load_audio, extract_mfcc, extract_mfcc_with_deltas,
                                extract_svm_features)
from src.noise import add_white_noise_at_snr, add_pink_noise_at_snr, add_harmonic_noise_at_snr

# ── Noise conditions ───────────────────────────────────────────────────────────
AWGN_CONDITIONS  = [("awgn",     "clean", None),
                    ("awgn",    "20db",  20),
                    ("awgn",    "10db",  10),
                    ("awgn",    "0db",    0)]
PINK_CONDITIONS  = [("pink",    "20db",  20),
                    ("pink",    "10db",  10),
                    ("pink",    "0db",    0)]
HUM_CONDITIONS   = [("hum",     "20db",  20),
                    ("hum",     "10db",  10),
                    ("hum",     "0db",    0)]
ALL_CONDITIONS   = AWGN_CONDITIONS + PINK_CONDITIONS + HUM_CONDITIONS

NOISE_FN_MAP = {
    "awgn": add_white_noise_at_snr,
    "pink": add_pink_noise_at_snr,
    "hum":  add_harmonic_noise_at_snr,
}
COND_DISPLAY = {"clean": "Clean", "20db": "20 dB", "10db": "10 dB", "0db": "0 dB"}
NOISE_DISPLAY = {"awgn": "AWGN", "pink": "Pink", "hum": "Hum (50 Hz)"}

# ── Model configs ──────────────────────────────────────────────────────────────
MODEL_CONFIGS = [
    {
        "key": "original", "label": "Original CNN-LSTM",
        "type": "keras",
        "model_path": MODEL_DIR / "emotionsense_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn": extract_mfcc,
        "feature_name": "MFCC-40",
    },
    {
        "key": "improved", "label": "Improved CNN-LSTM",
        "type": "keras",
        "model_path": MODEL_DIR / "improved_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "improved_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "improved_test_manifest.csv",
        "feat_fn": extract_mfcc_with_deltas,
        "feature_name": "MFCC+Delta",
    },
    {
        "key": "svm", "label": "SVM Baseline",
        "type": "svm",
        "model_path": MODEL_DIR / "svm_model.pkl",
        "scaler_path": MODEL_DIR / "svm_scaler.pkl",
        "mapping_path": MODEL_DIR / "svm_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn": extract_svm_features,
        "feature_name": "MFCC-stats-160",
    },
]


def load_manifest(path: Path) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    return df["path"].tolist(), df["y_true"].to_numpy(dtype=int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    labels: list[str]) -> dict:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro",
                        labels=range(len(labels)), zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted",
                           labels=range(len(labels)), zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=labels,
        labels=range(len(labels)), zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "n_samples": int(len(y_true)),
    }


def save_cm_figure(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, linewidths=0.4)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=11)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_keras_model(cfg: dict, paths: list[str], y_true: np.ndarray,
                          noise_type: str, snr_db, seed_base: int = 3000) -> dict:
    model = tf.keras.models.load_model(cfg["model_path"])
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    labels = [mapping[str(i)] for i in range(len(mapping))]
    feat_fn = cfg["feat_fn"]
    noise_fn = NOISE_FN_MAP[noise_type]

    feats, idx = [], []
    for i, p in enumerate(paths):
        try:
            audio = load_audio(p)
            if snr_db is not None:
                audio = noise_fn(audio, snr_db=snr_db, seed=seed_base + i)
            feats.append(feat_fn(audio))
            idx.append(i)
        except Exception as exc:
            print(f"  Skip {Path(p).name}: {exc}")
    if not feats:
        return {}
    idx_arr = np.asarray(idx)
    probs = model.predict(np.stack(feats), batch_size=32, verbose=0)
    preds = probs.argmax(axis=1)
    return compute_metrics(y_true[idx_arr], preds, labels)


def evaluate_svm_model(cfg: dict, paths: list[str], y_true: np.ndarray,
                        noise_type: str, snr_db, seed_base: int = 3000) -> dict:
    with open(cfg["model_path"], "rb") as f:
        svm = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    labels = [mapping[str(i)] for i in range(len(mapping))]
    feat_fn = cfg["feat_fn"]
    noise_fn = NOISE_FN_MAP[noise_type]

    feats, idx = [], []
    for i, p in enumerate(paths):
        try:
            audio = load_audio(p)
            if snr_db is not None:
                audio = noise_fn(audio, snr_db=snr_db, seed=seed_base + i)
            feats.append(feat_fn(audio))
            idx.append(i)
        except Exception as exc:
            print(f"  Skip {Path(p).name}: {exc}")
    if not feats:
        return {}
    idx_arr = np.asarray(idx)
    X = scaler.transform(np.stack(feats))
    preds = svm.predict(X)
    return compute_metrics(y_true[idx_arr], preds, labels)


def make_snr_plots(df: pd.DataFrame) -> None:
    """Plot accuracy and macro-F1 vs. SNR for each model × noise type."""
    snr_map = {"clean": None, "20db": 20, "10db": 10, "0db": 0}
    cond_order_awgn = ["clean", "20db", "10db", "0db"]
    cond_order_noisy = ["20db", "10db", "0db"]
    colors = {"original": "#4c72b0", "improved": "#c44e52", "svm": "#55a868"}
    ls_map  = {"awgn": "-", "pink": "--", "hum": ":"}

    for metric, ylabel in [("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
        for ax, noise_type in zip(axes, ["awgn", "pink", "hum"]):
            cond_order = cond_order_awgn if noise_type == "awgn" else cond_order_noisy
            snr_vals = [snr_map[c] for c in cond_order]
            x_vals = [0 if s is None else s for s in snr_vals]  # use 0 for clean
            for mk, color in colors.items():
                sub = df[(df["model_key"] == mk) & (df["noise_type"] == noise_type)]
                if sub.empty:
                    continue
                sub = sub.set_index("snr_label")
                ys = [float(sub.loc[c, metric]) if c in sub.index else np.nan
                      for c in cond_order]
                ax.plot(x_vals[:len(ys)], ys,
                        label=df.loc[df["model_key"]==mk, "model"].iloc[0],
                        color=color, linewidth=2.0, marker="o", markersize=6)
            ax.set_title(f"{NOISE_DISPLAY[noise_type]}", fontsize=11)
            ax.set_xlabel("Target SNR (dB)" if noise_type != "awgn" else "SNR (dB, ∞=clean)")
            ax.set_ylabel(ylabel if noise_type == "awgn" else "")
            ax.grid(linestyle="--", alpha=0.4)
            ax.set_ylim(0, 1.05)
            if noise_type == "awgn":
                ax.set_xticks([0, 20, 10, 0])
                ax.set_xticklabels(["Clean", "20", "10", "0"])
        axes[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"EmotionSense — {ylabel} vs SNR by Noise Type", fontsize=12)
        fig.tight_layout()
        stem = "snr_accuracy_extended" if metric == "accuracy" else "snr_f1_extended"
        fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=160)
        plt.close(fig)
        print(f"  Saved {stem}.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for cfg in MODEL_CONFIGS:
        if not cfg["model_path"].exists():
            print(f"[SKIP] {cfg['label']}: model not found.")
            continue

        paths, y_true = load_manifest(cfg["manifest_path"])
        print(f"\n{'='*60}\n Evaluating: {cfg['label']}\n{'='*60}")

        for noise_type, snr_label, snr_db in ALL_CONDITIONS:
            cond_str = f"{noise_type}/{snr_label}"
            print(f"  {cond_str} …", end="  ", flush=True)

            if cfg["type"] == "keras":
                result = evaluate_keras_model(cfg, paths, y_true, noise_type, snr_db)
            else:
                result = evaluate_svm_model(cfg, paths, y_true, noise_type, snr_db)

            if not result:
                print("no result")
                continue

            print(f"acc={result['accuracy']:.4f}  macro_f1={result['macro_f1']:.4f}  "
                  f"weighted_f1={result['weighted_f1']:.4f}")

            # Save per-condition CM and classification report
            cm = np.array(result["confusion_matrix"])
            mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
            labels = [mapping[str(i)] for i in range(len(mapping))]
            cm_title = (f"{cfg['label']} — {NOISE_DISPLAY.get(noise_type, noise_type)} "
                        f"{COND_DISPLAY.get(snr_label, snr_label)}")
            cm_path = OUTPUT_DIR / f"{cfg['key']}_{noise_type}_{snr_label}_cm.png"
            save_cm_figure(cm, labels, cm_title, cm_path)

            report_path = OUTPUT_DIR / f"{cfg['key']}_{noise_type}_{snr_label}_report.csv"
            pd.DataFrame(result["report"]).T.to_csv(report_path)

            all_records.append({
                "model": cfg["label"],
                "model_key": cfg["key"],
                "feature_config": cfg["feature_name"],
                "noise_type": noise_type,
                "snr_label": snr_label,
                "snr_db": snr_db if snr_db is not None else float("inf"),
                "n_samples": result["n_samples"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
            })

    if not all_records:
        raise SystemExit("No models available for evaluation.")

    df = pd.DataFrame(all_records)
    df.to_csv(OUTPUT_DIR / "extended_noise_comparison.csv", index=False)
    df.to_json(OUTPUT_DIR / "extended_noise_comparison.json", orient="records", indent=2)
    print(f"\n{'='*60}\n EXTENDED SUMMARY\n{'='*60}")
    print(df[["model", "noise_type", "snr_label", "accuracy", "macro_f1", "weighted_f1"]]
          .to_string(index=False))

    make_snr_plots(df)
    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
