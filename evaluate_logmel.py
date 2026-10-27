"""evaluate_logmel.py — Evaluates the trained Log-Mel CNN-LSTM model across noise conditions.

Runs on held-out test actors (IDs 10, 16, 17, 19, 20 = 300 files) under:
  - Clean
  - AWGN 20 dB SNR
  - AWGN 10 dB SNR
  - AWGN 0 dB SNR

Saves:
  - outputs/logmel_classification_report_[cond].csv
  - outputs/logmel_confusion_matrix_[cond].png
  - outputs/logmel_noise_metrics.csv
Appends or updates all_models_comparison.csv with logmel entries.
"""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from config import OUTPUT_DIR, MODEL_DIR
from src.preprocessing import load_audio, extract_log_mel
from src.noise import add_white_noise_at_snr

LOGMEL_MODEL = MODEL_DIR / "logmel_cnn_lstm.keras"
LOGMEL_MAP = MODEL_DIR / "logmel_label_mapping.json"
MANIFEST_PATH = OUTPUT_DIR / "logmel_test_manifest.csv"

SNR_CONDITIONS = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]
COND_DISPLAY = {"clean": "Clean", "snr_20db": "20 dB", "snr_10db": "10 dB", "snr_0db": "0 dB"}

def main():
    if not LOGMEL_MODEL.exists():
        print(f"Log-mel model not found at {LOGMEL_MODEL}")
        return

    model = tf.keras.models.load_model(LOGMEL_MODEL)
    with open(LOGMEL_MAP, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = [mapping[str(i)] for i in range(len(mapping))]

    test_df = pd.read_csv(MANIFEST_PATH)
    paths = test_df["path"].tolist()
    y_true = test_df["y_true"].to_numpy(dtype=int)

    results = []
    clean_acc, clean_f1 = None, None

    for cond_key, snr in SNR_CONDITIONS:
        print(f"Evaluating Log-Mel under {cond_key}...")
        feats = []
        for p in paths:
            audio = load_audio(p)
            if snr is not None:
                audio = add_white_noise_at_snr(audio, snr_db=snr, seed=42)
            feat = extract_log_mel(audio)
            feats.append(feat)
        X = np.stack(feats, axis=0)
        probs = model.predict(X, batch_size=32, verbose=0)
        y_pred = probs.argmax(axis=1)

        acc = accuracy_score(y_true, y_pred)
        macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if cond_key == "clean":
            clean_acc = acc
            clean_f1 = macro_f1

        delta_acc = acc - clean_acc if clean_acc is not None else 0.0
        delta_f1 = macro_f1 - clean_f1 if clean_f1 is not None else 0.0

        results.append({
            "model_key": "logmel",
            "model": "Log-Mel CNN-LSTM",
            "feature": "LogMel-64",
            "condition": cond_key,
            "snr_db": snr if snr is not None else "inf",
            "n_test_files": len(paths),
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy_delta_vs_clean": delta_acc,
            "macro_f1_delta_vs_clean": delta_f1,
        })

        # Classification report
        rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).transpose()
        rep_df.to_csv(OUTPUT_DIR / f"logmel_classification_report_{cond_key}.csv")

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title(f"Log-Mel CNN-LSTM — {COND_DISPLAY[cond_key]}")
        ax.set_ylabel("True Emotion")
        ax.set_xlabel("Predicted Emotion")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"logmel_confusion_matrix_{cond_key}.png", dpi=150)
        plt.close(fig)

    logmel_df = pd.DataFrame(results)
    logmel_df.to_csv(OUTPUT_DIR / "logmel_noise_metrics.csv", index=False)
    print("Log-Mel evaluation complete:")
    print(logmel_df[["condition", "accuracy", "macro_f1", "weighted_f1"]])

    # Append to all_models_comparison.csv if it exists
    comp_file = OUTPUT_DIR / "all_models_comparison.csv"
    if comp_file.exists():
        comp_df = pd.read_csv(comp_file)
        # Drop old logmel entries if present
        comp_df = comp_df[comp_df["model_key"] != "logmel"]
        updated_comp = pd.concat([comp_df, logmel_df], ignore_index=True)
        updated_comp.to_csv(comp_file, index=False)
        print(f"Updated {comp_file} with Log-Mel results.")

if __name__ == "__main__":
    main()
