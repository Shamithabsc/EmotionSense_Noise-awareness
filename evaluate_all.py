"""evaluate_all.py — Compare all EmotionSense models across noise conditions.

Evaluates any combination of:
  - Original CNN-LSTM (models/emotionsense_cnn_lstm.keras)
  - Improved CNN-LSTM with delta features + augmentation (models/improved_cnn_lstm.keras)
  - SVM baseline (models/svm_model.pkl)

For each model, runs evaluation under:
  clean / 20 dB SNR / 10 dB SNR / 0 dB SNR

Saves:
  outputs/all_models_comparison.csv / .json
  outputs/[model]_confusion_matrix_[condition].png
  outputs/[model]_classification_report_[condition].csv
  outputs/accuracy_comparison.png / macro_f1_comparison.png

Run:   python evaluate_all.py
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

from config import OUTPUT_DIR, MODEL_DIR
from src.preprocessing import load_audio, extract_mfcc, extract_mfcc_with_deltas, extract_svm_features
from src.noise import add_white_noise_at_snr
from src.metrics import classification_metrics

SNR_CONDITIONS = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]
COND_DISPLAY   = {"clean": "Clean", "snr_20db": "20 dB", "snr_10db": "10 dB", "snr_0db": "0 dB"}

MODEL_CONFIGS = [
    {
        "key":      "original",
        "label":    "Original CNN-LSTM",
        "type":     "keras",
        "model_path":   MODEL_DIR / "emotionsense_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn":  extract_mfcc,
    },
    {
        "key":      "improved",
        "label":    "Improved CNN-LSTM",
        "type":     "keras",
        "model_path":   MODEL_DIR / "improved_cnn_lstm.keras",
        "mapping_path": MODEL_DIR / "improved_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "improved_test_manifest.csv",
        "feat_fn":  extract_mfcc_with_deltas,
    },
    {
        "key":      "svm",
        "label":    "SVM Baseline",
        "type":     "svm",
        "model_path":   MODEL_DIR / "svm_model.pkl",
        "scaler_path":  MODEL_DIR / "svm_scaler.pkl",
        "mapping_path": MODEL_DIR / "svm_label_mapping.json",
        "manifest_path": OUTPUT_DIR / "test_manifest.csv",
        "feat_fn":  extract_svm_features,
    },
]


def _confusion_fig(cm: np.ndarray, labels: list[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    return fig


def evaluate_keras(cfg: dict) -> list[dict]:
    model   = tf.keras.models.load_model(cfg["model_path"])
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    labels  = [mapping[str(i)] for i in range(len(mapping))]
    manifest = pd.read_csv(cfg["manifest_path"])
    y_true_all = manifest.y_true.to_numpy(dtype=int)
    feat_fn = cfg["feat_fn"]
    key     = cfg["key"]
    records = []

    for cond, snr in SNR_CONDITIONS:
        feats, idx = [], []
        for i, row in enumerate(manifest.itertuples(index=False)):
            try:
                audio = load_audio(row.path)
                if snr is not None:
                    audio = add_white_noise_at_snr(audio, snr_db=snr, seed=2000 + i)
                feats.append(feat_fn(audio))
                idx.append(i)
            except Exception as e:
                print(f"  Skipping {Path(row.path).name}: {e}")
        if not feats:
            continue
        idx_arr   = np.asarray(idx)
        true_sub  = y_true_all[idx_arr]
        probs     = model.predict(np.stack(feats), batch_size=32, verbose=0)
        pred      = probs.argmax(axis=1)
        result    = classification_metrics(true_sub, pred, labels)
        print(f"  [{cfg['label']}] {cond}: acc={result['accuracy']:.4f}  f1={result['macro_f1']:.4f}")
        records.append({**_base_record(cfg, cond, idx_arr, result)})
        _save_outputs(result, labels, cfg["key"], cond)
    return records


def evaluate_svm(cfg: dict) -> list[dict]:
    with open(cfg["model_path"], "rb") as f:
        svm = pickle.load(f)
    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    mapping = json.loads(cfg["mapping_path"].read_text(encoding="utf-8"))
    labels  = [mapping[str(i)] for i in range(len(mapping))]
    manifest = pd.read_csv(cfg["manifest_path"])
    y_true_all = manifest.y_true.to_numpy(dtype=int)
    feat_fn = cfg["feat_fn"]
    records = []

    for cond, snr in SNR_CONDITIONS:
        feats, idx = [], []
        for i, row in enumerate(manifest.itertuples(index=False)):
            try:
                audio = load_audio(row.path)
                if snr is not None:
                    audio = add_white_noise_at_snr(audio, snr_db=snr, seed=2000 + i)
                feats.append(feat_fn(audio))
                idx.append(i)
            except Exception as e:
                print(f"  Skipping {Path(row.path).name}: {e}")
        if not feats:
            continue
        idx_arr  = np.asarray(idx)
        true_sub = y_true_all[idx_arr]
        X        = scaler.transform(np.stack(feats))
        pred     = svm.predict(X)
        result   = classification_metrics(true_sub, pred, labels)
        print(f"  [{cfg['label']}] {cond}: acc={result['accuracy']:.4f}  f1={result['macro_f1']:.4f}")
        records.append({**_base_record(cfg, cond, idx_arr, result)})
        _save_outputs(result, labels, cfg["key"], cond)
    return records


def _base_record(cfg: dict, cond: str, idx: np.ndarray, result: dict) -> dict:
    return {
        "model":           cfg["label"],
        "model_key":       cfg["key"],
        "condition":       cond,
        "n_test_files":    int(len(idx)),
        "accuracy":        result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall":    result["macro_recall"],
        "macro_f1":        result["macro_f1"],
    }


def _save_outputs(result: dict, labels: list[str], key: str, cond: str) -> None:
    cm  = np.array(result["confusion_matrix"])
    fig = _confusion_fig(cm, labels,
                         f"EmotionSense ({key}) — {COND_DISPLAY.get(cond, cond)}")
    fig.savefig(OUTPUT_DIR / f"{key}_confusion_matrix_{cond}.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(result["classification_report"]).T.to_csv(
        OUTPUT_DIR / f"{key}_classification_report_{cond}.csv"
    )


def _comparison_chart(results: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Grouped bar chart: x=noise condition, bars grouped by model."""
    cond_order = ["clean", "snr_20db", "snr_10db", "snr_0db"]
    models     = results["model_key"].unique().tolist()
    colors     = ["#4c72b0", "#55a868", "#c44e52"][:len(models)]
    x          = np.arange(len(cond_order))
    width      = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (mk, color) in enumerate(zip(models, colors)):
        sub = results[results["model_key"] == mk].set_index("condition")[metric]
        vals = [float(sub.get(c, np.nan)) for c in cond_order]
        bars = ax.bar(x + (i - len(models)/2 + 0.5) * width, vals,
                      width=width * 0.9, label=results.loc[results["model_key"]==mk, "model"].iloc[0],
                      color=color, edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.008,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([COND_DISPLAY.get(c, c) for c in cond_order])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"{metric.replace('_',' ').title()} — Model Comparison by Noise Condition",
                 fontsize=12, pad=8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_records: list[dict] = []

    for cfg in MODEL_CONFIGS:
        if not cfg["model_path"].exists():
            print(f"[SKIP] {cfg['label']}: model not found at {cfg['model_path']}")
            continue
        print(f"\n{'='*60}")
        print(f" Evaluating: {cfg['label']}")
        print(f"{'='*60}")
        if cfg["type"] == "keras":
            all_records.extend(evaluate_keras(cfg))
        elif cfg["type"] == "svm":
            all_records.extend(evaluate_svm(cfg))

    if not all_records:
        raise SystemExit("No models available for evaluation.")

    results = pd.DataFrame(all_records)

    # Add delta columns vs clean within each model
    rows = []
    for mk in results["model_key"].unique():
        sub = results[results["model_key"] == mk].copy()
        clean_row = sub[sub["condition"] == "clean"]
        if not clean_row.empty:
            clean_acc = clean_row["accuracy"].iloc[0]
            clean_f1  = clean_row["macro_f1"].iloc[0]
            sub["accuracy_delta_vs_clean"] = sub["accuracy"] - clean_acc
            sub["macro_f1_delta_vs_clean"] = sub["macro_f1"] - clean_f1
        rows.append(sub)
    results = pd.concat(rows, ignore_index=True)

    results.to_csv(OUTPUT_DIR / "all_models_comparison.csv", index=False)
    results.to_json(OUTPUT_DIR / "all_models_comparison.json", orient="records", indent=2)

    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    print(results[["model", "condition", "accuracy", "macro_f1"]].to_string(index=False))

    _comparison_chart(results, "accuracy",  OUTPUT_DIR / "accuracy_comparison.png")
    _comparison_chart(results, "macro_f1",  OUTPUT_DIR / "macro_f1_comparison.png")

    print(f"\nAll outputs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
