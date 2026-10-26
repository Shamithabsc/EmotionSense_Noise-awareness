from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless and Streamlit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from config import OUTPUT_DIR, MODEL_DIR
from src.preprocessing import load_audio, extract_mfcc
from src.noise import add_white_noise_at_snr
from src.metrics import classification_metrics

COND_DISPLAY = {
    "clean":    "Clean",
    "snr_20db": "20 dB SNR",
    "snr_10db": "10 dB SNR",
    "snr_0db":  "0 dB SNR",
}


def _bar_chart(results: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Save a styled bar chart for a given metric across noise conditions."""
    cond_order = ["clean", "snr_20db", "snr_10db", "snr_0db"]
    ordered = results.set_index("condition").reindex(
        [c for c in cond_order if c in results["condition"].values]
    ).reset_index()
    labels = [COND_DISPLAY.get(c, c) for c in ordered["condition"]]
    vals   = ordered[metric].tolist()
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"][:len(vals)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.6)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{metric.replace('_', ' ').title()} by Noise Condition", fontsize=13, pad=10)
    ax.set_xlabel("Test audio condition", fontsize=11)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, v + 0.012,
            f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _learning_curves(history_path: Path, out_path: Path) -> None:
    """Plot training and validation loss/accuracy from the saved CSV."""
    if not history_path.exists():
        return
    hist = pd.read_csv(history_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist) + 1)

    for ax, metric, label in [
        (axes[0], "loss",     "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        if metric in hist.columns:
            ax.plot(epochs, hist[metric],     label="Train", linewidth=1.8)
        val_col = f"val_{metric}"
        if val_col in hist.columns:
            ax.plot(epochs, hist[val_col], label="Validation", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"Training {label}")
        ax.legend()
        ax.grid(linestyle="--", alpha=0.4)

    fig.suptitle("EmotionSense — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    model_path   = MODEL_DIR / "emotionsense_cnn_lstm.keras"
    mapping_path = MODEL_DIR / "label_mapping.json"
    if not model_path.exists() or not mapping_path.exists():
        raise SystemExit("Trained model or label mapping missing. Run `python train.py` first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model   = tf.keras.models.load_model(model_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels  = [mapping[str(i)] for i in range(len(mapping))]
    print(f"Loaded model. Classes: {labels}")

    manifest_path = OUTPUT_DIR / "test_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit("Test manifest missing. Run `python train.py` first.")
    manifest  = pd.read_csv(manifest_path)
    y_true_all = manifest.y_true.to_numpy(dtype=int)
    print(f"Test set: {len(manifest)} files.")

    records: list[dict] = []
    conditions = [("clean", None), ("snr_20db", 20), ("snr_10db", 10), ("snr_0db", 0)]

    for condition, snr in conditions:
        print(f"\n--- Condition: {condition} ---")
        features: list[np.ndarray] = []
        valid_idx: list[int]       = []

        for i, row in enumerate(manifest.itertuples(index=False)):
            try:
                audio = load_audio(row.path)
                if snr is not None:
                    audio = add_white_noise_at_snr(audio, snr_db=snr, seed=1000 + i)
                features.append(extract_mfcc(audio))
                valid_idx.append(i)
            except Exception as exc:
                print(f"  Skipping {Path(row.path).name}: {exc}")

        if not features:
            print(f"  No usable files for {condition}; skipping.")
            continue

        idx        = np.asarray(valid_idx)
        true_sub   = y_true_all[idx]
        probs      = model.predict(np.stack(features), batch_size=32, verbose=0)
        pred       = probs.argmax(axis=1)
        result     = classification_metrics(true_sub, pred, labels)

        print(f"  Accuracy={result['accuracy']:.4f}  "
              f"P={result['macro_precision']:.4f}  "
              f"R={result['macro_recall']:.4f}  "
              f"F1={result['macro_f1']:.4f}")

        records.append({
            "condition":       condition,
            "n_test_files":    int(len(idx)),
            "accuracy":        result["accuracy"],
            "macro_precision": result["macro_precision"],
            "macro_recall":    result["macro_recall"],
            "macro_f1":        result["macro_f1"],
        })

        # Per-class CSV
        pd.DataFrame(result["classification_report"]).T.to_csv(
            OUTPUT_DIR / f"classification_report_{condition}.csv"
        )

        # Confusion matrix heatmap
        cm_arr = np.array(result["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted emotion", fontsize=11)
        ax.set_ylabel("True emotion",      fontsize=11)
        ax.set_title(f"EmotionSense confusion matrix — {COND_DISPLAY.get(condition, condition)}",
                     fontsize=12)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{condition}.png", dpi=160)
        plt.close(fig)

    if not records:
        raise SystemExit("No evaluation results produced.")

    results = pd.DataFrame(records)

    # Add delta columns relative to clean
    if "clean" in results["condition"].values:
        clean_row = results.loc[results["condition"] == "clean"].iloc[0]
        results["accuracy_change_vs_clean"] = results["accuracy"] - clean_row["accuracy"]
        results["macro_f1_change_vs_clean"] = results["macro_f1"] - clean_row["macro_f1"]

    results.to_csv(OUTPUT_DIR / "noise_evaluation_metrics.csv", index=False)
    results.to_json(OUTPUT_DIR / "noise_evaluation_metrics.json", orient="records", indent=2)

    print("\n=== Summary ===")
    print(results[["condition", "n_test_files", "accuracy", "macro_f1"]].to_string(index=False))

    # Bar charts — use plain matplotlib, not seaborn barplot (API changed in 0.13)
    _bar_chart(results, "accuracy",  OUTPUT_DIR / "accuracy_by_noise.png")
    _bar_chart(results, "macro_f1",  OUTPUT_DIR / "macro_f1_by_noise.png")

    # Learning curves from training history
    _learning_curves(OUTPUT_DIR / "training_history.csv",
                     OUTPUT_DIR / "learning_curves.png")

    print(f"\nAll evaluation outputs saved under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

