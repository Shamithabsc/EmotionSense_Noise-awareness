"""train_logmel.py — CNN-LSTM with log-Mel spectrogram features.

This is an OPTIONAL experiment (Task C). It uses the same CNN-LSTM architecture
and the SAME actor-independent split as train.py, with log-Mel spectrograms
(200, 64, 1) instead of MFCC (200, 40, 1) as input.

Results are saved separately to:
  models/logmel_cnn_lstm.keras
  models/logmel_label_mapping.json
  outputs/logmel_training_history.csv
  outputs/logmel_test_predictions.csv
  outputs/logmel_test_manifest.csv
  outputs/logmel_learning_curves.png

The original MFCC models are NOT overwritten.

Run: python train_logmel.py

NOTE: The log-Mel model and MFCC model must NOT share checkpoint files.
      Do not use MFCC checkpoints for log-Mel inference (shape: 64 vs 40).
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, RANDOM_SEED
from src.dataset import discover_ravdess
from src.preprocessing import load_audio, extract_log_mel, LOG_MEL_N_MELS
from src.model import build_cnn_lstm

LOGMEL_MODEL   = MODEL_DIR / "logmel_cnn_lstm.keras"
LOGMEL_MAP     = MODEL_DIR / "logmel_label_mapping.json"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)


def load_split() -> tuple[list[int], list[int], list[int]]:
    path = MODEL_DIR / "split_actors.json"
    if not path.exists():
        raise SystemExit("models/split_actors.json not found. Run python train.py first.")
    s = json.loads(path.read_text(encoding="utf-8"))
    return s["train"], s["validation"], s["test"]


def extract_features(frame: pd.DataFrame, encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray, list]:
    X, y, kept = [], [], []
    for row in frame.itertuples(index=False):
        try:
            feat = extract_log_mel(load_audio(row.path))
            X.append(feat)
            y.append(encoder.transform([row.emotion])[0])
            kept.append(row.path)
        except Exception as exc:
            print(f"  Skip {Path(row.path).name}: {exc}")
    if not X:
        raise ValueError("No usable log-Mel features extracted.")
    return np.stack(X), np.asarray(y, dtype=np.int64), kept


def main() -> None:
    set_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_actors, val_actors, test_actors = load_split()
    df = discover_ravdess(DATA_DIR)
    if df.empty:
        raise SystemExit(f"No RAVDESS files found under {DATA_DIR}.")

    labels  = sorted(df.emotion.unique())
    encoder = LabelEncoder().fit(labels)
    print("Emotion labels:", list(encoder.classes_))

    train_df = df[df.actor_id.isin(train_actors)]
    val_df   = df[df.actor_id.isin(val_actors)]
    test_df  = df[df.actor_id.isin(test_actors)]

    print(f"Extracting log-Mel features — train ({len(train_df)} files)…")
    X_train, y_train, _ = extract_features(train_df, encoder)
    print(f"Extracting log-Mel features — val ({len(val_df)} files)…")
    X_val,   y_val,   _ = extract_features(val_df,   encoder)
    print(f"Extracting log-Mel features — test ({len(test_df)} files)…")
    X_test,  y_test, test_paths = extract_features(test_df, encoder)

    print(f"X_train: {X_train.shape}  (expected: 840, {200}, {LOG_MEL_N_MELS}, 1)")
    assert X_train.shape[1:] == (200, LOG_MEL_N_MELS, 1), (
        f"Unexpected shape: {X_train.shape}")

    # Class weights — neutral is underrepresented in RAVDESS
    cw_vals = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw_vals)}
    print("Class weights:", {labels[i]: round(class_weight[i], 2) for i in range(len(labels))})

    model = build_cnn_lstm(tuple(X_train.shape[1:]), len(labels))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            str(LOGMEL_MODEL),
            monitor="val_accuracy", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=16,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    model.save(LOGMEL_MODEL)

    with open(LOGMEL_MAP, "w", encoding="utf-8") as f:
        json.dump({str(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}, f, indent=2)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_DIR / "logmel_training_history.csv", index=False)

    # Test predictions
    test_probs = model.predict(X_test, batch_size=32, verbose=0)
    test_preds = test_probs.argmax(axis=1)
    pd.DataFrame({"path": test_paths, "y_true": y_test,
                  "y_pred": test_preds.tolist()}).to_csv(
        OUTPUT_DIR / "logmel_test_predictions.csv", index=False)
    pd.DataFrame({"path": test_paths, "y_true": y_test}).to_csv(
        OUTPUT_DIR / "logmel_test_manifest.csv", index=False)

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist_df) + 1)
    for ax, col, lbl in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        if col in hist_df: ax.plot(epochs, hist_df[col], label="Train", linewidth=1.8)
        if f"val_{col}" in hist_df:
            ax.plot(epochs, hist_df[f"val_{col}"], label="Val", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(lbl)
        ax.set_title(f"Log-Mel CNN-LSTM {lbl}"); ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("EmotionSense Log-Mel — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "logmel_learning_curves.png", dpi=150)
    plt.close(fig)

    val_acc = hist_df.val_accuracy.max()
    from sklearn.metrics import accuracy_score
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nBest val accuracy: {val_acc:.4f}")
    print(f"Test accuracy (clean): {test_acc:.4f}")
    print(f"Model saved: {LOGMEL_MODEL}")
    print("Run `python evaluate_all.py` to compare with other models.")


if __name__ == "__main__":
    main()
