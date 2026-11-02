"""train_improved.py — Improved CNN-LSTM for EmotionSense.

Improvements over the original train.py:
  1. MFCC + delta + delta-delta features  (200, 120, 1) instead of (200, 40, 1)
  2. Online noise augmentation on training batches  (random SNR 5–30 dB, p=0.5)
  3. Class weighting for imbalanced neutral class
  4. Increased patience (15) and lower initial LR (5e-4) + cosine decay
  5. Saves artifacts to models/improved_* so the original baseline is preserved

Uses the SAME actor split from models/split_actors.json.
Run:   python train_improved.py
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
from src.preprocessing import load_audio, extract_mfcc_with_deltas
from src.model import build_improved_cnn_bilstm
from src.noise import add_white_noise_at_snr, add_pink_noise_at_snr

# ── Model save paths (separate from original baseline) ────────────────────────
IMPROVED_MODEL  = MODEL_DIR / "improved_cnn_lstm.keras"
IMPROVED_MAP    = MODEL_DIR / "improved_label_mapping.json"


def set_seeds(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split() -> tuple[list[int], list[int], list[int]]:
    split_path = MODEL_DIR / "split_actors.json"
    if not split_path.exists():
        raise SystemExit("models/split_actors.json missing. Run `python train.py` first.")
    s = json.loads(split_path.read_text(encoding="utf-8"))
    return s["train"], s["validation"], s["test"]


def apply_spec_augment(feat: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply time and frequency masking to (max_frames, n_features, 1) tensor."""
    f = feat.copy()
    frames, n_feats, _ = f.shape
    # Time masking: zero out 1-2 random time strips
    for _ in range(rng.integers(1, 3)):
        w = rng.integers(6, 16)
        if frames > w:
            t = rng.integers(0, frames - w)
            f[t:t+w, :, :] = 0.0
    # Frequency masking: zero out 1-2 random coefficient strips
    for _ in range(rng.integers(1, 3)):
        w = rng.integers(6, 16)
        if n_feats > w:
            freq = rng.integers(0, n_feats - w)
            f[:, freq:freq+w, :] = 0.0
    return f


def load_clean_features(frame: pd.DataFrame, encoder: LabelEncoder) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract clean delta-MFCC features for validation or test sets without augmentation."""
    X, y, kept = [], [], []
    for row in frame.itertuples(index=False):
        try:
            audio = load_audio(row.path)
            feat = extract_mfcc_with_deltas(audio)
            X.append(feat)
            y.append(encoder.transform([row.emotion])[0])
            kept.append(row.path)
        except Exception as exc:
            print(f"  Skipping {Path(row.path).name}: {exc}")
    if not X:
        raise ValueError("No usable features extracted.")
    return np.stack(X), np.asarray(y, dtype=np.int64), kept


def load_augmented_train_features(
    frame: pd.DataFrame, encoder: LabelEncoder, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract training features with 4-fold dataset expansion:
    
    1. 100% clean recordings
    2. Additive White Gaussian Noise (AWGN at SNR 12–26 dB)
    3. Additive Pink Noise (1/f spectral attenuation at SNR 12–26 dB)
    4. SpecAugment masked copies (time and frequency strips)
    This quadruples training diversity without corrupting clean samples.
    """
    X, y, kept = [], [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        try:
            audio = load_audio(row.path)
            clean_feat = extract_mfcc_with_deltas(audio)
            label_idx = encoder.transform([row.emotion])[0]

            # 1. Clean feature
            X.append(clean_feat)
            y.append(label_idx)
            kept.append(row.path)

            # 2. Additive White Gaussian Noise
            snr_w = float(rng.uniform(12.0, 26.0))
            noisy_w = add_white_noise_at_snr(audio, snr_db=snr_w, seed=RANDOM_SEED + i * 3)
            X.append(extract_mfcc_with_deltas(noisy_w))
            y.append(label_idx)
            kept.append(row.path)

            # 3. Additive Pink Noise
            snr_p = float(rng.uniform(12.0, 26.0))
            noisy_p = add_pink_noise_at_snr(audio, snr_db=snr_p, seed=RANDOM_SEED + i * 5)
            X.append(extract_mfcc_with_deltas(noisy_p))
            y.append(label_idx)
            kept.append(row.path)

            # 4. SpecAugment masked feature
            masked_feat = apply_spec_augment(clean_feat, rng)
            X.append(masked_feat)
            y.append(label_idx)
            kept.append(row.path)

        except Exception as exc:
            print(f"  Skipping {Path(row.path).name}: {exc}")

    if not X:
        raise ValueError("No usable features extracted.")
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
    num_classes = len(labels)
    print("Emotion labels:", list(encoder.classes_))
    print(f"Train: {len(train_actors)} actors | Val: {len(val_actors)} | Test: {len(test_actors)}")

    train_df = df[df.actor_id.isin(train_actors)]
    val_df   = df[df.actor_id.isin(val_actors)]
    test_df  = df[df.actor_id.isin(test_actors)]

    rng = np.random.default_rng(RANDOM_SEED)

    print("\nExtracting training features with 4-fold dataset expansion (Clean + AWGN + Pink + SpecAugment)…")
    X_train, y_train, _ = load_augmented_train_features(train_df, encoder, rng)
    print("Extracting validation features (strictly clean, unaugmented)…")
    X_val,   y_val,   _ = load_clean_features(val_df, encoder)
    print("Extracting test features (strictly clean, held-out)…")
    X_test,  y_test, test_paths = load_clean_features(test_df, encoder)

    print(f"\nX_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    # One-hot encoding for label smoothing
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat   = tf.keras.utils.to_categorical(y_val, num_classes)

    # Shuffle training set
    perm = rng.permutation(len(X_train))
    X_train = X_train[perm]
    y_train_cat = y_train_cat[perm]

    # ── Class weights (handles neutral having fewer samples) ───────────────
    cw_vals = compute_class_weight("balanced", classes=np.arange(len(labels)), y=y_train)
    class_weight = {i: float(w) for i, w in enumerate(cw_vals)}
    print("Class weights:", {labels[i]: round(class_weight[i], 2) for i in range(len(labels))})

    # ── Model ──────────────────────────────────────────────────────────────
    input_shape = tuple(X_train.shape[1:])  # (200, 120, 1)
    model = build_improved_cnn_bilstm(input_shape, num_classes, learning_rate=4e-4, label_smoothing=0.08)
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=16, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(IMPROVED_MODEL),
            monitor="val_accuracy", save_best_only=True, verbose=1
        ),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=70,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # Save final best model & mapping
    model.save(IMPROVED_MODEL)
    with open(IMPROVED_MAP, "w", encoding="utf-8") as f:
        json.dump({str(i): str(lbl) for i, lbl in enumerate(encoder.classes_)}, f, indent=2)

    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(OUTPUT_DIR / "improved_training_history.csv", index=False)

    # Save test predictions for evaluation pipeline
    test_probs = model.predict(X_test, batch_size=32, verbose=0)
    test_preds = test_probs.argmax(axis=1)
    pd.DataFrame({
        "path":   test_paths,
        "y_true": y_test.tolist(),
        "y_pred": test_preds.tolist(),
    }).to_csv(OUTPUT_DIR / "improved_test_predictions.csv", index=False)
    pd.DataFrame({"path": test_paths, "y_true": y_test}).to_csv(
        OUTPUT_DIR / "improved_test_manifest.csv", index=False
    )

    # ── Learning curves ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist_df) + 1)
    for ax, col, label in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        if col in hist_df.columns:
            ax.plot(epochs, hist_df[col], label="Train", linewidth=1.8)
        if f"val_{col}" in hist_df.columns:
            ax.plot(epochs, hist_df[f"val_{col}"], label="Val", linewidth=1.8, linestyle="--")
        ax.set_xlabel("Epoch"); ax.set_ylabel(label)
        ax.set_title(f"Improved CNN-BiLSTM {label}")
        ax.legend(); ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("EmotionSense Improved CNN-BiLSTM with Attention — Learning Curves", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "improved_learning_curves.png", dpi=150)
    plt.close(fig)

    val_acc = hist_df.val_accuracy.max()
    print(f"\nBest val accuracy: {val_acc:.4f}")
    print(f"Model saved to {IMPROVED_MODEL}")
    print("Run `python evaluate_all.py` to compare all models.")


if __name__ == "__main__":
    main()

