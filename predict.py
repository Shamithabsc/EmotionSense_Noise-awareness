from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

from config import MODEL_DIR
from src.preprocessing import load_audio, extract_mfcc, audio_quality


def predict_file(audio_path):
    model_path = MODEL_DIR / "emotionsense_cnn_lstm.keras"
    mapping_path = MODEL_DIR / "label_mapping.json"
    if not model_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("Trained model not found. Run `python train.py` first.")
    model = tf.keras.models.load_model(model_path)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    labels = [mapping[str(i)] for i in range(len(mapping))]
    audio = load_audio(audio_path)
    features = extract_mfcc(audio)
    probabilities = model.predict(np.expand_dims(features, axis=0), verbose=0)[0]
    ranked = sorted(
        [{"emotion": labels[i], "confidence": float(probabilities[i])}
         for i in range(len(labels))],
        key=lambda item: item["confidence"], reverse=True
    )
    return {
        "predicted_emotion": ranked[0]["emotion"],
        "softmax_confidence": ranked[0]["confidence"],
        "class_probabilities": ranked,
        "audio_quality": audio_quality(audio),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file.")
    parser.add_argument("audio_path")
    args = parser.parse_args()
    print(json.dumps(predict_file(args.audio_path), indent=2))
