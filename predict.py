from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Union
import numpy as np
import tensorflow as tf

from config import MODEL_DIR, SAMPLE_RATE
from src.preprocessing import (
    load_audio_from_buffer,
    extract_mfcc,
    extract_mfcc_with_deltas,
    extract_svm_features,
    audio_quality,
)

# In-memory model cache to prevent reloading large weights per inference
_CACHED_MODELS: dict[str, tuple[object, list[str]]] = {}


def get_model(model_name: str = "improved") -> tuple[object, list[str]]:
    """Retrieve or load a cached model and its class label mapping.

    Supported options:
      - 'improved': Improved CNN-BiLSTM with Attention (MFCC + Deltas)
      - 'original': Original CNN-LSTM (MFCC-40)
      - 'svm': SVM Baseline (MFCC pooled stats)
    """
    model_name = (model_name or "improved").lower()
    if model_name in _CACHED_MODELS:
        return _CACHED_MODELS[model_name]

    if model_name == "improved":
        m_path = MODEL_DIR / "improved_cnn_lstm.keras"
        map_path = MODEL_DIR / "improved_label_mapping.json"
        if not m_path.exists():
            return get_model("original")
        model = tf.keras.models.load_model(m_path)
        labels = [json.loads(map_path.read_text(encoding="utf-8"))[str(i)] for i in range(8)]
        _CACHED_MODELS["improved"] = (model, labels)
        return model, labels

    elif model_name == "svm":
        import pickle
        m_path = MODEL_DIR / "svm_model.pkl"
        scaler_path = MODEL_DIR / "svm_scaler.pkl"
        map_path = MODEL_DIR / "svm_label_mapping.json"
        with open(m_path, "rb") as f:
            svm_model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        labels = [json.loads(map_path.read_text(encoding="utf-8"))[str(i)] for i in range(8)]
        _CACHED_MODELS["svm"] = ((svm_model, scaler), labels)
        return (svm_model, scaler), labels

    else:  # 'original'
        m_path = MODEL_DIR / "emotionsense_cnn_lstm.keras"
        map_path = MODEL_DIR / "label_mapping.json"
        if not m_path.exists():
            raise FileNotFoundError("Original CNN-LSTM model not found.")
        model = tf.keras.models.load_model(m_path)
        labels = [json.loads(map_path.read_text(encoding="utf-8"))[str(i)] for i in range(8)]
        _CACHED_MODELS["original"] = (model, labels)
        return model, labels


def compute_reliability(confidence: float, margin: float, aq: dict) -> dict:
    """Assess reliability of the prediction based on confidence, margin, and audio quality."""
    is_uncertain = False
    warning_msg = None

    if aq.get("silence_like", False):
        status = "Low (Silent)"
        is_uncertain = True
        warning_msg = "Recording appears silent or extremely quiet. Please speak closer to the microphone."
    elif aq.get("clipping_fraction", 0.0) > 0.03:
        status = "Moderate (Clipped)"
        is_uncertain = True
        warning_msg = "Audio signal clipping detected. Lower your microphone input gain for better accuracy."
    elif confidence >= 0.50 and margin >= 0.20:
        status = "High Reliability"
    elif confidence >= 0.35 and margin >= 0.10:
        status = "Moderate Reliability"
    else:
        status = "Uncertain / Low Reliability"
        is_uncertain = True
        warning_msg = "Prediction uncertain — top predictions are close. Try recording in a quieter environment or speaking with clearer vocal projection."

    return {
        "status": status,
        "margin": round(float(margin), 3),
        "is_uncertain": is_uncertain,
        "warning_message": warning_msg,
    }


def predict_audio(
    audio_source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    model_name: str = "improved",
    sample_rate: int = SAMPLE_RATE,
) -> dict:
    """Predict emotion from audio path, bytes buffer, or existing float32 array.

    Returns structured results including:
      - predicted_emotion: str
      - softmax_confidence: float
      - top_3: list[dict]
      - class_probabilities: list[dict]
      - reliability: dict
      - audio_quality: dict
      - model_used: str
    """
    if isinstance(audio_source, np.ndarray):
        audio = audio_source.astype(np.float32).reshape(-1)
    else:
        audio = load_audio_from_buffer(audio_source, sample_rate=sample_rate)

    aq = audio_quality(audio)
    model_obj, labels = get_model(model_name)

    if model_name == "svm":
        svm, scaler = model_obj
        feat = extract_svm_features(audio, sample_rate=sample_rate).reshape(1, -1)
        feat_norm = scaler.transform(feat)
        probs = svm.predict_proba(feat_norm)[0]
    elif model_name == "improved":
        feat = extract_mfcc_with_deltas(audio, sample_rate=sample_rate)
        probs = model_obj.predict(np.expand_dims(feat, axis=0), verbose=0)[0]
    else:  # original
        feat = extract_mfcc(audio, sample_rate=sample_rate)
        probs = model_obj.predict(np.expand_dims(feat, axis=0), verbose=0)[0]

    ranked = sorted(
        [{"emotion": labels[i], "confidence": float(probs[i])} for i in range(len(labels))],
        key=lambda item: item["confidence"],
        reverse=True,
    )

    top_conf = ranked[0]["confidence"]
    second_conf = ranked[1]["confidence"] if len(ranked) > 1 else 0.0
    margin = top_conf - second_conf

    reliability = compute_reliability(top_conf, margin, aq)

    return {
        "predicted_emotion": ranked[0]["emotion"],
        "softmax_confidence": top_conf,
        "top_3": ranked[:3],
        "class_probabilities": ranked,
        "reliability": reliability,
        "audio_quality": aq,
        "model_used": model_name,
    }


def predict_file(audio_path: Union[str, Path], model_name: str = "original") -> dict:
    """Backwards-compatible wrapper matching original predict_file signature."""
    return predict_audio(audio_path, model_name=model_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict emotion from an audio file.")
    parser.add_argument("audio_path")
    parser.add_argument("--model", default="improved", choices=["original", "improved", "svm"])
    args = parser.parse_args()
    print(json.dumps(predict_audio(args.audio_path, model_name=args.model), indent=2))
