import json
import math
import re
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"

mlflow.set_tracking_uri(
    "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")
)
mlflow.set_experiment("EmotionSense Experiments")


def log_existing_file(path):
    if path.exists():
        mlflow.log_artifact(str(path))


def flatten_metrics(data, prefix=""):
    metrics = {}

    if isinstance(data, dict):
        for key, value in data.items():
            name = f"{prefix}_{key}" if prefix else str(key)
            metrics.update(flatten_metrics(value, name))

    elif isinstance(data, list):
        for index, value in enumerate(data):
            metrics.update(flatten_metrics(value, f"{prefix}_{index}"))

    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        if math.isfinite(float(data)):
            name = re.sub(r"[^a-zA-Z0-9_]", "_", prefix)[:200]
            if name:
                metrics[name] = float(data)

    return metrics


# Register the four existing trained models.
model_runs = [
    {
        "name": "Original CNN-LSTM",
        "files": ["emotionsense_cnn_lstm.keras", "label_mapping.json"],
        "outputs": ["training_history.csv", "test_manifest.csv",
                    "learning_curves.png"],
    },
    {
        "name": "Improved CNN-LSTM",
        "files": ["improved_cnn_lstm.keras", "improved_label_mapping.json"],
        "outputs": ["improved_training_history.csv",
                    "improved_test_predictions.csv",
                    "improved_test_manifest.csv",
                    "improved_learning_curves.png"],
    },
    {
        "name": "Log-Mel CNN-LSTM",
        "files": ["logmel_cnn_lstm.keras", "logmel_label_mapping.json"],
        "outputs": ["logmel_training_history.csv",
                    "logmel_test_predictions.csv",
                    "logmel_test_manifest.csv",
                    "logmel_learning_curves.png"],
    },
    {
        "name": "SVM Baseline",
        "files": ["svm_model.pkl", "svm_scaler.pkl",
                  "svm_label_mapping.json", "svm_config.json"],
        "outputs": ["svm_test_predictions.csv"],
    },
]

for model in model_runs:
    with mlflow.start_run(run_name=model["name"]):
        mlflow.set_tag("project", "EmotionSense")
        mlflow.set_tag("model", model["name"])
        mlflow.set_tag("stage", "existing_model_registration")

        for filename in model["files"]:
            log_existing_file(MODELS / filename)

        for filename in model["outputs"]:
            log_existing_file(OUTPUTS / filename)

        print(f"Registered: {model['name']}")


# Register the complete evaluation and noise-robustness results.
with mlflow.start_run(run_name="Complete Evaluation Results"):
    mlflow.set_tag("project", "EmotionSense")
    mlflow.set_tag("stage", "evaluation_and_noise_analysis")

    for path in OUTPUTS.rglob("*"):
        if path.is_file():
            log_existing_file(path)

    for filename in [
        "all_models_comparison.json",
        "extended_noise_comparison.json",
        "noise_evaluation_metrics.json",
    ]:
        path = OUTPUTS / filename
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metrics = flatten_metrics(data)
                if metrics:
                    mlflow.log_metrics(metrics)
            except (json.JSONDecodeError, OSError, ValueError) as error:
                print(f"Could not extract metrics from {filename}: {error}")

    print("Registered complete evaluation results.")

print("\nMLflow registration completed.")
print("Open the MLflow UI to view your experiments.")
