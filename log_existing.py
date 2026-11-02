from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parent

mlflow.set_tracking_uri(
    "sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/")
)
mlflow.set_experiment("EmotionSense Experiments")

models_dir = ROOT / "models"
outputs_dir = ROOT / "outputs"
reports_dir = ROOT / "reports"

with mlflow.start_run(run_name="Existing Models and Evaluation Results"):
    mlflow.set_tag("project", "EmotionSense")
    mlflow.set_tag("stage", "MLOps Integration")
    mlflow.set_tag("dataset", "RAVDESS Audio Speech")

    model_files = [
        f for f in models_dir.iterdir()
        if f.is_file() and f.name != "README.md"
    ]

    mlflow.log_param("model_files_count", len(model_files))
    mlflow.log_param("evaluation_output_files", len(list(outputs_dir.rglob("*"))))
    mlflow.log_param("framework", "TensorFlow / Scikit-learn")

    if model_files:
        mlflow.log_artifacts(str(models_dir), artifact_path="models")

    if outputs_dir.exists():
        mlflow.log_artifacts(str(outputs_dir), artifact_path="evaluation_outputs")

    if reports_dir.exists():
        mlflow.log_artifacts(str(reports_dir), artifact_path="reports")

    print("Existing project artifacts registered successfully.")
    print("Models registered:", len(model_files))
    print("MLflow run ID:", mlflow.active_run().info.run_id)
