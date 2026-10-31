import json
from pathlib import Path
import mlflow

ROOT = Path(__file__).resolve().parent
TRACKING_DIR = ROOT / "mlruns"

mlflow.set_tracking_uri("sqlite:///" + str(ROOT / "mlflow.db").replace("\\", "/"))
mlflow.set_experiment("EmotionSense Experiments")


def log_experiment(model_name, params=None, metrics=None, artifacts=None):
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("project", "EmotionSense")
        mlflow.set_tag("model", model_name)

        if params:
            mlflow.log_params(params)

        if metrics:
            mlflow.log_metrics({
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
            })

        if artifacts:
            for artifact in artifacts:
                path = Path(artifact)
                if path.exists():
                    mlflow.log_artifact(str(path))

        print("MLflow run logged:", mlflow.active_run().info.run_id)


if __name__ == "__main__":
    log_experiment(
        model_name="MLflow Setup Test",
        params={"project": "EmotionSense", "tracking": "local"},
        metrics={"setup_status": 1.0},
    )

