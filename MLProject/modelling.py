from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XDG_CACHE_HOME", str(BASE_DIR / ".cache"))

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = BASE_DIR / "breast_cancer_preprocessing" / "breast_cancer_preprocessed.csv"
TRACKING_DIR = BASE_DIR / "mlruns"


def load_dataset(data_path: Path = DATA_PATH):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)


def main():
    mlflow.set_tracking_uri(TRACKING_DIR.as_uri())
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment("breast_cancer_basic_autolog")
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True,
        log_post_training_metrics=False,
    )

    X_train, X_test, y_train, y_test = load_dataset()
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name="random_forest_basic"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        print("Training selesai. Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        print(f"MLflow tracking URI: {TRACKING_DIR.as_uri()}")


if __name__ == "__main__":
    main()
