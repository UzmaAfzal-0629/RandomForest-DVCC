
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def run_training():
    # Load configuration
    with open("params.yaml", "r") as file:
        params = yaml.safe_load(file)

    model_params = params["model"]
    data_params = params["data"]

    # Load dataset
    df = pd.read_csv("data/dataset.csv")

    if "target" not in df.columns:
        raise ValueError("Dataset must contain a 'target' column.")

    X = df.drop(columns=["target"])
    y = df["target"]

    # Use numeric features
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("Dataset has no numeric feature columns.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_params["test_size"],
        random_state=data_params["random_state"],
        stratify=y,
    )

    # Build model pipeline
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            min_samples_split=model_params["min_samples_split"],
            min_samples_leaf=model_params["min_samples_leaf"],
            random_state=model_params["random_state"],
            class_weight=model_params["class_weight"],
        )),
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluate
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "recall": float(recall_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "f1_score": float(f1_score(
            y_test, predictions, average="weighted", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "features": int(X.shape[1]),
    }

    # Save model and metrics
    Path("models").mkdir(exist_ok=True)

    joblib.dump(pipeline, "models/random_forest.joblib")

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("\nRandom Forest Evaluation")
    print("------------------------")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    run_training()
