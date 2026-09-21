import yaml
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load hyperparameters
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

rf_params = params["model"]


# Load dataset
df = pd.read_csv("data/dataset.csv")

X = df.drop("target", axis=1)
y = df["target"]


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=rf_params["n_estimators"],
    max_depth=rf_params["max_depth"],
    min_samples_split=rf_params["min_samples_split"],
    min_samples_leaf=rf_params["min_samples_leaf"],
    random_state=rf_params["random_state"]
)


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Calculate metrics
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="weighted"),
    "recall": recall_score(y_test, y_pred, average="weighted"),
    "f1_score": f1_score(y_test, y_pred, average="weighted")
}


# Save metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


# Display metrics
print("\nRandom Forest Evaluation Metrics")
print("--------------------------------")
print(f"Accuracy : {metrics['accuracy']:.5f}")
print(f"Precision: {metrics['precision']:.5f}")
print(f"Recall   : {metrics['recall']:.5f}")
print(f"F1-score : {metrics['f1_score']:.5f}")


