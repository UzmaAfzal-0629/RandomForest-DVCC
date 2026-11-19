
# Breast Cancer Classification using Random Forest

## Project Overview

This project implements a Random Forest Classification model
using the Breast Cancer Wisconsin dataset.

The project demonstrates a basic MLOps workflow using Python,
Scikit-learn, DVC and GitHub Actions.

## Objectives

- Load and understand the dataset.
- Handle missing values.
- Train a Random Forest classifier.
- Evaluate model performance.
- Track model parameters and pipeline using DVC.
- Automate testing using GitHub Actions.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- DVC
- Git and GitHub
- Pytest
- GitHub Actions

## Project Structure

```text
RandomForest-DVCC/
│
├── data/
│   └── dataset.csv
│
├── models/
│
├── tests/
│   └── test_train.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── train.py
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── metrics.json
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/UzmaAfzal-0629/RandomForest-DVCC.git
cd RandomForest-DVCC
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Model

```bash
python train.py
```

The trained model is saved in:

```text
models/random_forest.joblib
```

Evaluation results are saved in:

```text
metrics.json
```

## Run Tests

```bash
pytest -v
```

## Run with DVC

```bash
dvc repro
```

Check the metrics:

```bash
dvc metrics show
```

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## Conclusion

This project demonstrates the development and evaluation
of a Random Forest classification model with a simple,
reproducible MLOps workflow.
