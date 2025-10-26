import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from .evaluate import evaluate_predictions

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(cfg: dict):
    df = pd.read_csv(cfg["paths"]["processed_dataset_path"])

    target_col = cfg["dataset"]["target_column"]
    id_col = cfg["dataset"]["id_column"]

    X = df.drop([target_col, id_col], axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg["split"]["test_size"],
        random_state=cfg["split"]["random_state"],
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def save_model(model, cfg: dict):
    out_path = cfg["paths"]["model_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(model, out_path)
    return out_path

def train_and_eval():
    cfg = load_config()
    X_train, X_test, y_train, y_test = load_dataset(cfg)

    model = train_model(X_train, y_train)
    model_path = save_model(model, cfg)

    y_pred_proba = model.predict_proba(X_test)[:,1]
    metrics = evaluate_predictions(y_test, y_pred_proba)

    metrics_path = cfg["paths"]["metrics_path"]
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Modelo guardado en:", model_path)
    print("Métricas:", metrics)

if __name__ == "__main__":
    train_and_eval()
