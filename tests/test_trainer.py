import os
import pickle
import yaml
import pytest
import pandas as pd
import numpy as np

import src.trainer as trainer


def _make_fake_features_df():
    """
    Dataset sintético multiclase parecido a churn.
    """
    return pd.DataFrame(
        {
            "numero_de_cliente": [100, 100, 200, 200],
            "foto_mes": [202401, 202402, 202401, 202402],
            "clase_ternaria": ["CONTINUA", "BAJA+1", "BAJA+2", "CONTINUA"],
            "feature_a": [1.0, 2.0, 0.5, 1.5],
            "feature_b": [10.0, 11.0, 9.0, 9.5],
            "feature_c": [100.0, 120.0, 80.0, 90.0],
        }
    )


def test_trainer_train_final_model(tmp_path, monkeypatch):
    """
    Valida que trainer.train_final_model():
    - cargue el dataset de features completo
    - lea best_params.yaml
    - entrene un modelo final
    - guarde final_model.pkl y final_metrics.yaml
    """

    # 1. Dataset sintético => competencia_01_features.csv
    features_df = _make_fake_features_df()

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_csv_path = processed_dir / "competencia_01_features.csv"
    features_df.to_csv(feature_csv_path, index=False)

    # 2. Crear models_dir y best_params.yaml
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    fake_best_params = {
        "best_params": {
            "n_estimators": 50,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": 5,
            "min_child_samples": 10,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
        },
        "best_score_logloss": 0.5,
        "target_column": "clase_ternaria",
    }

    best_params_path = models_dir / "best_params.yaml"
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fake_best_params, f, sort_keys=False, allow_unicode=True)

    # 3. config.yaml temporal apuntando a este dataset y este models_dir
    cfg_text = f"""
paths:
  raw_dataset: "{tmp_path.as_posix()}/data/raw/competencia_01_crudo.csv"
  processed_dataset: "{tmp_path.as_posix()}/data/processed/competencia_01.csv"
  feature_dataset: "{feature_csv_path.as_posix()}"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column: "clase_ternaria"

train:
  models_dir: "{models_dir.as_posix()}"
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")

    # 4. Cargar config real una vez y monkeypatch para que la devuelva fija
    real_cfg = trainer.load_config(str(cfg_path))

    def _fake_load_config():
        return real_cfg

    monkeypatch.setattr(trainer, "load_config", _fake_load_config)


    # 5. Ejecutar la función bajo test
    result = trainer.train_final_model()

    # 6. Validaciones sobre el dict resultado
    assert "model_path" in result
    assert "metrics_path" in result
    assert "logloss_in_sample" in result

    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["metrics_path"])

    # 7. Cargar el modelo final
    with open(result["model_path"], "rb") as f:
        final_model = pickle.load(f)

    assert hasattr(final_model, "predict")
    assert hasattr(final_model, "predict_proba")

    # 8. Cargar métricas
    with open(result["metrics_path"], "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)

    assert "logloss_in_sample" in metrics_data
    assert "confusion_matrix_in_sample" in metrics_data
    assert "classes" in metrics_data

    # Sanity check: debe haber más de una clase
    assert len(np.unique(features_df["clase_ternaria"])) >= 2
    assert len(metrics_data["classes"]) >= 2
