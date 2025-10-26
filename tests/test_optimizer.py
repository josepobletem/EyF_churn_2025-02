import os
import pickle
import yaml
import pytest
import pandas as pd

import src.optimizer as optimizer

def _make_fake_features_df():
    """
    Dataset sintético multiclase para probar optimizer.
    Aseguramos >=2 ejemplos por clase para que stratify funcione.
    """
    return pd.DataFrame(
        {
            "numero_de_cliente": [100, 100, 200, 200, 300, 300],
            "foto_mes": [202401, 202402, 202401, 202402, 202401, 202402],
            "clase_ternaria": [
                "CONTINUA", "CONTINUA",
                "BAJA+1", "BAJA+1",
                "BAJA+2", "BAJA+2",
            ],
            "feature_a": [1.0, 2.0, 0.5, 1.5, 3.0, 2.5],
            "feature_b": [10.0, 11.0, 9.0, 9.5, 7.5, 7.8],
            "feature_c": [100.0, 120.0, 80.0, 90.0, 60.0, 65.0],
        }
    )


def _write_temp_config(tmp_path, feature_csv_path, models_dir):
    """
    Crea un config.yaml temporal apuntando al dataset de features
    y al directorio models.
    """
    cfg_text = f"""
paths:
  raw_dataset: "{tmp_path.as_posix()}/data/raw/competencia_01_crudo.csv"
  processed_dataset: "{tmp_path.as_posix()}/data/processed/competencia_01.csv"
  feature_dataset: "{feature_csv_path.as_posix()}"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column: "clase_ternaria"

features:
  base_table_name: "base_clientes"
  steps:
    - "sql/01_base_tables.sql"
    - "sql/02_feat_numeric.sql"
    - "sql/03_final_model.sql"

train:
  test_size: 0.5
  random_state: 123
  n_trials: 5
  models_dir: "{models_dir.as_posix()}"
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def test_optimizer_run_hyperparam_search(tmp_path, monkeypatch):
    """
    Valida que optimizer.run_hyperparam_search():
    - cargue el dataset de features
    - haga la búsqueda simulada
    - guarde best_params.yaml y best_model.pkl
    - devuelva info coherente
    """

    # 1. Creamos dataset sintético de features y lo guardamos
    features_df = _make_fake_features_df()

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_csv_path = processed_dir / "competencia_01_features.csv"
    features_df.to_csv(feature_csv_path, index=False)

    # 2. Creamos directorio models/
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # 3. Creamos config.yaml temporal
    cfg_path = _write_temp_config(
        tmp_path=tmp_path,
        feature_csv_path=feature_csv_path,
        models_dir=models_dir,
    )

    # 4. Cargar config real una vez y monkeypatch para que la reutilice
    real_cfg = optimizer.load_config(str(cfg_path))

    def _fake_load_config():
        return real_cfg

    monkeypatch.setattr(optimizer, "load_config", _fake_load_config)


    # 5. Monkeypatch de optuna para NO correr búsqueda real
    class DummyStudy:
        def __init__(self):
            self.best_params = {
                "n_estimators": 123,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "max_depth": 6,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
            }
            self.best_value = 0.42  # logloss simulado

        def optimize(self, objective, n_trials):
            # No ejecutamos objective() para acelerar tests.
            return

    def fake_create_study(direction):
        return DummyStudy()

    monkeypatch.setattr(optimizer.optuna, "create_study", fake_create_study)

    # 6. Ejecutar la función bajo test
    result = optimizer.run_hyperparam_search()

    # 7. Validaciones de salida
    assert "best_params" in result
    assert "best_score" in result
    assert "model_path" in result
    assert "params_path" in result

    # Debe haber guardado los archivos en disco
    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["params_path"])

    # Cargar YAML de hiperparámetros
    with open(result["params_path"], "r", encoding="utf-8") as f:
        saved_params = yaml.safe_load(f)

    assert "best_params" in saved_params
    assert "best_score_logloss" in saved_params
    assert saved_params["target_column"] == "clase_ternaria"

    # Cargar el modelo pickled
    with open(result["model_path"], "rb") as f:
        model_obj = pickle.load(f)

    # Debe comportarse como un clasificador sklearn-ish
    assert hasattr(model_obj, "predict_proba")
    assert hasattr(model_obj, "fit")
    assert hasattr(model_obj, "predict")
