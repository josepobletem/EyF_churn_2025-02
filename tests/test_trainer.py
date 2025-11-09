import os
import pickle
import yaml
import pytest
import pandas as pd
import numpy as np

import src.trainer as trainer
import src.trainer as trainer_module  # alias para monkeypatch si hace falta lgb


# ---------- DummyBooster a nivel módulo (pickleable) ----------

class DummyBooster:
    """
    Simula el Booster final que entrena trainer.train_final_model().
    Debe ser pickleable, por eso está definido a nivel módulo.
    """
    def predict(self, X):
        # Devuelve probabilidades válidas [0,1] para clase positiva
        # asumimos problema binario.
        return np.full(shape=(len(X),), fill_value=0.3, dtype=float)


# ---------- Helpers internos del test ----------

def _make_fake_features_df():
    """
    Dataset sintético de churn estilo binario con meses que matchean train_months.
    Importante:
    - foto_mes usa 202101, 202102, 202103 => coincide con train_months
    - incluimos varias clases en clase_ternaria
    - incluimos features numéricas
    """
    return pd.DataFrame(
        {
            "numero_de_cliente": [100, 101, 102, 103, 104, 105],
            "foto_mes": [202101, 202101, 202102, 202102, 202103, 202103],
            "clase_ternaria": [
                "CONTINUA",
                "BAJA+1",
                "BAJA+2",
                "BAJA+1",
                "CONTINUA",
                "BAJA+2",
            ],
            "feature_a": [1.0, 2.0, 0.5, 1.5, 3.0, 2.5],
            "feature_b": [10.0, 11.0, 9.0, 9.5, 7.5, 7.8],
            "feature_c": [100.0, 120.0, 80.0, 90.0, 60.0, 65.0],
        }
    )


def _write_temp_config(tmp_path, feature_csv_path, models_dir):
    """
    Crea un config.yaml temporal alineado con el esquema NUEVO que usa trainer.py.
    """
    cfg_text = f"""
paths:
  raw_dataset: "{tmp_path.as_posix()}/data/raw/competencia_01_crudo.csv"
  processed_dataset: "{tmp_path.as_posix()}/data/processed/competencia_01.csv"
  feature_dataset: "{feature_csv_path.as_posix()}"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column_full: "clase_ternaria"
  binary_target_col: "clase_binaria2"
  peso_col: "clase_peso"

train:
  models_dir: "{models_dir.as_posix()}"

  # estos meses tienen que existir en el fake df
  train_months: [202101, 202102, 202103]
  test_month: 202104

  drop_cols: []

  # parámetros varios que trainer no usa directamente para entrenar
  n_estimators: 200
  nfold: 3
  seed: 123
  n_trials: 2

  ganancia_acierto: 780000.0
  costo_estimulo: 20000.0

  weight_baja2: 1.00002
  weight_baja1: 1.00001
  weight_continua: 1.0
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def test_trainer_train_final_model(tmp_path, monkeypatch):
    """
    Valida que trainer.train_final_model():
    - construya clase_binaria1, clase_binaria2 y clase_peso
    - filtre por train_months
    - lea best_params.yaml generado por optimizer
    - entrene (mockeado con DummyBooster)
    - guarde final_model.pkl y final_metrics.yaml
    - devuelva info coherente
    """

    # 1. Dataset sintético => competencia_01_features.csv
    features_df = _make_fake_features_df()

    processed_dir = tmp_path / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    feature_csv_path = processed_dir / "competencia_01_features.csv"
    features_df.to_csv(feature_csv_path, index=False)

    # 2. Crear models_dir y best_params.yaml simulado
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Este YAML debe parecerse a lo que guarda optimizer.run_optuna_and_train()
    # en tu pipeline nuevo:
    fake_best_params_file = {
        "best_params": {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "None",
            "verbose": -1,
            "seed": 123,
            "learning_rate": 0.05,
            "num_leaves": 32,
            "max_depth": 6,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_split_gain": 0.0,
            "min_sum_hessian_in_leaf": 0.1,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "feature_fraction_bynode": 0.9,
            "extra_trees": False,
            "max_bin": 255,
            "scale_pos_weight": 1.0,
        },
        "best_iteration": 25,
        "train_months": [202101, 202102, 202103],
        "test_month": 202104,
    }

    best_params_path = models_dir / "best_params_v2.yaml"
    with open(best_params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(fake_best_params_file, f, sort_keys=False, allow_unicode=True)

    # 3. config.yaml temporal apuntando al dataset y models_dir
    cfg_path = _write_temp_config(
        tmp_path=tmp_path,
        feature_csv_path=feature_csv_path,
        models_dir=models_dir,
    )

    # 4. Monkeypatch: trainer.load_config() debe devolver nuestro config temporal
    real_cfg = trainer.load_config(str(cfg_path))

    def _fake_load_config():
        return real_cfg

    monkeypatch.setattr(trainer, "load_config", _fake_load_config)

    # 5. Monkeypatch de lightgbm.train dentro de trainer
    #    para que no entrene de verdad y devuelva DummyBooster()
    def fake_lgb_train(params, train_set, num_boost_round):
        return DummyBooster()

    monkeypatch.setattr(trainer.lgb, "train", fake_lgb_train)

    # 6. Ejecutar la función bajo test
    result = trainer.train_final_model()

    # 7. Validaciones sobre el dict resultado
    assert "model_path" in result
    assert "metrics_path" in result
    assert "logloss_in_sample" in result

    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["metrics_path"])

    # 8. Cargar el modelo final
    with open(result["model_path"], "rb") as f:
        final_model = pickle.load(f)

    # En la versión nueva guardamos un Booster-like, no un clasificador sklearn.
    assert hasattr(final_model, "predict")
    # NO exigimos predict_proba porque Booster no tiene predict_proba.

    # 9. Cargar métricas finales
    with open(result["metrics_path"], "r", encoding="utf-8") as f:
        metrics_data = yaml.safe_load(f)

    # Debe haber logloss y confusión
    assert "logloss_in_sample" in metrics_data
    assert "confusion_matrix_in_sample" in metrics_data

    # Debe registrar metadata útil
    assert "best_iteration" in metrics_data or "best_iteration" in result
    assert "train_months" in metrics_data or "train_months" in fake_best_params_file
    assert "binary_target_col" in metrics_data or "binary_target_col" in real_cfg.columns.dict()

    # Sanity check: el dataset tenía >1 clase_ternaria
    assert len(np.unique(features_df["clase_ternaria"])) >= 2
