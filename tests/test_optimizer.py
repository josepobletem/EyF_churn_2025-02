import os
import pickle
import yaml
import pytest
import pandas as pd
import numpy as np
import src.optimizer as optimizer


# ---------- Dummies globales (para que pickle no rompa) ----------

class DummyBooster:
    """
    Simula el objeto que devuelve lightgbm.train().
    Debe ser pickleable (definida a nivel módulo).
    """
    def predict(self, X):
        # Devuelve probabilidades válidas [0,1] con misma cantidad de filas que X
        # En binary LGBM esto normalmente es prob de la clase positiva.
        return np.full(shape=(len(X),), fill_value=0.2, dtype=float)


class DummyStudy:
    """
    Simula el objeto Study de Optuna.
    Tiene .best_params y .best_value,
    y una .optimize() que no hace nada.
    """
    def __init__(self):
        # Hiperparámetros tipo LightGBM binary que matchean tu pipeline real
        self.best_params = {
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
        }
        # simulamos la métrica de negocio gan_eval
        self.best_value = 123456789.0

    def optimize(self, objective, n_trials):
        # no corremos nada en tests
        return


# ---------- Helpers internos del test ----------

def _make_fake_features_df():
    """
    Dataset sintético compatible con el pipeline actual.
    Incluye:
    - numero_de_cliente, foto_mes, clase_ternaria
    - columnas numéricas
    - meses para train_months/test_month
    """
    return pd.DataFrame(
        {
            "numero_de_cliente": [100, 101, 102, 103, 104, 105],
            "foto_mes": [202101, 202101, 202102, 202102, 202103, 202104],
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
    Crea un config.yaml temporal alineado con el esquema NUEVO del pipeline.
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

  train_months: [202101, 202102, 202103]
  test_month: 202104

  drop_cols: []

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


# ---------- El test principal ----------

def test_optimizer_run_hyperparam_search(tmp_path, monkeypatch):
    """
    Valida que run_optuna_and_train():
    - lea config
    - construya el dataset de entrenamiento a partir de train_months
    - "ejecute" una pseudo-búsqueda y entrenamiento final (mockeado)
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

    # 4. Monkeypatch: cargar esa config en vez de leer la real del repo
    real_cfg = optimizer.load_config(str(cfg_path))

    def _fake_load_config():
        return real_cfg

    monkeypatch.setattr(optimizer, "load_config", _fake_load_config)

    # 5. Monkeypatch de optuna.create_study
    #    Acepta cualquier firma para no romper si optimizer pasa study_name, direction, etc.
    def fake_create_study(*args, **kwargs):
        return DummyStudy()

    monkeypatch.setattr(optimizer.optuna, "create_study", fake_create_study)

    # 6. Monkeypatch de lightgbm.cv y lightgbm.train
    #    - cv() devuelve una curva sintética de gan_eval
    #    - train() devuelve DummyBooster (pickleable y con predict())
    def fake_cv(params,
                train_set,
                num_boost_round,
                feval,
                stratified,
                nfold,
                seed,
                callbacks):
        # Simulamos evolución de la métrica gan_eval
        # clave: tu código busca la serie con mayor longitud y luego toma max()
        return {
            "gan_eval": [10.0, 20.0, 30.0, 42.0]  # mejor = 42.0 al final
        }

    def fake_train(params, train_set, num_boost_round):
        return DummyBooster()

    monkeypatch.setattr(optimizer.lgb, "cv", fake_cv)
    monkeypatch.setattr(optimizer.lgb, "train", fake_train)

    # 7. Ejecutar tu orquestador real
    #    Asegurate de que optimizer.py tenga esta función pública.
    result = optimizer.run_optuna_and_train()

    # 8. Validaciones de salida básicas
    assert "model_path" in result
    assert "params_path" in result
    assert "best_iteration" in result

    # Puede venir "best_params" y/o "best_value" dependiendo de tu implementación
    assert os.path.exists(result["model_path"])
    assert os.path.exists(result["params_path"])

    # 9. Validar YAML escrito
    with open(result["params_path"], "r", encoding="utf-8") as f:
        saved_yaml = yaml.safe_load(f)

    # cosas que esperamos en tu pipeline actual
    assert "train_months" in saved_yaml
    assert "test_month" in saved_yaml
    assert "best_iteration" in saved_yaml
    # puede ser "lgbm_params" o "best_params" según cómo lo guardes:
    assert ("lgbm_params" in saved_yaml) or ("best_params" in saved_yaml)

    # 10. Cargar el modelo pickled y ver que se pueda usar
    with open(result["model_path"], "rb") as f:
        model_obj = pickle.load(f)

    assert hasattr(model_obj, "predict")
    # no exigimos .fit() porque ahora guardamos Booster, no sklearn
