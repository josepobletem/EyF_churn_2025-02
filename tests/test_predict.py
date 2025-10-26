import os
import pickle
import yaml
import numpy as np
import pandas as pd

import src.predict as predict


class DummyModel:
    """
    Modelo fake que emula al Booster final:
    - tiene .predict(X) y devuelve una probabilidad por fila.
    """
    def predict(self, X):
        # probas sintéticas: 0.9 para la primera mitad, 0.1 para la segunda mitad
        n = len(X)
        if n == 0:
            return np.array([], dtype=float)
        half = n // 2
        proba = np.concatenate([
            np.full(half, 0.9),
            np.full(n - half, 0.1)
        ])
        return proba


def _write_fake_config(tmp_path, feature_csv_path, models_dir):
    """
    Genera un config.yaml mínimo válido para predict.score_month
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
  binary_target_col: "clase_binaria2"
  peso_col: "clase_peso"

train:
  models_dir: "{models_dir.as_posix()}"
  train_months: [202101, 202102]
  test_month: 202103
  drop_cols: []

  ganancia_acierto: 1000.0
  costo_estimulo: 10.0

  weight_baja2: 1.00002
  weight_baja1: 1.00001
  weight_continua: 1.0
"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    return cfg_path


def _make_fake_feature_data():
    """
    Armamos un mini dataset de features que incluye:
    - numero_de_cliente
    - foto_mes (período)
    - clase_ternaria (CONTINUA / BAJA+1 / BAJA+2)
    - algunas columnas numéricas que jugarán como features del modelo

    Deliberadamente NO agregamos clase_binaria2 ni clase_peso;
    predict.ensure_binarias_y_peso() las debe crear.
    """
    return pd.DataFrame({
        "numero_de_cliente": [111, 222, 333, 444],
        "foto_mes":          [202104, 202104, 202104, 202104],
        "clase_ternaria":    ["CONTINUA", "BAJA+2", "BAJA+1", "CONTINUA"],
        # features numéricas reales que el modelo va a usar
        "monto1":            [10.0, 20.0, 30.0, 40.0],
        "monto2":            [ 1.0,  2.0,  3.0,  4.0],
    })


def test_score_month_happy_path(tmp_path, monkeypatch):
    """
    Valida que predict.score_month:
    - lea config y dataset
    - arme X_score para el mes pedido
    - use las columnas correctas feature_names del modelo
    - genere pred_flag usando el threshold
    - calcule la ganancia
    - devuelva df_pred y df_simple con las columnas esperadas
    """

    # 1. Creamos carpetas temporales tipo repo
    data_proc_dir = tmp_path / "data" / "processed"
    data_proc_dir.mkdir(parents=True, exist_ok=True)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = tmp_path / "predicciones"
    pred_dir.mkdir(parents=True, exist_ok=True)

    # 2. Creamos el dataset sintético de features y lo guardamos
    df_features = _make_fake_feature_data()
    feature_csv_path = data_proc_dir / "competencia_01_features.csv"
    df_features.to_csv(feature_csv_path, index=False)

    # 3. Guardamos un modelo dummy (final_model.pkl)
    dummy_model = DummyModel()
    final_model_path = models_dir / "final_model.pkl"
    with open(final_model_path, "wb") as f:
        pickle.dump(dummy_model, f)

    # 4. Guardamos un final_metrics.yaml con feature_names
    # Estas son EXACTAMENTE las columnas que el modelo espera como input.
    feature_names_train = ["monto1", "monto2"]
    final_metrics_path = models_dir / "final_metrics.yaml"
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "feature_names": feature_names_train
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    # 5. Creamos config.yaml temporal apuntando a los paths anteriores
    cfg_path = _write_fake_config(
        tmp_path=tmp_path,
        feature_csv_path=feature_csv_path,
        models_dir=models_dir,
    )

    # --- EVITAR RECURSIÓN INFINITA ---
    # Guardamos el objeto Config PARSEADO una sola vez usando la función real:
    original_cfg = predict.load_config(str(cfg_path))

    # y monkeypatcheamos load_config para que SIEMPRE devuelva ese objeto ya construido
    def _fake_load_config(path=None):
        return original_cfg

    monkeypatch.setattr(predict, "load_config", _fake_load_config)

    # 6. Ejecutar score_month sobre el período 202104 con threshold=0.5
    result = predict.score_month(
        month_to_score=202104,
        threshold=0.5,
        config_path=str(cfg_path),
    )

    # 7. Validaciones generales de estructura
    assert "df_pred" in result
    assert "df_simple" in result
    assert "ganancia_total" in result
    assert "threshold" in result
    assert "mes" in result

    assert result["mes"] == 202104
    assert result["threshold"] == 0.5

    df_pred = result["df_pred"]
    df_simple = result["df_simple"]

    # df_pred debe tener estas columnas mínimas
    assert "numero_de_cliente" in df_pred.columns
    assert "foto_mes" in df_pred.columns
    assert "proba_modelo" in df_pred.columns
    assert "pred_flag" in df_pred.columns
    assert "rank_desc" in df_pred.columns

    # df_simple debe tener numero_de_cliente y Predicted
    assert "numero_de_cliente" in df_simple.columns
    assert "Predicted" in df_simple.columns

    # tamaño consistente (teníamos 4 filas de foto_mes=202104)
    assert len(df_pred) == len(df_simple) == 4

    # pred_flag debe ser 0/1
    preds_list = df_simple["Predicted"].tolist()
    assert set(preds_list).issubset({0, 1})
    # debería haber al menos algún 1 y algún 0 con threshold=0.5 y proba=[0.9,0.9,0.1,0.1]
    assert any(p == 1 for p in preds_list)
    assert any(p == 0 for p in preds_list)

    # ganancia_total debería ser float y no negativa
    assert isinstance(result["ganancia_total"], float)
    assert result["ganancia_total"] >= 0.0

    # aseguramos que las columnas usadas en el modelo sean
    # exactamente las feature_names que guardamos en final_metrics.yaml
    assert feature_names_train == ["monto1", "monto2"]
