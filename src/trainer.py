"""
trainer
=======

Entrena el modelo final usando los mejores hiperparámetros encontrados
por optimizer.py y guarda el modelo entrenado definitivo.

Flujo:
1. Carga dataset de features finales (paths.feature_dataset).
2. Carga best_params.yaml generado por optimizer.py.
3. Entrena LightGBM con TODO el dataset.
4. Guarda el modelo final en models/final_model.pkl.
5. Loggea métricas in-sample básicas.

Uso:
    python -m src.trainer
"""

import os
import logging
import pickle
import yaml
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import log_loss, confusion_matrix
from lightgbm import LGBMClassifier


# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# --------- Reusar esquemas livianos ---------

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str


class ColumnsConfig(BaseModel):
    id_column: str
    period_column: str
    target_column: str = Field("clase_ternaria")


class TrainConfig(BaseModel):
    models_dir: str = Field("models")


class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: TrainConfig | None = None


def load_config(path: str = "config/config.yaml") -> FullConfig:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    if "train" not in raw_cfg:
        raw_cfg["train"] = {}
    try:
        return FullConfig(**raw_cfg)
    except ValidationError as e:
        logger.error("Config inválida:\n%s", e)
        raise


def _prepare_xy(df: pd.DataFrame, id_col: str, period_col: str, target_col: str):
    # Igual que en optimizer.py, pero duplicado acá para no tener dependencias cruzadas
    drop_cols = [c for c in [id_col, period_col, target_col] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col].astype("category")
    return X, y


def train_final_model() -> dict:
    """
    Entrena el modelo final usando TODOS los datos y
    guarda el modelo y algunas métricas in-sample.

    Returns
    -------
    dict
        Info resumida con paths y métricas básicas.
    """
    logger.info("🚂 Entrenamiento final del modelo...")

    cfg = load_config()
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")

    df = pd.read_csv(feature_path)
    X, y = _prepare_xy(df, cfg.columns.id_column, cfg.columns.period_column, cfg.columns.target_column)

    # cargar hiperparámetros óptimos
    params_path = os.path.join(train_cfg.models_dir, "best_params.yaml")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"No encontré {params_path}. Corré primero optimizer.py."
        )

    with open(params_path, "r", encoding="utf-8") as f:
        best_info = yaml.safe_load(f)

    best_params = best_info["best_params"]

    # agregamos lo que el optimizer usó fijo
    best_params_for_fit = {
        **best_params,
        "objective": "multiclass",
        "class_weight": None,
    }

    logger.info("Usando hiperparámetros óptimos: %s", best_params_for_fit)

    final_model = LGBMClassifier(**best_params_for_fit)
    final_model.fit(X, y)

    # métricas in-sample (para sanity check)
    proba_in = final_model.predict_proba(X)
    logloss_in = log_loss(y, proba_in)
    preds_in = final_model.predict(X)
    cm_in = confusion_matrix(y, preds_in)

    logger.info("📊 LogLoss in-sample: %.6f", logloss_in)
    logger.info("📊 Matriz de confusión in-sample:\n%s", cm_in)

    # guardar modelo final
    os.makedirs(train_cfg.models_dir, exist_ok=True)
    final_model_path = os.path.join(train_cfg.models_dir, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(final_model, f)

    logger.info("💾 Modelo final guardado en %s", final_model_path)

    # también guardo métricas de control rápido
    # también guardo métricas de control rápido
    metrics_path = os.path.join(train_cfg.models_dir, "final_metrics.yaml")

    # Aseguramos tipos puros de Python (str, float, list de lists)
    classes_python = [str(c) for c in final_model.classes_]
    cm_python = cm_in.tolist()
    logloss_python = float(logloss_in)

    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "logloss_in_sample": logloss_python,
                "confusion_matrix_in_sample": cm_python,
                "classes": classes_python,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    logger.info("💾 Métricas finales guardadas en %s", metrics_path)

    return {
        "model_path": final_model_path,
        "metrics_path": metrics_path,
        "logloss_in_sample": logloss_in,
    }


if __name__ == "__main__":
    train_final_model()
