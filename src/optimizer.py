"""
optimizer
=========

Búsqueda de hiperparámetros con Optuna para el modelo de churn.

Flujo:
1. Carga el dataset con features finales (`paths.feature_dataset`).
2. Separa X (features) e y (target).
3. Split en train/valid.
4. Usa Optuna para optimizar hiperparámetros de LightGBM.
5. Guarda:
   - mejores hiperparámetros en models/best_params.yaml
   - mejor modelo en models/best_model.pkl

Uso:
    python -m src.optimizer
"""

import os
import logging
import pickle
import yaml
import optuna
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

# logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# Config schema (re-uso)
# -------------------------

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str = Field(..., description="Dataset final con features")
    
    # opcional futuro:
    # predictions_output: str


class ColumnsConfig(BaseModel):
    id_column: str
    period_column: str
    target_column: str = Field(
        "clase_ternaria",
        description="Columna target multiclase"
    )


class FeaturesConfig(BaseModel):
    base_table_name: str
    steps: list[str]


class TrainConfig(BaseModel):
    test_size: float = Field(
        0.2,
        description="Proporción hold-out validación"
    )
    random_state: int = Field(
        42,
        description="Semilla reproducible"
    )
    n_trials: int = Field(
        30,
        description="Cantidad de trials Optuna"
    )
    models_dir: str = Field(
        "models",
        description="Carpeta donde guardar modelo y params óptimos"
    )


class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    features: FeaturesConfig
    train: TrainConfig | None = None  # si no está, usamos defaults en runtime


def load_config(path: str = "config/config.yaml") -> FullConfig:
    """
    Cargar configuración completa.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encontré {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    # si falta 'train' en YAML, le damos defaults
    if "train" not in raw_cfg:
        raw_cfg["train"] = {}

    try:
        return FullConfig(**raw_cfg)
    except ValidationError as e:
        logger.error("Config inválida:\n%s", e)
        raise


def _prepare_xy(df: pd.DataFrame, id_col: str, period_col: str, target_col: str):
    """
    Quita las columnas no-modelables (id, periodo, target),
    devuelve X (features num/cat) y y (target multiclase).
    """
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} no está en el dataset final.")

    drop_cols = [id_col, period_col, target_col]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col].astype("category")

    return X, y


def _objective_factory(X_train, y_train, X_valid, y_valid):
    """
    Devuelve una función objetivo para Optuna.
    Optuna llama a esta función en cada trial para probar hiperparámetros.
    """

    def objective(trial: optuna.Trial):
        # espacio de búsqueda de hiperparámetros LightGBM
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # importante para multiclase en churn:
            "objective": "multiclass",
            "class_weight": None,
        }

        model = LGBMClassifier(**params)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="multi_logloss",
        )

        # predicciones prob (para logloss multiclase)
        proba_valid = model.predict_proba(X_valid)

        score = log_loss(y_valid, proba_valid)
        return score

    return objective


def run_hyperparam_search() -> dict:
    """
    Ejecuta búsqueda de hiperparámetros con Optuna y guarda:
    - best_params.yaml
    - best_model.pkl

    Returns
    -------
    dict
        Diccionario con 'best_params' y 'best_score'.
    """
    logger.info("🔍 Iniciando búsqueda de hiperparámetros...")

    cfg = load_config()
    train_cfg = cfg.train or TrainConfig()  # fallback a defaults en caso extremo

    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré el dataset de features {feature_path}")

    df = pd.read_csv(feature_path)

    X, y = _prepare_xy(df, cfg.columns.id_column, cfg.columns.period_column, cfg.columns.target_column)

    # split train/valid
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=train_cfg.test_size,
        random_state=train_cfg.random_state,
        stratify=y,
    )

    logger.info("Shapes: train=%s valid=%s", X_train.shape, X_valid.shape)

    # Optuna study
    objective = _objective_factory(X_train, y_train, X_valid, y_valid)

    study = optuna.create_study(direction="minimize")  # minimizar logloss
    study.optimize(objective, n_trials=train_cfg.n_trials)

    best_params = study.best_params
    best_score = study.best_value

    logger.info("🏆 Mejores hiperparámetros: %s", best_params)
    logger.info("🏁 Mejor logloss valid: %.6f", best_score)

    # entrenar un modelo final con los mejores hiperparámetros en train split (no todo el dataset aún)
    best_params_for_fit = {
        **best_params,
        "objective": "multiclass",
        "class_weight": None,
    }
    best_model = LGBMClassifier(**best_params_for_fit)
    best_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="multi_logloss"
    )

    # guardado
    os.makedirs(train_cfg.models_dir, exist_ok=True)

    params_path = os.path.join(train_cfg.models_dir, "best_params.yaml")
    model_path = os.path.join(train_cfg.models_dir, "best_model.pkl")

    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "best_params": best_params,
                "best_score_logloss": float(best_score),
                "target_column": cfg.columns.target_column,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    logger.info("💾 Guardé hiperparámetros óptimos en %s", params_path)
    logger.info("💾 Guardé el mejor modelo preliminar en %s", model_path)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "model_path": model_path,
        "params_path": params_path,
    }


if __name__ == "__main__":
    run_hyperparam_search()
