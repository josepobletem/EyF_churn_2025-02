"""
trainer
=======

Entrena el modelo final usando los mejores hiperparámetros encontrados
por optimizer.py y guarda el modelo entrenado definitivo.

Flujo:
1. Carga dataset de features finales (paths.feature_dataset).
2. Construye columnas de negocio:
   - clase_binaria1
   - clase_binaria2
   - clase_peso
3. Filtra SOLO los meses de train_months definidos en config.train.train_months.
4. Carga best_params.yaml generado por optimizer.py.
5. Reentrena LightGBM en TODO el set de entrenamiento.
6. Guarda:
   - models/final_model.pkl
   - models/final_metrics.yaml  (logloss, matriz de confusión, etc.)
7. Loggea métricas in-sample básicas.

Uso:
    python -m src.trainer
"""

import os
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import log_loss, confusion_matrix
import lightgbm as lgb  # usamos lgb.train, igual que en optimizer


# -----------------------
# logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------
# Config schemas
# -----------------------

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str


class ColumnsConfig(BaseModel):
    id_column: str                  # ej "numero_de_cliente"
    period_column: str              # ej "foto_mes"
    target_column_full: str = Field(
        "clase_ternaria",
        description="Target multiclase original (BAJA+1 / BAJA+2 / CONTINUA)"
    )
    binary_target_col: str = Field(
        "clase_binaria2",
        description=(
            "Cuál usar como target binario final: "
            "clase_binaria1 = 1 si BAJA+2, 0 si no "
            "clase_binaria2 = 1 si BAJA+1 o BAJA+2, 0 si CONTINUA"
        )
    )
    peso_col: str = Field(
        "clase_peso",
        description="Columna de pesos que se usa como weight"
    )


class TrainConfig(BaseModel):
    models_dir: str = Field("models")

    # estos vienen del mismo config.yaml que usa optimizer
    train_months: list[int] = Field(
        default_factory=lambda: [202101, 202102, 202103, 202104],
        description="Meses que usamos efectivamente para entrenar"
    )
    test_month: int = Field(
        202104,
        description="Mes de holdout / validación temporal"
    )

    # las mismas columnas peligrosas que sacamos en optimizer
    drop_cols: list[str] = Field(
        default_factory=lambda: ["lag_3_ctrx_quarter"],
        description="Columnas a eliminar por fuga/leak/etc."
    )

    # negocio / pesos
    weight_baja2: float = Field(1.00002)
    weight_baja1: float = Field(1.00001)
    weight_continua: float = Field(1.0)

    # reproducibilidad
    seed: int = Field(12345)

    ganancia_acierto: float = Field(
        780000.0,
        description="Ganancia por acertar churn verdadero (TP)"
    )
    costo_estimulo: float = Field(
        20000.0,
        description="Costo por contactar un no-churn (FP)"
    )

class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: TrainConfig | None = None


# -----------------------
# Helpers de config
# -----------------------

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


# -----------------------
# Feature engineering de negocio
# (misma lógica que usamos en optimizer)
# -----------------------

def ensure_binarias_y_peso(df: pd.DataFrame, cfg_cols: ColumnsConfig, train_cfg: TrainConfig) -> pd.DataFrame:
    """
    Crea (o sobrescribe) en df:
      - clase_binaria1 = 1 si BAJA+2, 0 si no
      - clase_binaria2 = 1 si BAJA+1 o BAJA+2, 0 si CONTINUA
      - clase_peso     usando weight_baja2 / weight_baja1 / weight_continua
    Esto mantiene consistencia con optimizer.
    """
    df = df.copy()

    tcol = cfg_cols.target_column_full
    peso_col = cfg_cols.peso_col

    # binaria1: BAJA+2 vs resto
    df["clase_binaria1"] = np.where(df[tcol] == "BAJA+2", 1, 0)

    # binaria2: churn (BAJA+1 o BAJA+2) vs CONTINUA
    df["clase_binaria2"] = np.where(df[tcol] == "CONTINUA", 0, 1)

    # peso por clase
    if peso_col not in df.columns:
        df[peso_col] = np.nan

    df.loc[df[tcol] == "BAJA+2", peso_col] = train_cfg.weight_baja2
    df.loc[df[tcol] == "BAJA+1", peso_col] = train_cfg.weight_baja1
    df.loc[~df[tcol].isin(["BAJA+1", "BAJA+2"]), peso_col] = train_cfg.weight_continua

    df[peso_col] = df[peso_col].fillna(train_cfg.weight_continua)

    return df


def build_train_matrix(df: pd.DataFrame, cfg: FullConfig, train_cfg: TrainConfig):
    """
    Replica el criterio del optimizer:
    - filtra df a solo meses de train_months
    - dropea columnas con fuga
    - separa target binario (cfg.columns.binary_target_col)
    - usa peso como sample_weight
    - filtra features no numéricas
    """
    per_col = cfg.columns.period_column
    peso_col = cfg.columns.peso_col
    target_bin_col = cfg.columns.binary_target_col

    # quedarnos sólo con meses de entrenamiento
    df_train = df[df[per_col].isin(train_cfg.train_months)].copy()

    # sacar columnas que sabemos que no deben ir
    if train_cfg.drop_cols:
        df_train = df_train.drop(
            columns=[c for c in train_cfg.drop_cols if c in df_train.columns],
            errors="ignore"
        )

    # columnas que nunca deben entrar como features
    block_cols = {
        cfg.columns.id_column,
        cfg.columns.period_column,
        cfg.columns.target_column_full,
        "clase_binaria1",
        "clase_binaria2",
        peso_col,
    }

    # Features crudas
    X = df_train.drop(
        columns=[c for c in block_cols if c in df_train.columns],
        errors="ignore"
    )

    # Target binario
    if target_bin_col not in df_train.columns:
        raise KeyError(
            f"No encontré {target_bin_col} en df_train. "
            "Revisar binary_target_col en config.columns."
        )
    y = df_train[target_bin_col].astype(int).to_numpy()

    # Weight
    w = df_train[peso_col].astype(float).to_numpy()

    # LightGBM requiere numérico/bool
    valid_dtypes = (
        "int8","int16","int32","int64",
        "uint8","uint16","uint32","uint64",
        "float16","float32","float64",
        "bool"
    )
    X_numeric = X.select_dtypes(include=list(valid_dtypes)).copy()

    feature_names = list(X_numeric.columns)

    return X_numeric, y, w, feature_names


def train_final_model() -> dict:
    """
    Entrena el modelo final usando TODOS los datos de entrenamiento (train_months)
    con los mejores hiperparámetros encontrados por optimizer.py.

    También guarda métricas in-sample usando esas mismas filas de train_months.
    """
    logger.info("🚂 Entrenamiento final del modelo...")

    # 1. cargar config
    cfg = load_config()
    train_cfg = cfg.train or TrainConfig()

    # 2. cargar dataset de features finales
    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")

    df_full = pd.read_csv(feature_path)

    # 3. asegurar columnas de negocio (binarias + peso)
    df_full = ensure_binarias_y_peso(df_full, cfg.columns, train_cfg)

    # 4. construir matrices de train según meses definidos en config
    X_train, y_train, w_train, feature_names = build_train_matrix(df_full, cfg, train_cfg)

    logger.info("Shape train X: %s, y: %s", X_train.shape, y_train.shape)
    logger.info("Meses usados para entrenar: %s", train_cfg.train_months)
    logger.info("Mes holdout reservado (no entrenado): %s", train_cfg.test_month)

    # 5. cargar hiperparámetros óptimos del optimizer
    params_path = os.path.join(train_cfg.models_dir, "best_params.yaml")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"No encontré {params_path}. Corré primero optimizer.py."
        )

    with open(params_path, "r", encoding="utf-8") as f:
        best_info = yaml.safe_load(f)

    best_params = best_info["best_params"]
    best_iteration = int(best_info.get("best_iteration", 500))  # fallback defensivo

    logger.info("Hiperparámetros óptimos cargados de optimizer:")
    logger.info(best_params)
    logger.info("Usando best_iteration = %d", best_iteration)

    # 6. entrenar modelo final con LightGBM "core"
    lgb_train = lgb.Dataset(
        data=X_train,
        label=y_train,
        weight=w_train,
        free_raw_data=False
    )

    final_model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=best_iteration
    )

    # 7. métricas in-sample (sanity check) sobre el mismo train
    # LightGBM en binario devuelve prob(clase positiva)
    y_pred_proba = final_model.predict(X_train)

    # evitar probs 0/1 exactas antes de log_loss
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

    # logloss binaria
    logloss_in = log_loss(y_train, y_pred_proba)

    # matriz de confusión usando threshold 0.5
    y_pred_label = (y_pred_proba >= 0.5).astype(int)
    cm_in = confusion_matrix(y_train, y_pred_label).tolist()

    logger.info("📊 LogLoss in-sample: %.6f", logloss_in)
    logger.info("📊 Matriz de confusión in-sample:\n%s", cm_in)

    # 8. guardar modelo final
    os.makedirs(train_cfg.models_dir, exist_ok=True)

    final_model_path = os.path.join(train_cfg.models_dir, "final_model.pkl")
    with open(final_model_path, "wb") as f:
        pickle.dump(final_model, f)

    logger.info("💾 Modelo final guardado en %s", final_model_path)

    # 9. guardar métricas finales y metadata de entrenamiento
    metrics_path = os.path.join(train_cfg.models_dir, "final_metrics.yaml")
    with open(metrics_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                "train_months": train_cfg.train_months,
                "test_month": int(train_cfg.test_month),
                "feature_names": feature_names,
                "binary_target_col": cfg.columns.binary_target_col,
                "period_column": cfg.columns.period_column,
                "id_column": cfg.columns.id_column,

                "best_iteration": best_iteration,
                "logloss_in_sample": float(logloss_in),
                "confusion_matrix_in_sample": cm_in,
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
