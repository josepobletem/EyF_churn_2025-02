"""
predict
=======

Uso (CLI):
    python -m src.predict --mes 202104 --threshold 0.025

Qué hace:
- Carga config.yaml
- Carga el modelo final entrenado (models/final_model.pkl)
- Filtra el dataset de features al mes pedido
- Calcula probabilidad de churn binario (clase_binaria2)
- Aplica un umbral (threshold) para generar pred_flag (0/1)
- Calcula la ganancia de negocio usando ese threshold
- Devuelve ranking de clientes por probabilidad
- Exporta:
    - un CSV detallado con proba/rank
    - un CSV simple con numero_de_cliente y Predicted
"""

import os
import sys
import yaml
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from pydantic import BaseModel, Field, ValidationError


# -----------------------------------------------------------------------------
# logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config models (coherente con trainer/optimizer actuales)
# -----------------------------------------------------------------------------

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str


class ColumnsConfig(BaseModel):
    id_column: str                  # ej "numero_de_cliente"
    period_column: str              # ej "foto_mes"
    target_column: str              # ej "clase_ternaria"
    binary_target_col: str          # ej "clase_binaria2"
    peso_col: str                   # ej "clase_peso"


class TrainConfig(BaseModel):
    models_dir: str = Field("models", description="Carpeta donde están final_model.pkl y final_metrics.yaml")
    train_months: list[int] | None = None
    test_month: int | None = None
    drop_cols: list[str] = Field(default_factory=list)

    # negocio
    ganancia_acierto: float = Field(
        780000.0,
        description="Ganancia por contactar un churn verdadero (TP)"
    )
    costo_estimulo: float = Field(
        20000.0,
        description="Costo por contactar un no-churn (FP)"
    )


class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: TrainConfig | None = None


def load_config(path: str = "config/config.yaml") -> FullConfig:
    """
    Lee config/config.yaml y valida estructura básica.
    """
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


# -----------------------------------------------------------------------------
# Métrica de negocio (ganancia)
# -----------------------------------------------------------------------------

def calcular_ganancia(y_true_bin: np.ndarray,
                      y_pred_bin: np.ndarray,
                      ganancia_acierto: float,
                      costo_estimulo: float) -> float:
    """
    Calcula la ganancia total según política comercial:
    - TP aporta ganancia_acierto
    - FP cuesta costo_estimulo
    - FN / TN no suman

    y_true_bin: array binario real (1 = cliente a contactar)
    y_pred_bin: array binario predicho (1 = lo contacto)
    """
    # verdaderos positivos = predije 1 y es 1
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    # falsos positivos = predije 1 pero en realidad es 0
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))

    gan = tp * ganancia_acierto - fp * costo_estimulo
    return float(gan)


# -----------------------------------------------------------------------------
# Helpers para preparar el dataset del mes a puntuar
# -----------------------------------------------------------------------------

def ensure_binarias_y_peso(df: pd.DataFrame,
                           cfg_cols: ColumnsConfig,
                           weight_baja2: float = 1.00002,
                           weight_baja1: float = 1.00001,
                           weight_continua: float = 1.0) -> pd.DataFrame:
    """
    Replica la lógica del pipeline:
        clase_binaria1 = 1 si BAJA+2, 0 si no
        clase_binaria2 = 0 si CONTINUA, 1 si BAJA+1 o BAJA+2
        clase_peso     = {BAJA+2:1.00002, BAJA+1:1.00001, resto:1.0}
    """
    df = df.copy()
    full_tgt = cfg_cols.target_column

    # clase_binaria1
    if "clase_binaria1" not in df.columns:
        df["clase_binaria1"] = np.where(df[full_tgt] == "BAJA+2", 1, 0)

    # clase_binaria2
    if "clase_binaria2" not in df.columns:
        df["clase_binaria2"] = np.where(df[full_tgt] == "CONTINUA", 0, 1)

    # clase_peso
    if cfg_cols.peso_col not in df.columns:
        df[cfg_cols.peso_col] = np.nan
        df.loc[df[full_tgt] == "BAJA+2", cfg_cols.peso_col] = weight_baja2
        df.loc[df[full_tgt] == "BAJA+1", cfg_cols.peso_col] = weight_baja1
        df.loc[~df[full_tgt].isin(["BAJA+1", "BAJA+2"]), cfg_cols.peso_col] = weight_continua
        df[cfg_cols.peso_col] = df[cfg_cols.peso_col].fillna(weight_continua)

    return df


def build_scoring_matrix(
    df_full: pd.DataFrame,
    cfg: FullConfig,
    month_to_score: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Toma el df completo de features,
    filtra sólo el mes 'month_to_score',
    y arma la matriz X_score y etiquetas reales (para ganancia).

    Devuelve:
    X_score        -> DataFrame numérico listo para .predict()
    info_df        -> DataFrame con columnas auxiliares:
                       [id_col, period_col, binary_target_col]
    feature_names  -> lista de features usadas como input
    """
    id_col = cfg.columns.id_column
    per_col = cfg.columns.period_column
    tgt_full = cfg.columns.target_column
    peso_col = cfg.columns.peso_col
    bin1 = "clase_binaria1"
    bin2 = cfg.columns.binary_target_col  # normalmente "clase_binaria2"

    # filtramos al mes específico
    df_mes = df_full[df_full[per_col] == month_to_score].copy()
    if df_mes.empty:
        raise ValueError(f"No hay filas para el mes {month_to_score} en el dataset de features")

    # armamos subset de info que vamos a devolver (id, mes, y verdadero binario)
    cols_info = [c for c in [id_col, per_col, bin2] if c in df_mes.columns]
    info_df = df_mes[cols_info].reset_index(drop=True)

    # dropear columnas no modelables
    block_cols = {
        id_col,
        per_col,
        tgt_full,
        peso_col,
        bin1,
        bin2,
    }

    drop_cols_cfg = []
    if cfg.train and cfg.train.drop_cols:
        drop_cols_cfg = cfg.train.drop_cols

    cols_to_remove = list(block_cols.union(drop_cols_cfg))
    X_score_raw = df_mes.drop(columns=[c for c in cols_to_remove if c in df_mes.columns],
                              errors="ignore")

    # quedarnos solo con columnas numéricas / bool
    valid_dtypes = (
        "int8","int16","int32","int64",
        "uint8","uint16","uint32","uint64",
        "float16","float32","float64",
        "bool"
    )
    X_score = X_score_raw.select_dtypes(include=list(valid_dtypes)).copy()

    if X_score.shape[1] == 0:
        raise ValueError(
            "Después de filtrar columnas y tipos, no quedaron features numéricas "
            f"para el mes {month_to_score}. Revisá drop_cols/config."
        )

    feature_names = list(X_score.columns)

    return X_score, info_df, feature_names


# -----------------------------------------------------------------------------
# Modelo final
# -----------------------------------------------------------------------------

def load_final_model_and_metadata(cfg: FullConfig):
    """
    Carga:
    - models/final_model.pkl (Booster entrenado en trainer.py)
    - models/final_metrics.yaml (info auxiliar)
    """
    train_cfg = cfg.train or TrainConfig()

    model_path = os.path.join(train_cfg.models_dir, "final_model_v2.pkl")
    metrics_path = os.path.join(train_cfg.models_dir, "final_metrics_v2.yaml")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No encontré el modelo final en {model_path}. Corré trainer.py primero.")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"No encontré métricas finales en {metrics_path}. Corré trainer.py primero.")

    with open(model_path, "rb") as f:
        final_model = pickle.load(f)

    with open(metrics_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    return final_model, meta, model_path, metrics_path


# -----------------------------------------------------------------------------
# Scoring + ganancia
# -----------------------------------------------------------------------------

def score_month(
    month_to_score: int,
    threshold: float = 0.025,
    config_path: str = "config/config.yaml",
) -> dict:
    """
    Calcula predicciones para el mes `month_to_score` y evalúa la ganancia
    con el umbral `threshold`.

    Devuelve un dict con:
    - df_pred (DataFrame con clientes y scores)
    - df_simple (numero_de_cliente + Predicted)
    - ganancia_total
    - threshold
    - mes
    """

    # 1. cargar config
    cfg = load_config(config_path)
    train_cfg = cfg.train or TrainConfig()

    # 2. cargar dataset completo de features
    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")
    df_full = pd.read_csv(feature_path)

    # 3. recrear columnas derivadas por consistencia
    df_full = ensure_binarias_y_peso(
        df_full,
        cfg.columns,
        weight_baja2=getattr(train_cfg, "weight_baja2", 1.00002),
        weight_baja1=getattr(train_cfg, "weight_baja1", 1.00001),
        weight_continua=getattr(train_cfg, "weight_continua", 1.0),
    )

    # 4. armar matriz X del mes pedido + info_df con verdad terreno binaria
    X_score, info_df, feature_names = build_scoring_matrix(
        df_full,
        cfg,
        month_to_score=month_to_score,
    )

    logger.info(
        "Mes %s -> scoring rows: %s, features: %s",
        month_to_score, X_score.shape[0], X_score.shape[1]
    )

    # 5. cargar modelo final
    final_model, meta_yaml, model_path, metrics_path = load_final_model_and_metadata(cfg)
    logger.info("Usando modelo final: %s", model_path)

    # columnas EXACTAS con las que el modelo fue entrenado
    feature_names_train = meta_yaml.get("feature_names", None)
    if feature_names_train is None:
        raise ValueError(
            "final_metrics.yaml no contiene 'feature_names'. "
            "Actualizá trainer.py para guardar la lista de columnas usadas en el entrenamiento."
        )

    # aseguramos que X_score tenga exactamente esas columnas y en ese orden
    # si alguna columna falta en este mes, la creamos con 0
    for col in feature_names_train:
        if col not in X_score.columns:
            X_score[col] = 0.0

    # si sobran columnas en X_score que no estaban en entrenamiento, las descartamos
    X_score = X_score[feature_names_train]

    # 6. predecir probabilidades
    proba = final_model.predict(X_score)
    proba = np.clip(proba, 1e-15, 1 - 1e-15)

    # 7. aplicar threshold
    pred_flag = (proba >= threshold).astype(int)

    # 8. calcular ganancia usando clase_binaria2 como y_true
    if cfg.columns.binary_target_col in info_df.columns:
        y_true_bin = info_df[cfg.columns.binary_target_col].astype(int).to_numpy()
    else:
        # fallback: por ejemplo scoring productivo sin ground truth del mes futuro
        y_true_bin = np.zeros(len(pred_flag), dtype=int)

    gan_total = calcular_ganancia(
        y_true_bin=y_true_bin,
        y_pred_bin=pred_flag,
        ganancia_acierto=getattr(train_cfg, "ganancia_acierto", 780000.0),
        costo_estimulo=getattr(train_cfg, "costo_estimulo", 20000.0),
    )

    # 9. armar tabla final ordenada
    id_col = cfg.columns.id_column
    per_col = cfg.columns.period_column

    out = pd.DataFrame({
        id_col: info_df[id_col].to_numpy(),
        per_col: info_df[per_col].to_numpy(),
        "proba_modelo": proba,
        "pred_flag": pred_flag,
    })

    out["rank_desc"] = out["proba_modelo"].rank(
        ascending=False,
        method="first"
    ).astype(int)

    out = out.sort_values("proba_modelo", ascending=False).reset_index(drop=True)

    # dataframe "simple" para negocio
    df_simple = pd.DataFrame({
        id_col: out[id_col],
        "Predicted": out["pred_flag"].astype(int),
    })

    return {
        "df_pred": out,
        "df_simple": df_simple,
        "ganancia_total": gan_total,
        "threshold": threshold,
        "mes": month_to_score,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Scoring mensual con el modelo final entrenado + ganancia de negocio"
    )
    parser.add_argument(
        "--mes",
        type=int,
        required=True,
        help="Mes (foto_mes) a puntuar, ej 202104"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.035,
        help="Corte de probabilidad para marcar pred_flag=1"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al config.yaml"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    result = score_month(
        month_to_score=args.mes,
        threshold=args.threshold,
        config_path=args.config,
    )

    df_pred = result["df_pred"]
    df_simple = result["df_simple"]
    gan_total = result["ganancia_total"]

    # métricas de cobertura (cuántos marcamos como 1)
    n_total = len(df_simple)
    n_pred1 = int((df_simple["Predicted"] == 1).sum())
    pct_pred1 = (n_pred1 / n_total * 100.0) if n_total > 0 else 0.0

    logger.info("==============================================")
    logger.info("Mes %s  | Threshold=%.4f", result["mes"], result["threshold"])
    logger.info("Ganancia estimada: $ %.0f", gan_total)
    logger.info("Contactables (Predicted=1): %d de %d (%.2f%%)", n_pred1, n_total, pct_pred1)
    logger.info("Top 10 predicciones:")
    logger.info("\n%s", df_pred.head(10).to_string(index=False))
    logger.info("==============================================")

    # Guardar outputs
    out_dir = "predicciones"
    os.makedirs(out_dir, exist_ok=True)

    # CSV detallado con proba / rank
    out_path_full = os.path.join(
        out_dir,
        f"pred_{result['mes']}_thr{args.threshold:.4f}_best_param_v2.csv"
    )
    df_pred.to_csv(out_path_full, index=False)
    logger.info("Guardé scoring detallado en %s", out_path_full)

    # CSV simple con numero_de_cliente y Predicted
    # Nota: usamos el nombre de la columna id_column real del config
    id_col = load_config(args.config).columns.id_column
    out_path_simple = os.path.join(
        out_dir,
        f"pred_simple_{result['mes']}_thr{args.threshold:.4f}_best_param_v2.csv"
    )
    df_simple[[id_col, "Predicted"]].to_csv(out_path_simple, index=False)
    logger.info("Guardé scoring simple en %s", out_path_simple)


if __name__ == "__main__":
    main()
