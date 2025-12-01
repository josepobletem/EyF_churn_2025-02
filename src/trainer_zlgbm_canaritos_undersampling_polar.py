# -*- coding: utf-8 -*-
"""
zLGBM pipeline con Polars
=========================

- Lee features (parquet) con Polars (local o GCS).
- Crea targets binarios y pesos con Polars.
- Añade canaritos con Polars.
- Separa train / future y hace undersampling de CONTINUA en Polars.
- Convierte solo los subsets finales a pandas para entrenar LightGBM.
"""

import os
import io
import logging
from typing import Dict, Any, List
import tempfile
import subprocess
import pickle
import yaml
import fsspec
import time

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb


from pydantic import BaseModel, Field, ValidationError

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG SCHEMAS
# =============================================================================

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str


class ColumnsConfig(BaseModel):
    id_column: str
    period_column: str
    target_column: str = Field(
        "clase_ternaria",
        description="Target multiclase original (BAJA+1 / BAJA+2 / CONTINUA)"
    )
    binary_target_col: str = Field(
        "clase_binaria2",
        description="Target binario general (BAJA+1/BAJA+2 vs CONTINUA)"
    )
    peso_col: str = "clase_peso"
    binary_target_gan: str = Field(
        "clase_binaria1",
        description="Target binario para ganancia (1 si BAJA+2, 0 resto)"
    )


class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: Dict[str, Any] | None = None  # dict genérico


# =============================================================================
# HELPERS CONFIG
# =============================================================================

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


# -----------------------------------------------------------------------------
# Helpers GCS / Local
# -----------------------------------------------------------------------------
def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")


def _join(base: str, *parts: str) -> str:
    if _is_gcs(base):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    return os.path.join(base, *parts)


def _path_exists(path: str) -> bool:
    if _is_gcs(path):
        fs = fsspec.filesystem("gcs", token="cloud")
        return fs.exists(path)
    return os.path.exists(path)


def _read_parquet(path: str) -> pl.DataFrame:
    """
    Lee parquet como Polars DataFrame (local o GCS).
    """
    if _is_gcs(path):
        with fsspec.open(path, "rb", **{"token": "cloud"}) as f:
            return pl.read_parquet(f)
    return pl.read_parquet(path)


def _read_yaml(path: str) -> dict:
    if _is_gcs(path):
        with fsspec.open(path, "r", **{"token": "cloud"}) as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_pickle(path: str):
    if _is_gcs(path):
        with fsspec.open(path, "rb", **{"token": "cloud"}) as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _gsutil_fallback_write(gcs_path: str, data: bytes):
    """
    Escribe en GCS usando gsutil cp, guardando primero en /dev/shm (RAM).
    Usa las credenciales de gcloud (user), no el service account de la VM.
    """
    os.makedirs("/dev/shm", exist_ok=True)
    with tempfile.NamedTemporaryFile(dir="/dev/shm", delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        tmp_path = tmp.name

    try:
        subprocess.run(["gsutil", "-q", "cp", tmp_path, gcs_path], check=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _write_text(path: str, text: str):
    """
    Escribe texto tanto local como en GCS.
    En GCS usamos siempre gsutil para evitar el problema de scopes.
    """
    if _is_gcs(path):
        data = text.encode("utf-8")
        _gsutil_fallback_write(path, data)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_bytes(path: str, data: bytes):
    """
    Escribe bytes (modelos, pickles, etc.) local o en GCS.
    """
    if _is_gcs(path):
        _gsutil_fallback_write(path, data)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def _write_csv_df(df: pd.DataFrame, path: str):
    """
    Guarda un DataFrame en CSV; si es GCS, usa gsutil cp.
    """
    if _is_gcs(path):
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        _gsutil_fallback_write(path, csv_bytes)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# CANARITOS (POLARS)
# =============================================================================

def create_canaritos(df: pl.DataFrame, qcanaritos: int = 100) -> pl.DataFrame:
    """
    Añade qcanaritos columnas canarito_i con ruido uniforme [0,1] al principio.
    """
    original_cols = df.columns
    num_filas = df.height

    canary_cols = [f"canarito_{i}" for i in range(1, qcanaritos + 1)]

    rand_matrix = np.random.rand(num_filas, qcanaritos)
    for i, name in enumerate(canary_cols):
        df = df.with_columns(pl.Series(name, rand_matrix[:, i]))

    df = df.select(canary_cols + original_cols)
    return df


# =============================================================================
# FEATURE ENGINEERING PARA TARGETS Y PESOS (POLARS)
# =============================================================================

def ensure_binarias_y_peso(df: pl.DataFrame, cfg_cols: ColumnsConfig) -> pl.DataFrame:
    """
    Versión Polars:
      - binary_target_gan: 1 si BAJA+2, 0 resto
      - binary_target_col: 1 si BAJA+1 o BAJA+2, 0 si CONTINUA
      - peso_col: por defecto = 1.0 si no existe
    """
    tcol = cfg_cols.target_column
    if tcol not in df.columns:
        raise KeyError(f"Target multiclase {tcol} no está en el dataset de features.")

    col_gan = cfg_cols.binary_target_gan
    col_bin = cfg_cols.binary_target_col
    peso_col = cfg_cols.peso_col

    df = df.with_columns([
        pl.when(pl.col(tcol) == "BAJA+2").then(1).otherwise(0).alias(col_gan),
        pl.when(pl.col(tcol) == "CONTINUA").then(0).otherwise(1).alias(col_bin),
    ])

    if peso_col not in df.columns:
        df = df.with_columns(pl.lit(1.0).alias(peso_col))

    return df


# =============================================================================
# CONFIG zLGBM + ENSEMBLE
# =============================================================================

class ZLGBMConfig(BaseModel):
    """Config específica del experimento tipo zLightGBM + ensemble."""
    models_dir: str = Field(
        "gs://jose_poblete_bukito3/eyf/models_zlgbm",
        description="Carpeta donde guardar modelos y artefactos"
    )
    pred_dir: str = Field(
        "gs://jose_poblete_bukito3/eyf/prediccion_zlgbm",
        description="Carpeta para predicciones detalladas"
    )
    kaggle_dir: str = Field(
        "gs://jose_poblete_bukito3/eyf/kaggle_zlgbm",
        description="Carpeta para archivos estilo Kaggle"
    )

    future_months: List[int] = Field(
        default_factory=lambda: [202107],
        description="Meses de holdout/competencia"
    )

    ganancia_acierto: float = 780000.0
    costo_estimulo: float = 20000.0

    qcanaritos: int = 20
    experimento: str = "zlgbm_canaritos_ensamble_01"

    n_models: int = 50
    seeds: List[int] | None = None
    base_seed: int = 464939

    n_envios: int = 11500

    max_bin: int = 31
    min_data_in_leaf: int = 50
    num_iterations: int = 700
    num_leaves: int = 999
    learning_rate: float = 1.0
    feature_fraction: float = 0.50
    gradient_bound: float = 0.1  # sólo documental

    undersample_factor: float = Field(
        6.0,
        description="Máximo ratio CONTINUA / no-CONTINUA en train (undersampling)"
    )


def build_lgbm_params(zcfg: ZLGBMConfig, seed_value: int) -> Dict[str, Any]:
    return {
        "boosting_type": "gbdt",
        "objective": "binary",
        "metric": "None",
        "first_metric_only": False,
        "boost_from_average": True,
        "feature_pre_filter": False,
        "force_row_wise": True,
        "verbosity": -100,
        "seed": seed_value,
        "max_bin": zcfg.max_bin,
        "min_data_in_leaf": zcfg.min_data_in_leaf,
        "num_leaves": zcfg.num_leaves,
        "learning_rate": zcfg.learning_rate,
        "feature_fraction": zcfg.feature_fraction,
        "canaritos": zcfg.qcanaritos,
        "gradient_bound": zcfg.gradient_bound,
    }


def _build_seeds_from_cfg(cfg: FullConfig, zcfg: ZLGBMConfig) -> List[int]:
    seeds = None

    if isinstance(cfg.train, dict) and "seeds" in cfg.train:
        seeds_raw = cfg.train["seeds"]
        if isinstance(seeds_raw, list) and len(seeds_raw) > 0:
            seeds = [int(s) for s in seeds_raw]

    if seeds is None and zcfg.seeds:
        seeds = [int(s) for s in zcfg.seeds]

    if seeds is None:
        seeds = [zcfg.base_seed + i for i in range(zcfg.n_models)]

    if len(seeds) > zcfg.n_models:
        seeds = seeds[: zcfg.n_models]

    logger.info("Semillas para ensemble zLGBM: %s", seeds)
    return seeds


# =============================================================================
# GANANCIA CON MERGE EXPLÍCITO (PANDAS)
# =============================================================================

def calcular_ganancia_ordenada(
    df_pred: pd.DataFrame,
    df_true: pd.DataFrame,
    cfg_cols: ColumnsConfig,
    ganancia_acierto: float,
    costo_estimulo: float,
    top_n: int | None = None,
) -> float:
    id_col = cfg_cols.id_column
    per_col = cfg_cols.period_column
    bin_col = cfg_cols.binary_target_gan

    merged = df_pred.merge(
        df_true[[id_col, per_col, bin_col]],
        on=[id_col, per_col],
        how="left",
        suffixes=("", "_true"),
    )

    if merged[bin_col].isna().all():
        logger.warning(
            "Todas las labels están en NaN en el merge. "
            "¿Seguro que hay targets para esos meses future?"
        )

    merged[bin_col] = merged[bin_col].fillna(0).astype(int)

    merged = merged.sort_values("prob", ascending=False).reset_index(drop=True)
    y_true = merged[bin_col].to_numpy().astype(int)

    if top_n is not None:
        top_n_eff = min(top_n, len(y_true))
        y_true_top = y_true[:top_n_eff]
        cash_flow_top = np.where(
            y_true_top == 1,
            ganancia_acierto,
            -costo_estimulo,
        )
        gan_total = float(cash_flow_top.sum())

        logger.info(
            "Ganancia en top_n=%d (ordenando por prob ensemble) = %.2f",
            top_n_eff,
            gan_total,
        )
        return gan_total

    cash_flow = np.where(y_true == 1, ganancia_acierto, -costo_estimulo)
    gan_acum = np.cumsum(cash_flow)

    gan_max = float(np.max(gan_acum))
    idx_opt = int(np.argmax(gan_acum)) + 1

    logger.info(
        "Ganancia máxima (ordenando por prob ensemble) = %.2f en top %d clientes",
        gan_max,
        idx_opt,
    )

    return gan_max


# =============================================================================
# PIPELINE PRINCIPAL (zLGBM + ENSEMBLE con Polars)
# =============================================================================

def run_zlgbm_pipeline(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    t_start = time.time()

    # ---------------- Config ----------------
    cfg = load_config(config_path)
    zcfg = ZLGBMConfig()

    feature_path = cfg.paths.feature_dataset
    logger.info("Leyendo dataset de features (Polars) desde: %s", feature_path)

    df_pl = _read_parquet(feature_path)
    logger.info("Dataset original (Polars): %s filas, %s columnas", df_pl.height, len(df_pl.columns))

    # Targets/pesos en Polars
    df_pl = ensure_binarias_y_peso(df_pl, cfg.columns)

    # Canaritos en Polars
    logger.info("Creando canaritos en Polars (q=%d)...", zcfg.qcanaritos)
    df_pl = create_canaritos(df_pl, qcanaritos=zcfg.qcanaritos)
    logger.info("Dataset con canaritos (Polars): %s filas, %s columnas", df_pl.height, len(df_pl.columns))

    # ---------------- Resolver meses de train y future ----------------
    id_col = cfg.columns.id_column
    per_col = cfg.columns.period_column
    t_bin_gan = cfg.columns.binary_target_gan
    peso_col = cfg.columns.peso_col
    target_multiclase = cfg.columns.target_column

    all_months = sorted(df_pl.select(per_col).unique().to_series().to_list())
    logger.info("Meses disponibles en dataset: %s", all_months)

    future_months = zcfg.future_months
    if isinstance(cfg.train, dict) and "future_months" in cfg.train:
        fm = cfg.train.get("future_months") or []
        if isinstance(fm, list) and len(fm) > 0:
            future_months = [int(m) for m in fm]
    logger.info("Meses future (holdout/competencia): %s", future_months)

    if isinstance(cfg.train, dict) and "train_months" in cfg.train:
        tm_raw = cfg.train.get("train_months") or []
        train_months = [int(m) for m in tm_raw]
        if len(train_months) == 0:
            train_months = [m for m in all_months if m not in set(future_months)]
            logger.warning(
                "config.train.train_months está vacío; uso default (all_months sin future): %s",
                train_months,
            )
    else:
        train_months = [m for m in all_months if m not in set(future_months)]
        logger.info(
            "No se encontró train.train_months en config; uso default (all_months sin future): %s",
            train_months,
        )

    overlap = set(train_months).intersection(future_months)
    if overlap:
        logger.warning(
            "Meses presentes en train_months y future_months: %s (se usarán como train y future)",
            overlap,
        )

    # Subsets en Polars
    df_train_pl = df_pl.filter(pl.col(per_col).is_in(train_months))
    df_future_pl = df_pl.filter(pl.col(per_col).is_in(future_months))

    logger.info(
        "Train (train_months=%s): %s filas | Future (future_months=%s): %s filas",
        train_months,
        df_train_pl.height,
        future_months,
        df_future_pl.height,
    )

    # -----------------------------------------------------------------
    # UNDERSAMPLING DE CONTINUA EN TRAIN (Polars)
    # -----------------------------------------------------------------
    if target_multiclase in df_train_pl.columns:
        df_no_continua_pl = df_train_pl.filter(pl.col(target_multiclase) != "CONTINUA")
        df_continua_pl = df_train_pl.filter(pl.col(target_multiclase) == "CONTINUA")

        n_no_continua = df_no_continua_pl.height
        n_continua = df_continua_pl.height

        undersample_factor = max(zcfg.undersample_factor, 1.0)
        max_continua = int(undersample_factor * n_no_continua)

        if n_continua > max_continua and n_no_continua > 0:
            logger.info(
                "Aplicando undersampling de CONTINUA (factor=%.2f): original CONTINUA=%d, no_CONTINUA=%d, max_CONTINUA=%d",
                undersample_factor,
                n_continua,
                n_no_continua,
                max_continua,
            )
            df_continua_sample_pl = df_continua_pl.sample(
                n=max_continua,
                shuffle=True,
                seed=zcfg.base_seed,
            )
            df_train_pl = pl.concat(
                [df_no_continua_pl, df_continua_sample_pl],
                how="vertical"
            ).sample(
                fraction=1.0,
                shuffle=True,
                seed=zcfg.base_seed,
            )
            logger.info(
                "Post-undersampling (Polars): df_train=%s filas",
                df_train_pl.height,
            )
        else:
            logger.info(
                "No se aplica undersampling (factor=%.2f): CONTINUA=%d, no_CONTINUA=%d, max_CONTINUA=%d",
                undersample_factor,
                n_continua,
                n_no_continua,
                max_continua,
            )
    else:
        logger.warning(
            "No se encontró la columna de target multiclase '%s' en df_train; no se aplica undersampling.",
            target_multiclase,
        )

    # ---------------- Pasar a pandas SOLO train/future ----------------
    df_train = df_train_pl.to_pandas()
    df_future = df_future_pl.to_pandas()

    # ---------------- Matriz de features ----------------
    block_cols = {
        id_col,
        per_col,
        cfg.columns.target_column,
        cfg.columns.binary_target_col,
        cfg.columns.binary_target_gan,
        peso_col,
        "mprestamos_personales",
        "cprestamos_personales",
        "Master_Finiciomora",
        "Visa_Finiciomora",
        "mes_idx",
        "foto_mes_int",
    }

    X_train = df_train.drop(columns=[c for c in block_cols if c in df_train.columns])
    y_train = df_train[t_bin_gan].astype(int).to_numpy()
    w_train = df_train[peso_col].astype(float).to_numpy()

    valid_dtypes = (
        "int8", "int16", "int32", "int64",
        "uint8", "uint16", "uint32", "uint64",
        "float16", "float32", "float64",
        "bool"
    )
    X_train = X_train.select_dtypes(include=list(valid_dtypes)).copy()
    feature_names = list(X_train.columns)

    logger.info("X_train: %s, y_train: %s, features: %d", X_train.shape, y_train.shape, len(feature_names))

    lgb_train = lgb.Dataset(
        data=X_train,
        label=y_train,
        weight=w_train,
        free_raw_data=False
    )

    # ---------------- Seeds y ensemble ----------------
    seeds_list = _build_seeds_from_cfg(cfg, zcfg)
    models: List[lgb.Booster] = []

    for seed_value in seeds_list:
        logger.info("Entrenando modelo zLGBM con seed=%d ...", seed_value)

        params = build_lgbm_params(zcfg, seed_value)
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            num_boost_round=zcfg.num_iterations
        )
        models.append(model)

        model_str = model.model_to_string()
        model_txt_path = _join(zcfg.models_dir, f"zmodelo_seed{seed_value}.txt")
        _write_text(model_txt_path, model_str)
        logger.info("Modelo seed=%d guardado en: %s", seed_value, model_txt_path)

    if not models:
        raise RuntimeError("No se entrenó ningún modelo en el ensemble.")

    # ---------------- Predict ensemble en future_months ----------------
    logger.info("Armando dataset future para meses %s ...", future_months)

    X_future = df_future.drop(columns=[c for c in block_cols if c in df_future.columns])
    X_future = X_future[feature_names]

    all_probas = []
    for i, model in enumerate(models, start=1):
        logger.info("Predict con modelo %d/%d ...", i, len(models))
        p = model.predict(X_future)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        all_probas.append(p)

    proba_ensemble = np.mean(np.vstack(all_probas), axis=0)
    proba_ensemble = np.clip(proba_ensemble, 1e-15, 1 - 1e-15)

    tb_prediccion = pd.DataFrame({
        id_col: df_future[id_col].to_numpy(),
        per_col: df_future[per_col].to_numpy(),
        "prob": proba_ensemble,
    })

    # ---------------- Ganancia con merge explícito ----------------
    df_true_future = df_future[[id_col, per_col, t_bin_gan]].copy()
    ganancia_total = calcular_ganancia_ordenada(
        df_pred=tb_prediccion,
        df_true=df_true_future,
        cfg_cols=cfg.columns,
        ganancia_acierto=zcfg.ganancia_acierto,
        costo_estimulo=zcfg.costo_estimulo,
        top_n=zcfg.n_envios
    )
    logger.info("Ganancia total (ensemble): %.2f", ganancia_total)

    # ---------------- Top-N estilo Kaggle ----------------
    tb_prediccion = tb_prediccion.sort_values("prob", ascending=False).reset_index(drop=True)
    tb_prediccion["Predicted"] = 0
    top_n = min(zcfg.n_envios, len(tb_prediccion))
    tb_prediccion.loc[: top_n - 1, "Predicted"] = 1

    kaggle_path = os.path.join(
        zcfg.kaggle_dir,
        f"KA_{zcfg.experimento}_{top_n}_202105_minleaf50_20can_50seed_22month_gb01_prestamosless_driff1_0203less_final_julio_semillero_undersampling6_polar_drif_outl_less.csv"
    )
    _write_csv_df(tb_prediccion[[id_col, "Predicted"]], kaggle_path)
    logger.info("Archivo Kaggle (ensemble) guardado en: %s", kaggle_path)

    pred_detallado_path = os.path.join(
        zcfg.pred_dir,
        f"pred_zlgbm_{zcfg.experimento}_{top_n}_202105_minleaf50_20can_50seed_22month_gb01_prestamosless_driff1_0203less_final_julio_semillero_undersampling6_polar_drif_outl_less.csv"
    )
    _write_csv_df(tb_prediccion, pred_detallado_path)
    logger.info("Predicciones detalladas (ensemble) guardadas en: %s", pred_detallado_path)

    elapsed_sec = time.time() - t_start
    elapsed_min = elapsed_sec / 60.0
    logger.info(
        "Tiempo total de ejecución pipeline zLGBM: %.2f minutos (%.1f segundos)",
        elapsed_min,
        elapsed_sec,
    )

    return {
        "ganancia_total": ganancia_total,
        "n_models": len(models),
        "train_months": train_months,
        "future_months": future_months,
        "pred_detallado_path": pred_detallado_path,
        "kaggle_path": kaggle_path,
    }


if __name__ == "__main__":
    run_zlgbm_pipeline()
