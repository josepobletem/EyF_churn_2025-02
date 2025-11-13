# src/trainer_ensemble_and_predict.py

"""
Entrena un ensamble de LightGBM con los mejores hiperparámetros hallados por optimizer.py
y permite predecir (promediando el ensamble) de inmediato desde la CLI.

CLI:
  - Entrenar solo:
      python -m src.trainer_ensemble_and_predict --train

  - Predecir solo (requiere ensamble entrenado previamente):
      python -m src.trainer_ensemble_and_predict --mes 202104 --threshold 0.025

  - Entrenar y predecir en el mismo paso:
      python -m src.trainer_ensemble_and_predict --train --mes 202104 --threshold 0.025

Outputs:
  - models/final_model_seed<seed>.pkl        (uno por semilla)
  - models/final_ensemble_metadata.yaml      (métricas y metadata de ensamble)
  - predicciones/pred_<mes>_thr<thr>_ensemble.csv          (detallado)
  - predicciones/pred_simple_<mes>_thr<thr>_ensemble.csv   (simple negocio)
"""

# src/trainer_ensemble_and_predict_gcp.py

import os
import sys
import io
import yaml
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import log_loss, confusion_matrix
import lightgbm as lgb

# === NUEVO: soporte GCS con fsspec ===
import fsspec

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers GCS / Local
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Helpers GCS / Local (ADC / gcsfs)
# -----------------------------------------------------------------------------
import os, io, pickle, yaml
import fsspec
import pandas as pd

def _is_gcs(path: str) -> bool:
    return str(path).startswith("gs://")

def _gcs_fs():
    # Usará Application Default Credentials (ADC):
    # - gcloud auth application-default login (usuario)
    # - o service account si la VM tiene ADC configurado
    return fsspec.filesystem("gcs", token="google_default")  # sin token explícito

def _gcs_open(path: str, mode: str):
    return _gcs_fs().open(path, mode)

def _join(base: str, *parts: str) -> str:
    if _is_gcs(base):
        return "/".join([base.rstrip("/")] + [p.strip("/") for p in parts])
    return os.path.join(base, *parts)

def _path_exists(path: str) -> bool:
    if _is_gcs(path):
        return _gcs_fs().exists(path)
    return os.path.exists(path)

def _read_yaml(path: str) -> dict:
    if _is_gcs(path):
        with _gcs_open(path, "r") as f:
            return yaml.safe_load(f)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_pickle(path: str):
    if _is_gcs(path):
        with _gcs_open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)

def _write_pickle(obj, path: str):
    if _is_gcs(path):
        buf = io.BytesIO()
        pickle.dump(obj, buf)
        with _gcs_open(path, "wb") as f:
            f.write(buf.getvalue())
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _write_yaml(data: dict, path: str):
    txt = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    if _is_gcs(path):
        with _gcs_open(path, "w") as f:
            f.write(txt)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def _write_csv_df(df: pd.DataFrame, path: str):
    if _is_gcs(path):
        # Escribir CSV textual a GCS usando file-like
        with _gcs_open(path, "w") as f:
            df.to_csv(f, index=False)
        return
    os.makedirs(os.path.dirname(path), exist_ok=True    )
    df.to_csv(path, index=False)

def _read_parquet(path: str) -> pd.DataFrame:
    if _is_gcs(path):
        # Con fsspec/ADC no necesitas pasar storage_options explícito
        return pd.read_parquet(path)
    return pd.read_parquet(path)

def _pred_dir_from_models_dir(models_dir: str) -> str:
    """
    Si models_dir = gs://.../eyf/models -> gs://.../eyf/prediccion
    Si no termina en /models, cuelga prediccion al mismo prefijo.
    En local: "predicciones"
    """
    if _is_gcs(models_dir):
        base = models_dir.rstrip("/")
        if base.endswith("/models"):
            return base[:-len("/models")] + "/prediccion"
        return _join(base, "prediccion")
    return "predicciones"


# -----------------------------------------------------------------------------
# Config schemas
# -----------------------------------------------------------------------------
class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str

class ColumnsConfig(BaseModel):
    id_column: str
    period_column: str
    target_column_full: str = Field(
        "clase_ternaria",
        description="Target multiclase original (BAJA+1 / BAJA+2 / CONTINUA)"
    )
    binary_target_col: str = Field(
        "clase_binaria2",
        description=(
            "Target binario final: "
            "clase_binaria1 = 1 si BAJA+2, 0 si no; "
            "clase_binaria2 = 1 si BAJA+1 o BAJA+2, 0 si CONTINUA"
        )
    )
    peso_col: str = Field(
        "clase_peso",
        description="Columna de pesos que se usa como weight"
    )

class TrainConfig(BaseModel):
    models_dir: str = Field("gs://jose_poblete_bukito3/eyf/models")
    train_months: List[int] = Field(
        default_factory=lambda: [#201809, 201810, 201811, 201812,
                                 #201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908, 201909, 
                                 201910, 201911, 201912,
                                 202001, 202002, 202003, 202004, 202005, 202006, 202007, 202008, 202009, 202010, 202011, 202012,
                                 202101, 202102, 202103, 202104]
    )
    test_month: int = Field(202104)
    drop_cols: List[str] = Field(default_factory=lambda: ["lag_3_ctrx_quarter"])
    weight_baja2: float = Field(1.00002)
    weight_baja1: float = Field(1.00001)
    weight_continua: float = Field(1.0)
    seed: int = Field(12345)
    n_models: int = Field(10)
    seeds: List[int] | None = Field(default=None)
    decision_threshold: float = Field(0.025)
    ganancia_acierto: float = Field(780000.0)
    costo_estimulo: float = Field(20000.0)

class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: TrainConfig | None = None

# -----------------------------------------------------------------------------
# Helpers de config
# -----------------------------------------------------------------------------
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
# Feature engineering de negocio (binarias + pesos)
# -----------------------------------------------------------------------------
def ensure_binarias_y_peso(df: pd.DataFrame, cfg_cols: ColumnsConfig, train_cfg: TrainConfig) -> pd.DataFrame:
    df = df.copy()
    tcol = cfg_cols.target_column_full
    peso_col = cfg_cols.peso_col
    df["clase_binaria1"] = np.where(df[tcol] == "BAJA+2", 1, 0)
    df["clase_binaria2"] = np.where(df[tcol] == "CONTINUA", 0, 1)
    if peso_col not in df.columns:
        df[peso_col] = np.nan
    df.loc[df[tcol] == "BAJA+2", peso_col] = train_cfg.weight_baja2
    df.loc[df[tcol] == "BAJA+1", peso_col] = train_cfg.weight_baja1
    df.loc[~df[tcol].isin(["BAJA+1", "BAJA+2"]), peso_col] = train_cfg.weight_continua
    df[peso_col] = df[peso_col].fillna(train_cfg.weight_continua)
    return df

def build_train_matrix(df: pd.DataFrame, cfg: FullConfig, train_cfg: TrainConfig):
    per_col = cfg.columns.period_column
    peso_col = cfg.columns.peso_col
    target_bin_col = cfg.columns.binary_target_col
    df_train = df[df[per_col].isin(train_cfg.train_months)].copy()
    print(f"Cantidad de filas en el dataset de entrenamiento: {df_train.shape[0]}")
    if train_cfg.drop_cols:
        df_train = df_train.drop(
            columns=[c for c in train_cfg.drop_cols if c in df_train.columns],
            errors="ignore"
        )
    block_cols = {
        cfg.columns.id_column,
        cfg.columns.period_column,
        cfg.columns.target_column_full,
        "clase_binaria1",
        "clase_binaria2",
        peso_col,
    }
    X = df_train.drop(columns=[c for c in block_cols if c in df_train.columns], errors="ignore")
    if target_bin_col not in df_train.columns:
        raise KeyError(f"No encontré {target_bin_col} en df_train. Revisar binary_target_col en config.columns.")
    y = df_train[target_bin_col].astype(int).to_numpy()
    w = df_train[peso_col].astype(float).to_numpy()
    valid_dtypes = ("int8","int16","int32","int64","uint8","uint16","uint32","uint64",
                    "float16","float32","float64","bool")
    X_numeric = X.select_dtypes(include=list(valid_dtypes)).copy()
    feature_names = list(X_numeric.columns)
    return X_numeric, y, w, feature_names

# -----------------------------------------------------------------------------
# Entrenamiento (ensamble)
# -----------------------------------------------------------------------------
def _generate_seeds(train_cfg: TrainConfig) -> List[int]:
    if train_cfg.seeds is not None and len(train_cfg.seeds) > 0:
        return train_cfg.seeds[: train_cfg.n_models]
    base = train_cfg.seed
    return [base + i for i in range(train_cfg.n_models)]

def _train_single_model(X_train, y_train, w_train, best_params, best_iteration, seed_value: int):
    params_with_seed = dict(best_params)
    params_with_seed["seed"] = seed_value
    params_with_seed["feature_fraction_seed"] = seed_value
    params_with_seed["bagging_seed"] = seed_value
    lgb_train = lgb.Dataset(data=X_train, label=y_train, weight=w_train, free_raw_data=False)
    model = lgb.train(params_with_seed, lgb_train, num_boost_round=best_iteration)
    y_pred_proba = model.predict(X_train)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    return model, y_pred_proba, params_with_seed

def train_final_ensemble(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    logger.info("🚂 Entrenamiento final (ensamble de múltiples semillas)...")
    cfg = load_config(config_path)
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not _path_exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")
    # ⚠️ Ahora leemos PARQUET (tu pipeline guarda parquet en GCS)
    df_full = _read_parquet(feature_path)

    df_full = ensure_binarias_y_peso(df_full, cfg.columns, train_cfg)
    X_train, y_train, w_train, feature_names = build_train_matrix(df_full, cfg, train_cfg)

    logger.info("Shape train X: %s, y: %s", X_train.shape, y_train.shape)
    logger.info("Meses usados para entrenar: %s", train_cfg.train_months)
    logger.info("Mes holdout reservado (no entrenado): %s", train_cfg.test_month)

    # === cargar hiperparámetros desde GCS/local ===
    # Se espera que en config.yaml -> train.models_dir = gs://.../eyf/models
    params_path = _join(train_cfg.models_dir, "best_params_comp2.yaml")
    if not _path_exists(params_path):
        raise FileNotFoundError(f"No encontré {params_path}. Corré primero optimizer.py.")
    best_info = _read_yaml(params_path)

    best_params = best_info["best_params"]
    best_iteration = int(best_info.get("best_iteration", 500))  # fallback

    logger.info("Hiperparámetros base cargados de optimizer: %s", best_params)
    logger.info("Usando best_iteration = %d", best_iteration)

    seeds_list = _generate_seeds(train_cfg)
    logger.info("Semillas usadas en el ensamble: %s", seeds_list)

    ens_dir = _join(train_cfg.models_dir, "ensamble-4")  # puede ser GCS o local
    all_model_paths: List[str] = []
    all_probas: List[np.ndarray] = []

    for seed_value in seeds_list:
        logger.info("Entrenando modelo con seed=%d ...", seed_value)
        model, y_pred_proba_single, _ = _train_single_model(
            X_train, y_train, w_train, best_params, best_iteration, seed_value=seed_value
        )

        model_path = _join(ens_dir, f"final_model_seed{seed_value}.pkl")
        _write_pickle(model, model_path)
        logger.info("💾 Modelo guardado en %s", model_path)

        all_model_paths.append(model_path)
        all_probas.append(y_pred_proba_single)

    y_pred_proba_ensemble = np.mean(np.vstack(all_probas), axis=0)
    y_pred_proba_ensemble = np.clip(y_pred_proba_ensemble, 1e-15, 1 - 1e-15)

    logloss_in = log_loss(y_train, y_pred_proba_ensemble)
    thr = float(train_cfg.decision_threshold)
    y_pred_label_ensemble = (y_pred_proba_ensemble >= thr).astype(int)
    cm_in = confusion_matrix(y_train, y_pred_label_ensemble).tolist()

    ensemble_meta_path = _join(train_cfg.models_dir, "ensamble-4/final_ensemble_metadata.yaml")
    _write_yaml(
        {
            "train_months": train_cfg.train_months,
            "test_month": int(train_cfg.test_month),
            "binary_target_col": cfg.columns.binary_target_col,
            "period_column": cfg.columns.period_column,
            "id_column": cfg.columns.id_column,
            "best_iteration": int(best_iteration),
            "seeds_used": seeds_list,
            "model_paths": all_model_paths,
            "decision_threshold": thr,
            "logloss_in_sample_ensemble": float(logloss_in),
            "confusion_matrix_in_sample_ensemble": cm_in,
            "feature_names": list(X_train.columns),
        },
        ensemble_meta_path
    )

    logger.info("📊 LogLoss in-sample (ensamble): %.6f", logloss_in)
    logger.info("📊 Matriz de confusión in-sample (thr=%.4f): %s", thr, cm_in)
    logger.info("💾 Metadata de ensamble guardada en %s", ensemble_meta_path)

    return {
        "model_paths": all_model_paths,
        "ensemble_metadata_path": ensemble_meta_path,
        "logloss_in_sample_ensemble": logloss_in,
    }

# -----------------------------------------------------------------------------
# Carga de ensamble + helpers de scoring
# -----------------------------------------------------------------------------
def calcular_ganancia(y_true_bin: np.ndarray,
                      y_pred_bin: np.ndarray,
                      ganancia_acierto: float,
                      costo_estimulo: float) -> float:
    tp = np.sum((y_pred_bin == 1) & (y_true_bin == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true_bin == 0))
    gan = tp * ganancia_acierto - fp * costo_estimulo
    return float(gan)

def build_scoring_matrix(
    df_full: pd.DataFrame,
    cfg: FullConfig,
    month_to_score: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    id_col = cfg.columns.id_column
    per_col = cfg.columns.period_column
    tgt_full = cfg.columns.target_column_full
    peso_col = cfg.columns.peso_col
    bin1 = "clase_binaria1"
    bin2 = cfg.columns.binary_target_col

    df_mes = df_full[df_full[per_col] == month_to_score].copy()
    if df_mes.empty:
        raise ValueError(f"No hay filas para el mes {month_to_score} en el dataset de features")

    cols_info = [c for c in [id_col, per_col, bin2] if c in df_mes.columns]
    info_df = df_mes[cols_info].reset_index(drop=True)

    block_cols = {id_col, per_col, tgt_full, peso_col, bin1, bin2}
    drop_cols_cfg = []
    if cfg.train and cfg.train.drop_cols:
        drop_cols_cfg = cfg.train.drop_cols

    cols_to_remove = list(block_cols.union(drop_cols_cfg))
    X_score_raw = df_mes.drop(columns=[c for c in cols_to_remove if c in df_mes.columns],
                              errors="ignore")

    valid_dtypes = ("int8","int16","int32","int64","uint8","uint16","uint32","uint64",
                    "float16","float32","float64","bool")
    X_score = X_score_raw.select_dtypes(include=list(valid_dtypes)).copy()

    if X_score.shape[1] == 0:
        raise ValueError(
            "Después de filtrar columnas y tipos, no quedaron features numéricas "
            f"para el mes {month_to_score}. Revisá drop_cols/config."
        )

    feature_names = list(X_score.columns)
    return X_score, info_df, feature_names

def load_ensemble_models_and_meta(cfg: FullConfig) -> Tuple[List[Any], Dict[str, Any], str]:
    train_cfg = cfg.train or TrainConfig()
    meta_path = _join(train_cfg.models_dir, "ensamble-4/final_ensemble_metadata.yaml")
    if not _path_exists(meta_path):
        raise FileNotFoundError(f"No encontré metadata del ensamble en {meta_path}. Corré entrenamiento primero.")

    meta = _read_yaml(meta_path)
    model_paths = meta.get("model_paths", [])
    if not model_paths:
        raise ValueError("final_ensemble_metadata.yaml no contiene 'model_paths'.")

    models = []
    for mp in model_paths:
        if not _path_exists(mp):
            raise FileNotFoundError(f"Falta modelo del ensamble: {mp}")
        models.append(_read_pickle(mp))

    return models, meta, meta_path

def score_month_ensemble(
    month_to_score: int,
    threshold: float = 0.025,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    cfg = load_config(config_path)
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not _path_exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")
    df_full = _read_parquet(feature_path)

    df_full = ensure_binarias_y_peso(df_full, cfg.columns, train_cfg)
    X_score, info_df, _ = build_scoring_matrix(df_full, cfg, month_to_score=month_to_score)

    logger.info("Mes %s -> scoring rows: %s, features: %s", month_to_score, X_score.shape[0], X_score.shape[1])

    models, meta_yaml, _ = load_ensemble_models_and_meta(cfg)
    logger.info("Modelos cargados del ensamble: %d", len(models))

    feature_names_train = meta_yaml.get("feature_names", None)
    if feature_names_train is None:
        raise ValueError("final_ensemble_metadata.yaml no contiene 'feature_names'.")

    for col in feature_names_train:
        if col not in X_score.columns:
            X_score[col] = 0.0
    X_score = X_score[feature_names_train]

    probas = []
    for i, model in enumerate(models, start=1):
        p = model.predict(X_score)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        probas.append(p)
        logger.debug("Modelo %d listo", i)

    proba_mean = np.mean(np.vstack(probas), axis=0)
    proba_mean = np.clip(proba_mean, 1e-15, 1 - 1e-15)
    pred_flag = (proba_mean >= threshold).astype(int)

    bin_col = cfg.columns.binary_target_gan
    if bin_col in info_df.columns:
        y_true_bin = info_df[bin_col].astype(int).to_numpy()
    else:
        y_true_bin = np.zeros(len(pred_flag), dtype=int)

    gan_total = calcular_ganancia(
        y_true_bin=y_true_bin,
        y_pred_bin=pred_flag,
        ganancia_acierto=getattr(train_cfg, "ganancia_acierto", 780000.0),
        costo_estimulo=getattr(train_cfg, "costo_estimulo", 20000.0),
    )

    id_col = cfg.columns.id_column
    per_col = cfg.columns.period_column

    out = pd.DataFrame({
        id_col: info_df[id_col].to_numpy(),
        per_col: info_df[per_col].to_numpy(),
        "proba_modelo": proba_mean,
        "pred_flag": pred_flag,
    })
    out["rank_desc"] = out["proba_modelo"].rank(ascending=False, method="first").astype(int)
    out = out.sort_values("proba_modelo", ascending=False).reset_index(drop=True)

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
        description="Entrena ensamble LightGBM y/o realiza scoring mensual promediando el ensamble"
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--mes", type=int, help="Mes (foto_mes) a puntuar, ej 202104")
    parser.add_argument("--threshold", type=float, default=0.025)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args(argv)

def main(argv=None):
    args = _parse_args(argv)

    if args.train:
        train_result = train_final_ensemble(config_path=args.config)
        logger.info("Entrenamiento de ensamble finalizado. LogLoss in-sample: %.6f",
                    train_result["logloss_in_sample_ensemble"])

    if args.mes is not None:
        result = score_month_ensemble(
            month_to_score=args.mes,
            threshold=args.threshold,
            config_path=args.config,
        )
        logger.info("corrio la funcion score_month_ensemble")
        df_pred = result["df_pred"]
        df_simple = result["df_simple"]
        gan_total = result["ganancia_total"]

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

        # === SALIDA: si models_dir es GCS, escribir en gs://.../eyf/prediccion/ ===
        pred_dir = _pred_dir_from_models_dir((load_config(args.config).train or TrainConfig()).models_dir)
        if _is_gcs(pred_dir):
            out_path_full = _join(
                pred_dir,
                f"pred_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_20_{args.mes}.csv"
            )
            out_path_simple = _join(
                pred_dir,
                f"pred_simple_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_20_{args.mes}.csv"
            )
            _write_csv_df(df_pred, out_path_full)
            _write_csv_df(df_simple[[load_config(args.config).columns.id_column, "Predicted"]], out_path_simple)
        else:
            os.makedirs(pred_dir, exist_ok=True)
            out_path_full = os.path.join(
                pred_dir,
                f"pred_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_20_{args.mes}.csv"
            )
            df_pred.to_csv(out_path_full, index=False)
            id_col = load_config(args.config).columns.id_column
            out_path_simple = os.path.join(
                pred_dir,
                f"pred_simple_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_20_{args.mes}.csv"
            )
            df_simple[[id_col, "Predicted"]].to_csv(out_path_simple, index=False)

        logger.info("Guardé scoring detallado en %s", out_path_full)
        logger.info("Guardé scoring simple en %s", out_path_simple)

        # extra: imprimir cuántas predicciones=1 en el archivo de salida
        n_pred1_simple = int((df_simple["Predicted"] == 1).sum())
        logger.info("Predicciones=1 en %s: %d de %d (%.2f%%)", out_path_simple, n_pred1_simple, n_total, pct_pred1)

    if (not args.train) and (args.mes is None):
        logger.info("No se especificó --train ni --mes. Mostrando ayuda:")
        _ = _parse_args(["-h"])

if __name__ == "__main__":
    main()
