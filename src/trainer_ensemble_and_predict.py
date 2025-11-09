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

import os
import sys
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


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config schemas
# -----------------------------------------------------------------------------

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

    # meses para entrenar definidos por negocio
    train_months: List[int] = Field(
        default_factory=lambda: [202101, 202102, 202103, 202104],
        description="Meses que usamos efectivamente para entrenar"
    )
    test_month: int = Field(
        202104,
        description="Mes de holdout / validación temporal"
    )

    # columnas que hay que sacar (fuga/leak/etc.)
    drop_cols: List[str] = Field(
        default_factory=lambda: ["lag_3_ctrx_quarter"],
        description="Columnas a eliminar por fuga/leak/etc."
    )

    # negocio / pesos para las clases
    weight_baja2: float = Field(1.00002)
    weight_baja1: float = Field(1.00001)
    weight_continua: float = Field(1.0)

    # reproducibilidad base
    seed: int = Field(12345)

    # ensamble
    n_models: int = Field(
        10,
        description="Cantidad de modelos en el ensamble"
    )
    seeds: List[int] | None = Field(
        default=None,
        description="Lista explícita de semillas a usar. Si None se generan a partir de seed."
    )

    # umbral de decisión churn / fuga
    decision_threshold: float = Field(
        0.025,
        description="Umbral sobre la probabilidad promedio para marcar fuga (1)"
    )

    # métricas negocio (para ganancia)
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
    """
    Crea (o sobrescribe):
      - clase_binaria1 = 1 si BAJA+2, 0 si no
      - clase_binaria2 = 1 si BAJA+1 o BAJA+2, 0 si CONTINUA
      - clase_peso     usando weight_baja2 / weight_baja1 / weight_continua
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
    - filtra df a solo meses de train_months
    - dropea columnas con fuga
    - separa target binario
    - arma pesos
    - deja solo columnas numéricas/bool
    """
    per_col = cfg.columns.period_column
    peso_col = cfg.columns.peso_col
    target_bin_col = cfg.columns.binary_target_col

    df_train = df[df[per_col].isin(train_cfg.train_months)].copy()

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

    X = df_train.drop(
        columns=[c for c in block_cols if c in df_train.columns],
        errors="ignore"
    )

    if target_bin_col not in df_train.columns:
        raise KeyError(
            f"No encontré {target_bin_col} en df_train. "
            "Revisar binary_target_col en config.columns."
        )
    y = df_train[target_bin_col].astype(int).to_numpy()
    w = df_train[peso_col].astype(float).to_numpy()

    valid_dtypes = (
        "int8","int16","int32","int64",
        "uint8","uint16","uint32","uint64",
        "float16","float32","float64",
        "bool"
    )
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

    lgb_train = lgb.Dataset(
        data=X_train,
        label=y_train,
        weight=w_train,
        free_raw_data=False
    )

    model = lgb.train(
        params_with_seed,
        lgb_train,
        num_boost_round=best_iteration
    )

    y_pred_proba = model.predict(X_train)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)

    return model, y_pred_proba, params_with_seed


def train_final_ensemble(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Entrena N modelos con distintas semillas, promedia probabilidades y guarda metadata.
    """
    logger.info("🚂 Entrenamiento final (ensamble de múltiples semillas)...")

    cfg = load_config(config_path)
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")

    df_full = pd.read_csv(feature_path)
    df_full = ensure_binarias_y_peso(df_full, cfg.columns, train_cfg)

    X_train, y_train, w_train, feature_names = build_train_matrix(df_full, cfg, train_cfg)

    logger.info("Shape train X: %s, y: %s", X_train.shape, y_train.shape)
    logger.info("Meses usados para entrenar: %s", train_cfg.train_months)
    logger.info("Mes holdout reservado (no entrenado): %s", train_cfg.test_month)

    params_path = os.path.join(train_cfg.models_dir, "best_params_v2.yaml")
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"No encontré {params_path}. Corré primero optimizer.py.")
    with open(params_path, "r", encoding="utf-8") as f:
        best_info = yaml.safe_load(f)

    best_params = best_info["best_params"]
    #best_iteration = int(best_info.get("best_iteration", 500))  # fallback
    best_iteration = 500 # fallback

    logger.info("Hiperparámetros base cargados de optimizer: %s", best_params)
    logger.info("Usando best_iteration = %d", best_iteration)

    seeds_list = _generate_seeds(train_cfg)
    logger.info("Semillas usadas en el ensamble: %s", seeds_list)

    os.makedirs(train_cfg.models_dir, exist_ok=True)
    ens_dir = os.path.join(train_cfg.models_dir, "ensamble-4")
    os.makedirs(ens_dir, exist_ok=True)

    all_model_paths: List[str] = []
    all_probas: List[np.ndarray] = []

    for seed_value in seeds_list:
        logger.info("Entrenando modelo con seed=%d ...", seed_value)
        model, y_pred_proba_single, _ = _train_single_model(
            X_train, y_train, w_train, best_params, best_iteration, seed_value=seed_value
        )

        model_path = os.path.join(ens_dir, f"final_model_seed{seed_value}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info("💾 Modelo guardado en %s", model_path)

        all_model_paths.append(model_path)
        all_probas.append(y_pred_proba_single)

    y_pred_proba_ensemble = np.mean(np.vstack(all_probas), axis=0)
    y_pred_proba_ensemble = np.clip(y_pred_proba_ensemble, 1e-15, 1 - 1e-15)

    logloss_in = log_loss(y_train, y_pred_proba_ensemble)
    thr = float(train_cfg.decision_threshold)
    y_pred_label_ensemble = (y_pred_proba_ensemble >= thr).astype(int)
    cm_in = confusion_matrix(y_train, y_pred_label_ensemble).tolist()

    ensemble_meta_path = os.path.join(train_cfg.models_dir, "ensamble-4/final_ensemble_metadata.yaml")

    with open(ensemble_meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
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
            f,
            sort_keys=False,
            allow_unicode=True,
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
    """
    Filtra solo el mes, deja X_score numérica, y devuelve info_df (id, mes, y binaria si existe).
    """
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


def load_ensemble_models_and_meta(cfg: FullConfig) -> Tuple[List[Any], Dict[str, Any], str]:
    """
    Carga todos los modelos del ensamble (paths en final_ensemble_metadata.yaml) y devuelve:
    - lista de boosters
    - metadata yaml
    - path del metadata
    """
    train_cfg = cfg.train or TrainConfig()
    meta_path = os.path.join(train_cfg.models_dir, "ensamble-4/final_ensemble_metadata.yaml")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No encontré metadata del ensamble en {meta_path}. Corré entrenamiento primero.")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    model_paths = meta.get("model_paths", [])
    if not model_paths:
        raise ValueError("final_ensemble_metadata.yaml no contiene 'model_paths'.")

    models = []
    for mp in model_paths:
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Falta modelo del ensamble: {mp}")
        with open(mp, "rb") as f:
            models.append(pickle.load(f))

    return models, meta, meta_path


def score_month_ensemble(
    month_to_score: int,
    threshold: float = 0.025,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Predice con el ensamble (promedio de probabilidades) para un mes dado.
    """
    cfg = load_config(config_path)
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré {feature_path}")
    df_full = pd.read_csv(feature_path)

    # recrear columnas derivadas por consistencia con entrenamiento
    df_full = ensure_binarias_y_peso(df_full, cfg.columns, train_cfg)

    X_score, info_df, _ = build_scoring_matrix(df_full, cfg, month_to_score=month_to_score)

    logger.info("Mes %s -> scoring rows: %s, features: %s", month_to_score, X_score.shape[0], X_score.shape[1])

    # cargar ensamble y metadata
    models, meta_yaml, _ = load_ensemble_models_and_meta(cfg)
    logger.info("Modelos cargados del ensamble: %d", len(models))

    feature_names_train = meta_yaml.get("feature_names", None)
    if feature_names_train is None:
        raise ValueError("final_ensemble_metadata.yaml no contiene 'feature_names'.")

    # Alinear columnas exactamente como en entrenamiento
    for col in feature_names_train:
        if col not in X_score.columns:
            X_score[col] = 0.0
    X_score = X_score[feature_names_train]

    # predecir con cada modelo y promediar
    probas = []
    for i, model in enumerate(models, start=1):
        p = model.predict(X_score)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        probas.append(p)
        logger.debug("Modelo %d listo", i)

    proba_mean = np.mean(np.vstack(probas), axis=0)
    proba_mean = np.clip(proba_mean, 1e-15, 1 - 1e-15)

    pred_flag = (proba_mean >= threshold).astype(int)

    # calcular ganancia si hay verdad terreno binaria disponible
    bin_col = cfg.columns.binary_target_col
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
    parser.add_argument(
        "--train",
        action="store_true",
        help="Si se especifica, entrena el ensamble antes de predecir (si --mes también está presente)."
    )
    parser.add_argument(
        "--mes",
        type=int,
        help="Mes (foto_mes) a puntuar, ej 202104"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.025,
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

    # Entrenamiento si se pidió
    if args.train:
        train_result = train_final_ensemble(config_path=args.config)
        logger.info("Entrenamiento de ensamble finalizado. LogLoss in-sample: %.6f",
                    train_result["logloss_in_sample_ensemble"])

    # Scoring si se especificó mes
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

        # Guardar outputs
        out_dir = "predicciones/"
        os.makedirs(out_dir, exist_ok=True)

        out_path_full = os.path.join(
            out_dir,
            f"pred_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_25.csv"
        )
        df_pred.to_csv(out_path_full, index=False)
        logger.info("Guardé scoring detallado en %s", out_path_full)

        # CSV simple con id_column real del config
        id_col = load_config(args.config).columns.id_column
        out_path_simple = os.path.join(
            out_dir,
            f"pred_simple_{result['mes']}_thr{args.threshold:.4f}_ensemble_500_25.csv"
        )
        df_simple[[id_col, "Predicted"]].to_csv(out_path_simple, index=False)
        logger.info("Guardé scoring simple en %s", out_path_simple)
        
        # NEW: imprimir cuántas predicciones quedaron en 1 en el archivo de salida
        n_pred1_simple = int((df_simple["Predicted"] == 1).sum())  # NEW
        logger.info("Predicciones=1 en %s: %d de %d (%.2f%%)", out_path_simple, n_pred1_simple, n_total, pct_pred1)
        
    # Si no se pidió ni train ni mes, mostrar ayuda
    if (not args.train) and (args.mes is None):
        logger.info("No se especificó --train ni --mes. Mostrando ayuda:")
        _ = _parse_args(["-h"])


if __name__ == "__main__":
    main()
