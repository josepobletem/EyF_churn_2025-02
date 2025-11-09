import os
import logging
import pickle
import yaml
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from pydantic import BaseModel, Field, ValidationError
from pathlib import Path

# ======================
# logging
# ======================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ======================
# Config schemas
# ======================

class PathsConfig(BaseModel):
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str = Field(..., description="Dataset final con features")


class ColumnsConfig(BaseModel):
    id_column: str              # ej "numero_de_cliente"
    period_column: str          # ej "foto_mes"
    target_column_full: str = Field(
        "clase_ternaria",
        description="Columna target multiclase original (BAJA+1 / BAJA+2 / CONTINUA)"
    )
    
    # AHORA: elegimos directamente qué binaria queremos optimizar
    # Debe ser "clase_binaria1" o "clase_binaria2"
    binary_target_col: str = Field(
        "clase_binaria2",
        description=(
            "Cuál binaria usar como y:\n"
            " - clase_binaria1 = 1 si BAJA+2, 0 si no\n"
            " - clase_binaria2 = 1 si BAJA+1 o BAJA+2, 0 si CONTINUA"
        )
    )
    peso_col: str = "clase_peso"

    # nombre de la columna de pesos
    peso_col: str = Field(
        "clase_peso",
        description="Columna de pesos por fila"
    )


class TrainConfig(BaseModel):
    models_dir: str = Field(
        "models",
        description="Carpeta donde guardar mejor modelo y parámetros"
    )

    # split temporal
    train_months: list[int] = Field(
        default_factory=lambda: [201903, 201904, 201905, 201906, 201907, 201908, 201909,
                                 201910, 201911, 201912,
                                 202003, 202004, 202005, 202006, 202007, 202008, 202009,
                                 202010, 202011, 202012, 202101, 202102, 202103],
        description="Lista de foto_mes que se usan para train/CV"
    )
    test_month: int = Field(
        202104,
        description="Mes holdout (no se usa para entrenar, sólo referencia)"
    )

    # columnas a dropear por fuga/leak/etc
    drop_cols: list[str] = Field(
        default_factory=lambda: ["lag_3_ctrx_quarter","mprestamos_personales","cprestamos_personales"],
        description="Columnas a eliminar antes de entrenar"
    )

    # parámetros operativos de LGBM / CV
    objective: str = Field("binary", description="LightGBM objective")
    boosting_type: str = Field("gbdt", description="gbdt | dart | goss")
    n_estimators: int = Field(
        500,
        description="num_boost_round máximo en cv"
    )
    nfold: int = Field(
        15,
        description="k-fold CV estratificado"
    )
    seed: int = Field(
        12345,
        description="semilla reproducible"
    )
    n_trials: int = Field(
        2,
        description="cantidad de trials Optuna"
    )

    # negocio
    ganancia_acierto: float = Field(
        780000.0,
        description="Ganancia por acertar churn verdadero (TP)"
    )
    costo_estimulo: float = Field(
        20000.0,
        description="Costo por contactar un no-churn (FP)"
    )

    # pesos de clase_ternaria
    weight_baja2: float = Field(
        1.00002,
        description="Peso asignado a BAJA+2"
    )
    weight_baja1: float = Field(
        1.00001,
        description="Peso asignado a BAJA+1"
    )
    weight_continua: float = Field(
        1.0,
        description="Peso asignado a CONTINUA / resto"
    )


class FullConfig(BaseModel):
    paths: PathsConfig
    columns: ColumnsConfig
    train: TrainConfig | None = None


# ======================
# Config loader
# ======================

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


# ======================
# Feature engineering adicional en memoria
# ======================

def ensure_binarias_y_peso(df: pd.DataFrame, cfg_cols: ColumnsConfig, train_cfg: TrainConfig) -> pd.DataFrame:
    """
    Se asegura de que existan:
      - clase_binaria1: 1 si BAJA+2, 0 si no
      - clase_binaria2: 1 si (BAJA+1 o BAJA+2), 0 si CONTINUA
      - clase_peso:     pesos por fila según clase_ternaria
    Esto replica EXACTO tu snippet y tu lógica de peso.
    """
    df = df.copy()

    tcol = cfg_cols.target_column_full
    peso_col = cfg_cols.peso_col

    # ---- clase_binaria1 y clase_binaria2 ----
    if "clase_binaria1" not in df.columns:
        df["clase_binaria1"] = 0
    if "clase_binaria2" not in df.columns:
        df["clase_binaria2"] = 0

    # clase_binaria1 = 1 si BAJA+2, 0 si no
    df["clase_binaria1"] = np.where(df[tcol] == "BAJA+2", 1, 0)

    # clase_binaria2 = 1 si NO es CONTINUA (o sea BAJA+1 o BAJA+2), 0 si CONTINUA
    df["clase_binaria2"] = np.where(df[tcol] == "CONTINUA", 0, 1)

    # ---- clase_peso ----
    if peso_col not in df.columns:
        df[peso_col] = np.nan

    df.loc[df[tcol] == 'BAJA+2', peso_col] = train_cfg.weight_baja2
    df.loc[df[tcol] == 'BAJA+1', peso_col] = train_cfg.weight_baja1
    df.loc[~df[tcol].isin(['BAJA+1', 'BAJA+2']), peso_col] = train_cfg.weight_continua

    df[peso_col] = df[peso_col].fillna(train_cfg.weight_continua)

    return df


# ======================
# Métrica de negocio (gan_eval)
# ======================

def make_lgb_gan_eval(ganancia_acierto: float, costo_estimulo: float):
    """
    feval para LightGBM:
    - usa data.get_weight() para decidir TP/FP
    - ordena por y_pred desc
    - acumula ganancia incremental
    - devuelve el máximo acumulado
    """
    def lgb_gan_eval(y_pred, data: lgb.Dataset):
        weight = data.get_weight()

        ganancia = np.where(weight == 1.00002, ganancia_acierto, 0) - np.where(
            weight < 1.00002, costo_estimulo, 0
        )

        ganancia = ganancia[np.argsort(y_pred)[::-1]]
        ganancia = np.cumsum(ganancia)

        return "gan_eval", float(np.max(ganancia)), True

    return lgb_gan_eval


# ======================
# Dataset builder
# ======================
def build_train_dataset(df: pd.DataFrame, cfg: FullConfig, train_cfg: TrainConfig):
    """
    - filtra meses para train
    - dropea columnas peligrosas
    - arma X, y (binary_target_col), w (clase_peso)
    - filtra solo columnas numéricas/bool para LightGBM
    - devuelve (lgb.Dataset, X_train, y_train, w_train, feature_names)
    """
    per_col = cfg.columns.period_column
    peso_col = cfg.columns.peso_col
    target_bin_col = cfg.columns.binary_target_col  # ej "clase_binaria2"

    df_train = df[df[per_col].isin(train_cfg.train_months)].copy()

    if train_cfg.drop_cols:
        df_train = df_train.drop(
            columns=[c for c in train_cfg.drop_cols if c in df_train.columns],
            errors="ignore"
        )

    # columnas que NO deben entrar como features crudas
    block_cols = {
        cfg.columns.id_column,
        cfg.columns.period_column,
        cfg.columns.target_column_full,
        "clase_binaria1",
        "clase_binaria2",
        peso_col,
    }

    X_train = df_train.drop(
        columns=[c for c in block_cols if c in df_train.columns],
        errors="ignore"
    )

    # nos aseguramos de que exista la columna objetivo binaria
    if target_bin_col not in df_train.columns:
        raise KeyError(
            f"La columna binaria '{target_bin_col}' no existe ni después de ensure_binarias_y_peso(). "
            f"Chequeá config.columns.binary_target_col."
        )

    y_train = df_train[target_bin_col].astype(int).to_numpy()
    w_train = df_train[peso_col].astype(float).to_numpy()

    # 🔥 NUEVO: quedarnos sólo con columnas numéricas / bool
    valid_dtypes = ("int8","int16","int32","int64","uint8","uint16","uint32","uint64",
                    "float16","float32","float64","bool")
    X_train_numeric = X_train.select_dtypes(include=list(valid_dtypes)).copy()

    # guardo los nombres finales de features para inferencia futura
    feature_names = list(X_train_numeric.columns)

    lgb_train = lgb.Dataset(
        data=X_train_numeric,
        label=y_train,
        weight=w_train,
        free_raw_data=False
    )

    return lgb_train, X_train_numeric, y_train, w_train, feature_names



# ======================
# Optuna objective
# ======================

def make_objective(df_full: pd.DataFrame, cfg: FullConfig, train_cfg: TrainConfig):
    """
    objective(trial):
    - arma dataset de train
    - samplea hiperparámetros LightGBM con un espacio de búsqueda amplio
      (lo que pediste)
    - corre lgb.cv con métrica gan_eval
    - devuelve la mejor ganancia alcanzada en alguna iteración del CV
      (Optuna la maximiza)
    """

    def objective(trial: optuna.Trial):
        # dataset preparado (numéricas, weights, etc.)
        lgb_train, _, _, _, _ = build_train_dataset(df_full, cfg, train_cfg)

        # === Espacio de búsqueda ===
        # NOTA: usamos nombres equivalentes de LightGBM
        # min_data_in_leaf == min_child_samples
        # feature_fraction == colsample_bytree
        # bagging_fraction == subsample
        # bagging_freq se activa junto con bagging_fraction
        # min_sum_hessian_in_leaf -> min_sum_hessian_in_leaf
        # extra_trees -> extra_trees
        # scale_pos_weight -> scale_pos_weight (importante en clases desbalanceadas)
        params = {
            "objective": train_cfg.objective,
            "boosting_type": train_cfg.boosting_type,

            "learning_rate": trial.suggest_float(
                "learning_rate", 5e-4, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int(
                "num_leaves", 16, 512
            ),
            "max_depth": trial.suggest_int(
                "max_depth", -1, 16
            ),

            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 5, 2000
            ),
            # LightGBM alias: min_child_samples == min_data_in_leaf
            # Dejamos sólo uno para evitar conflictos:
            # "min_child_samples": ...  (lo omitimos para no pisarlo)

            "feature_fraction": trial.suggest_float(
                "feature_fraction", 0.4, 1.0
            ),  # alias de colsample_bytree
            "bagging_fraction": trial.suggest_float(
                "bagging_fraction", 0.5, 1.0
            ),  # alias de subsample
            "bagging_freq": trial.suggest_int(
                "bagging_freq", 0, 10
            ),

            "min_split_gain": trial.suggest_float(
                "min_split_gain", 0.0, 1.0
            ),
            "min_sum_hessian_in_leaf": trial.suggest_float(
                "min_sum_hessian_in_leaf", 1e-3, 10.0, log=True
            ),

            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-8, 10.0, log=True
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-8, 10.0, log=True
            ),

            "feature_fraction_bynode": trial.suggest_float(
                "feature_fraction_bynode", 0.4, 1.0
            ),

            "extra_trees": trial.suggest_categorical(
                "extra_trees", [False, True]
            ),

            "max_bin": trial.suggest_int(
                "max_bin", 63, 511
            ),

            "scale_pos_weight": trial.suggest_float(
                "scale_pos_weight", 0.5, 10.0, log=True
            ),

            # sugerencias de comportamiento general
            "first_metric_only": True,
            "boost_from_average": True,
            "feature_pre_filter": False,

            # métrica custom -> vamos con feval, así que no seteamos "metric": "custom".
            "metric": "None",

            "verbose": -1,
            "seed": train_cfg.seed,
        }

        callbacks = [
            lgb.log_evaluation(period=50),
            lgb.early_stopping(stopping_rounds=200, verbose=False),
        ]

        lgb_gan_eval = make_lgb_gan_eval(
            ganancia_acierto=train_cfg.ganancia_acierto,
            costo_estimulo=train_cfg.costo_estimulo
        )

        cv_results = lgb.cv(
            params=params,
            train_set=lgb_train,
            num_boost_round=train_cfg.n_estimators,
            feval=lgb_gan_eval,
            stratified=True,
            nfold=train_cfg.nfold,
            seed=train_cfg.seed,
            callbacks=callbacks
        )

        if not cv_results:
            raise RuntimeError("lgb.cv devolvió un dict vacío, no pude evaluar gan_eval.")

        # buscamos la serie más larga (nuestra métrica)
        best_key = max(cv_results.keys(), key=lambda k: len(cv_results[k]))
        history = cv_results[best_key]

        # score del trial = mejor valor visto en ese CV
        best_score_this_trial = max(history)

        return best_score_this_trial

    return objective


# ======================
# Final training con mejores params
# ======================

def train_final_model(df_full: pd.DataFrame, cfg: FullConfig, train_cfg: TrainConfig, best_params: dict):
    lgb_train, X_train_numeric, _, _, feature_names = build_train_dataset(df_full, cfg, train_cfg)

    callbacks = [
        lgb.log_evaluation(period=50),
        lgb.early_stopping(stopping_rounds=200, verbose=False),
    ]

    lgb_gan_eval = make_lgb_gan_eval(
        ganancia_acierto=train_cfg.ganancia_acierto,
        costo_estimulo=train_cfg.costo_estimulo
    )

    cv_results = lgb.cv(
        params=best_params,
        train_set=lgb_train,
        num_boost_round=train_cfg.n_estimators,
        feval=lgb_gan_eval,
        stratified=True,
        nfold=train_cfg.nfold,
        seed=train_cfg.seed,
        callbacks=callbacks
    )

    if not cv_results:
        raise RuntimeError("lgb.cv devolvió un dict vacío en train_final_model().")

    best_key = max(cv_results.keys(), key=lambda k: len(cv_results[k]))
    history = cv_results[best_key]

    best_iteration = len(history)
    best_gan_mean = max(history)
    best_gan_stdv = None  # no tenemos std separado con este truco

    final_model = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=best_iteration
    )

    return final_model, best_iteration, best_gan_mean, best_gan_stdv, feature_names

# ======================
# MAIN
# ======================

def run_optuna_and_train():
    # 1. cargar config y dataset
    cfg = load_config()
    train_cfg = cfg.train or TrainConfig()

    feature_path = cfg.paths.feature_dataset
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"No encontré el dataset de features {feature_path}")

    df = pd.read_csv(feature_path)

    # asegurarnos de tener clase_binaria1, clase_binaria2 y clase_peso
    df = ensure_binarias_y_peso(df, cfg.columns, train_cfg)

    logger.info(
        "Dataset cargado: %s filas, %s columnas",
        df.shape[0], df.shape[1]
    )
    logger.info(
        "Meses train=%s, mes test=%s",
        train_cfg.train_months,
        train_cfg.test_month
    )
    logger.info(
        "binary_target_col = %s",
        cfg.columns.binary_target_col
    )

    # 2. Optuna
    objective = make_objective(df, cfg, train_cfg)
    # Persistencia (sqlite local)
    db_path = Path.cwd() / "optimization_lgbm_modulo.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{db_path.as_posix()}"

    
    study = optuna.create_study(
        direction="maximize",
        study_name="lgbm_tpe_inteligente_2trial_comp2",
        storage=storage_url,
        load_if_exists=True,
        )
    study.optimize(objective, n_trials=train_cfg.n_trials)

    best_params = study.best_params
    best_value = study.best_value

    best_params_full = {
        "objective": train_cfg.objective,
        "boosting_type": train_cfg.boosting_type,
        "metric": "None",
        "verbose": -1,
        "seed": train_cfg.seed,
        **best_params,
    }

    logger.info("🏆 Mejores hiperparámetros Optuna: %s", best_params_full)
    logger.info("💰 Mejor gan_eval(mean) CV: %.2f", best_value)

    # 3. Entrenar modelo final con esos params
    final_model, best_iteration, best_gan_mean, best_gan_stdv, feature_names = train_final_model(
        df_full=df,
        cfg=cfg,
        train_cfg=train_cfg,
        best_params=best_params_full
    )

    # 4. Guardar
    os.makedirs(train_cfg.models_dir, exist_ok=True)

    params_path = os.path.join(train_cfg.models_dir, "best_params_comp2.yaml")
    model_path = os.path.join(train_cfg.models_dir, "best_model_comp2.pkl")

    with open(params_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
            {
                "best_params": best_params_full,
                "best_iteration": int(best_iteration),
                "gan_eval_best_cv": float(best_gan_mean),
                "gan_eval_stdv_cv": (None if best_gan_stdv is None else float(best_gan_stdv)),
                "best_gan_eval_from_optuna": float(best_value),

                "train_months": train_cfg.train_months,
                "test_month": int(train_cfg.test_month),
                "drop_cols": train_cfg.drop_cols,

                "ganancia_acierto": float(train_cfg.ganancia_acierto),
                "costo_estimulo": float(train_cfg.costo_estimulo),

                "weight_baja2": float(train_cfg.weight_baja2),
                "weight_baja1": float(train_cfg.weight_baja1),
                "weight_continua": float(train_cfg.weight_continua),

                "binary_target_col": cfg.columns.binary_target_col,
                "target_column_full": cfg.columns.target_column_full,
                "id_column": cfg.columns.id_column,
                "period_column": cfg.columns.period_column,

                "feature_names": feature_names,

                "n_trials": train_cfg.n_trials,
                "n_estimators_max": train_cfg.n_estimators,
                "nfold": train_cfg.nfold,
                "seed": train_cfg.seed,
            },
            f,
            sort_keys=False,
            allow_unicode=True,
        )

    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    logger.info("💾 Guardé params/ganancia en %s", params_path)
    logger.info("💾 Guardé el modelo final en %s", model_path)

    return {
        "best_params": best_params_full,
        "best_iteration": best_iteration,
        "gan_eval_best_cv": float(best_gan_mean),
        "gan_eval_stdv_cv": (None if best_gan_stdv is None else float(best_gan_stdv)),
        "best_gan_eval_from_optuna": float(best_value),
        "model_path": model_path,
        "params_path": params_path,
    }


if __name__ == "__main__":
    run_optuna_and_train()

