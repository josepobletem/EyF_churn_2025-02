# -*- coding: utf-8 -*-
"""
feature_engineering_polars
==========================

Genera variables derivadas (feature engineering) usando Polars:

- Sumas horizontales (saldos, consumos, inversiones, etc.).
- Suavizados exponenciales (EMA) y máximos/mínimos rolling
  para algunas columnas agregadas.
- Lags y deltas (orden 1 y 2) para TODAS las variables numéricas
  excepto las columnas clave (id, periodo, target) y las nuevas
  features de EMA / max / min.

Flujo:
1. Carga el dataset procesado con target (paths.processed_dataset en config.yaml).
2. Aplica feature engineering en Polars.
3. Filtra solo periodos >= 201907.
4. Guarda el dataset final en paths.feature_dataset (CSV o Parquet).

Uso:
    python -m src.feature_engineering_polars
"""

import os
import logging
from typing import List

import polars as pl
import yaml
from pydantic import BaseModel, Field, ValidationError
import gcsfs  # para leer/escribir gs://

# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Helpers para paths
# -------------------------------------------------

def is_gcs_path(path: str) -> bool:
    """Detecta si el path es de GCS (gs://...)."""
    return path.startswith("gs://")


# -------------------------------------------------
# Pydantic config models
# -------------------------------------------------

class PathsConfig(BaseModel):
    """
    PathsConfig
    -----------
    Rutas relevantes para el pipeline de datos.
    """
    raw_dataset: str
    processed_dataset: str
    feature_dataset: str = Field(
        ...,
        description="Ruta donde se guarda el dataset final con features para modelar."
    )


class ColumnsConfig(BaseModel):
    """
    ColumnsConfig
    -------------
    Columnas claves del dataset de clientes.
    """
    id_column: str
    period_column: str
    target_column: str = Field(
        "clase_ternaria",
        description="Nombre de la columna objetivo (ej. 'clase_ternaria')."
    )


class FeaturesConfig(BaseModel):
    """
    FeaturesConfig
    --------------
    Se deja por compatibilidad, aunque no sea estrictamente necesario
    en esta versión Polars.
    """
    base_table_name: str = Field(
        "base_clientes",
        description="Nombre lógico del dataset base (solo informativo aquí)."
    )
    steps: List[str] = Field(
        default_factory=list,
        description="No se usan en la versión Polars, pero se aceptan del YAML."
    )


class FullConfig(BaseModel):
    """
    FullConfig
    ----------
    Esquema completo esperado en config/config.yaml.
    """
    paths: PathsConfig
    columns: ColumnsConfig
    features: FeaturesConfig | None = None  # opcional para esta versión


def load_config(path: str = "config/config.yaml") -> FullConfig:
    """
    Cargar y validar la configuración del proyecto usando Pydantic.
    """
    logger.info("Cargando configuración de feature engineering (Polars) desde %s ...", path)

    if not os.path.exists(path):
        logger.error("No se encontró el archivo de configuración: %s", path)
        raise FileNotFoundError(f"No encontré el archivo de configuración {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    try:
        cfg = FullConfig(**raw_cfg)
    except ValidationError as e:
        logger.error("Error validando config.yaml con Pydantic:\n%s", e)
        raise

    logger.debug("Config validada (feature engineering Polars): %s", cfg)
    return cfg


# -------------------------------------------------
# Definición de grupos de columnas
# -------------------------------------------------

# Totales de saldos en pesos
SALDOS_PESOS = [
    "mcuenta_corriente",
    "mcaja_ahorro",
    "mcuenta_corriente_adicional",
    "mcaja_ahorro_adicional",
    "mcuentas_saldo",
]

# Totales de saldos en dólares
SALDOS_DOLARES = [
    "mcaja_ahorro_dolares",
]

# Consumo tarjetas pesos
CONSUMO_TARJETAS_PESOS = [
    "mtarjeta_visa_consumo",
    "mtarjeta_master_consumo",
]

# Consumo tarjetas dólares
CONSUMO_TARJETAS_DOLARES = [
    "Visa_mconsumosdolares",
    "Master_mconsumosdolares",
]

# Pagos recurrentes pesos
PAGOS_RECURRENTE_PESOS = [
    "mpagodeservicios",
    "mpagomiscuentas",
    "mcuenta_debitos_automaticos",
    "mttarjeta_visa_debitos_automaticos",
    "mttarjeta_master_debitos_automaticos",
]

# Sumas de productos de inversión y plazos fijos pesos
INVERSIONES_PESOS = [
    "mplazo_fijo_pesos",
    "minversion1_pesos",
    "minversion2",
]

# Sumas de productos de inversión y plazos fijos dólares
INVERSIONES_DOLARES = [
    "mplazo_fijo_dolares",
    "minversion1_dolares",
]

# Endeudamiento total pesos
ENDEUDAMIENTO_PESOS = [
    "mprestamos_prendarios",
    "mprestamos_hipotecarios",
    "Master_msaldopesos",
    "Visa_msaldopesos",
    "Master_madelantopesos",
    "Visa_madelantopesos",
]

# Endeudamiento total dólares
ENDEUDAMIENTO_DOLARES = [
    "Master_msaldodolares",
    "Visa_msaldodolares",
    "Master_madelantodolares",
    "Visa_madelantodolares",
]

# Ingresos payroll pesos
PAYROLL_PESOS = [
    "mpayroll",
    "mpayroll2",
]

# Límites de compra
LIMITES_COMPRA = [
    "Master_mlimitecompra",
    "Visa_mlimitecompra",
]

# Suma de seguros contratados
SEGUROS_COUNT = [
    "cseguro_vida",
    "cseguro_auto",
    "cseguro_vivienda",
    "cseguro_accidentes_personales",
]

# Totales de transacciones (conteo)
TRANSACCIONES_COUNT = [
    "ctarjeta_debito_transacciones",
    "ctarjeta_visa_transacciones",
    "ctarjeta_master_transacciones",
    "chomebanking_transacciones",
    "cmobile_app_trx",
    "ccallcenter_transacciones",
    "catm_trx",
    "catm_trx_other",
    "ccajas_transacciones",
]

fe_columns = {
    "m_saldo_total_pesos": SALDOS_PESOS,
    "m_saldo_total_dolares": SALDOS_DOLARES,
    "m_consumo_tarjetas_pesos": CONSUMO_TARJETAS_PESOS,
    "m_consumo_tarjetas_dolares": CONSUMO_TARJETAS_DOLARES,
    "m_pagos_recurrentes_pesos": PAGOS_RECURRENTE_PESOS,
    "m_inversiones_pesos": INVERSIONES_PESOS,
    "m_inversiones_dolares": INVERSIONES_DOLARES,
    "m_endeudamiento_pesos": ENDEUDAMIENTO_PESOS,
    "m_endeudamiento_dolares": ENDEUDAMIENTO_DOLARES,
    "m_payroll_total": PAYROLL_PESOS,
    "m_limite_compra_total": LIMITES_COMPRA,
    "c_seguros_total": SEGUROS_COUNT,
    "c_transacciones_total": TRANSACCIONES_COUNT,
}

# Columnas sobre las que queremos generar suavizados y extremos temporales
TEMPORAL_FE_COLS = [
    "m_saldo_total_pesos",
    "m_saldo_total_dolares",
    "m_consumo_tarjetas_pesos",
    "m_consumo_tarjetas_dolares",
    "m_pagos_recurrentes_pesos",
    "m_inversiones_pesos",
    "m_inversiones_dolares",
    "m_endeudamiento_pesos",
    "m_endeudamiento_dolares",
    "m_payroll_total",
    "m_limite_compra_total",
    "c_transacciones_total",
]


# -------------------------------------------------
# Funciones de feature engineering en Polars
# -------------------------------------------------

def sum_columns_safe(df: pl.DataFrame, cols: list[str]) -> pl.Expr:
    """
    Suma horizontal segura: ignora columnas que no existan en el DF.

    Si ninguna columna de `cols` existe, devuelve 0 literal.
    """
    existentes = [c for c in cols if c in df.columns]
    if not existentes:
        return pl.lit(0)
    return pl.sum_horizontal([pl.col(c) for c in existentes])


def add_temporal_smoothing_and_extremes(
    df: pl.DataFrame,
    id_col: str,
    period_col: str,
    cols: list[str],
    ema_alphas: tuple[float, ...] = (0.3, 0.6),
    windows: tuple[int, ...] = (3, 6, 12),
) -> pl.DataFrame:
    """
    Agrega para cada columna en `cols` (por cliente y ordenado por periodo):

    - Suavizados exponenciales (EMA) con distintos alphas:
        {col}_ema_alpha03, {col}_ema_alpha06, ...
    - Máximos y mínimos rolling:
        {col}_max_win3, {col}_min_win3, {col}_max_win6, ...

    Todo se hace en modo lazy para no romper memoria.
    """
    cols = [c for c in cols if c in df.columns]
    if not cols:
        logger.warning("add_temporal_smoothing_and_extremes: no hay columnas válidas en 'cols'.")
        return df

    logger.info(
        "Agregando suavizados exponenciales y max/min rolling para %d columnas...",
        len(cols),
    )

    lf = df.sort([id_col, period_col]).lazy()
    exprs: list[pl.Expr] = []

    for c in cols:
        # EMAs
        for alpha in ema_alphas:
            alpha_str = str(alpha).replace(".", "p")
            exprs.append(
                pl.col(c)
                .ewm_mean(alpha=alpha, adjust=True, min_periods=1)
                .over(id_col)
                .alias(f"{c}_ema_alpha{alpha_str}")
            )

        # Rolling max/min en ventanas de 3, 6, 12 meses
        for w in windows:
            exprs.append(
                pl.col(c)
                .rolling_max(window_size=w, min_periods=1)
                .over(id_col)
                .alias(f"{c}_max_win{w}")
            )
            exprs.append(
                pl.col(c)
                .rolling_min(window_size=w, min_periods=1)
                .over(id_col)
                .alias(f"{c}_min_win{w}")
            )

    lf = lf.with_columns(exprs)
    result = lf.collect()

    logger.info(
        "Suavizados y extremos temporales agregados. Shape ahora: %s",
        (result.shape,),
    )
    return result


def add_lags_and_deltas(
    df: pl.DataFrame,
    id_col: str,
    period_col: str,
    cols: list[str],
) -> pl.DataFrame:
    """
    Agregar lags y deltas de orden 1 y 2 para todas las columnas numéricas indicadas.

    - lag1: x_{t-1}
    - lag2: x_{t-2}
    - d1: x_t - x_{t-1}
    - d2: x_t - x_{t-2}
    """
    logger.info(
        "Aplicando lags y deltas (lag1, lag2, d1, d2) a %d columnas numéricas...",
        len(cols),
    )

    out = df.sort([id_col, period_col]).lazy()

    # Lags 1 y 2
    out = out.with_columns([
        pl.col(c).shift(1).over(id_col).alias(f"{c}_lag1") for c in cols
    ] + [
        pl.col(c).shift(2).over(id_col).alias(f"{c}_lag2") for c in cols
    ])

    # Deltas
    out = out.with_columns([
        (pl.col(c) - pl.col(f"{c}_lag1")).alias(f"{c}_d1") for c in cols
    ] + [
        (pl.col(c) - pl.col(f"{c}_lag2")).alias(f"{c}_d2") for c in cols
    ])

    result = out.collect()
    logger.info("Lags y deltas agregados. Shape final: %s", (result.shape,))
    return result


def run_feature_engineering_polars() -> str:
    """
    Ejecuta la etapa de feature engineering usando Polars.

    Pasos:
    1. Carga el dataset procesado con target (`paths.processed_dataset`).
    2. Calcula features agregadas (sumas horizontales).
    3. Agrega EMAs y max/min rolling para algunas columnas agregadas.
    4. Detecta columnas numéricas y aplica lags/deltas
       (excluyendo id, periodo, target y EMAs / max/min).
    5. Filtra periodos >= 201907.
    6. Guarda el resultado en `paths.feature_dataset`.

    Returns
    -------
    str
        Ruta del dataset final de features.
    """
    cfg = load_config()
    processed_path = cfg.paths.processed_dataset
    feature_out_path = cfg.paths.feature_dataset

    id_col = cfg.columns.id_column
    period_col = cfg.columns.period_column
    target_col = cfg.columns.target_column

    logger.info("Dataset base procesado (con target): %s", processed_path)
    logger.info("Output final de features: %s", feature_out_path)
    logger.info("Columnas clave -> id: %s | periodo: %s | target: %s", id_col, period_col, target_col)

    # 1) Cargar dataset base en Polars (local o GCS)
    logger.info("Leyendo dataset base con Polars...")

    if is_gcs_path(processed_path):
        fs = gcsfs.GCSFileSystem()
        if processed_path.endswith(".parquet"):
            with fs.open(processed_path, "rb") as f:
                df = pl.read_parquet(f)
        else:
            with fs.open(processed_path, "rb") as f:
                df = pl.read_csv(f)
    else:
        if not os.path.exists(processed_path):
            logger.error("No se encontró el dataset procesado: %s", processed_path)
            raise FileNotFoundError(
                f"No encontré el archivo procesado (con target) {processed_path}"
            )

        if processed_path.endswith(".parquet"):
            df = pl.read_parquet(processed_path)
        else:
            df = pl.read_csv(processed_path)

    logger.info("Dataset base leído. Shape: %s", (df.shape,))
    logger.debug("Columnas iniciales: %s", df.columns)

    # 2) Features agregadas (sumas horizontales)
    logger.info("Generando features agregadas (sumas horizontales)...")
    df = df.with_columns([
        sum_columns_safe(df, cols).alias(col_name)
        for col_name, cols in fe_columns.items()
    ])
    logger.info("Features agregadas creadas. Shape actual: %s", (df.shape,))

    # 2.b) Suavizados exponenciales y max/min temporales
    df = add_temporal_smoothing_and_extremes(
        df=df,
        id_col=id_col,
        period_col=period_col,
        cols=TEMPORAL_FE_COLS,
        ema_alphas=(0.3, 0.6),
        windows=(3, 6, 12),
    )

    # 3) Detectar columnas numéricas para lags/deltas
    #    (TODAS menos id, periodo, target y las nuevas EMA/max/min)
    EXCLUDE = {id_col, period_col, target_col}
    DERIVED_SUFFIXES = ("_ema_alpha", "_max_win", "_min_win")

    num_cols: list[str] = []
    for c, dt in zip(df.columns, df.dtypes):
        if c in EXCLUDE:
            continue
        if not dt.is_numeric():
            continue
        if any(sfx in c for sfx in DERIVED_SUFFIXES):
            continue
        num_cols.append(c)

    logger.info(
        "Columnas numéricas seleccionadas para lags/deltas: %d columnas.",
        len(num_cols),
    )
    logger.debug("Ejemplo de columnas numéricas: %s", num_cols[:20])

    # 4) Aplicar lags y deltas
    df = add_lags_and_deltas(df, id_col=id_col, period_col=period_col, cols=num_cols)

    # 5) Filtrar periodos desde 201907 en adelante
    logger.info("Filtrando filas con %s >= 201907 ...", period_col)
    df = df.filter(pl.col(period_col) >= 201907)
    logger.info("Shape luego de filtrar periodos >= 201907: %s", (df.shape,))

    # 6) Guardar resultado final (local o GCS)
    if is_gcs_path(feature_out_path):
        fs = gcsfs.GCSFileSystem()
        if feature_out_path.endswith(".parquet"):
            logger.info("Guardando features finales en Parquet (GCS): %s", feature_out_path)
            with fs.open(feature_out_path, "wb") as f:
                df.write_parquet(f)
        else:
            logger.info("Guardando features finales en CSV (GCS): %s", feature_out_path)
            with fs.open(feature_out_path, "wb") as f:
                df.write_csv(f)
    else:
        os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
        if feature_out_path.endswith(".parquet"):
            logger.info("Guardando features finales en Parquet: %s", feature_out_path)
            df.write_parquet(feature_out_path)
        else:
            logger.info("Guardando features finales en CSV: %s", feature_out_path)
            df.write_csv(feature_out_path)

    logger.info("Features finales guardadas en %s ✅", feature_out_path)
    return feature_out_path


if __name__ == "__main__":
    out_path = run_feature_engineering_polars()
    logger.info("🏁 Feature engineering (Polars) completado. Output: %s", out_path)
