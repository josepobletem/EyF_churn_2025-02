"""
data_prep
=========

Módulo de preparación de datos para el proyecto de churn.

Este script:
1. Lee el dataset crudo configurado en `config/config.yaml`.
2. Calcula la etiqueta de churn `clase_ternaria` usando DuckDB con la lógica:
   - BAJA+1 : el cliente no aparece en el mes siguiente
   - BAJA+2 : el cliente aparece en el mes siguiente pero no en el siguiente a ese
   - CONTINUA : el cliente sigue estando
3. Genera y guarda un dataset procesado con esa columna target.

Se puede ejecutar directamente desde la raíz del repo con:
    python -m src.data_prep
"""

import os
import logging
import pandas as pd
import duckdb
import yaml
from pydantic import BaseModel, Field, ValidationError


# -------------------------------------------------
# Configuración básica de logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------------------------------
# Modelos de configuración (Pydantic)
# -------------------------------------------------

class PathsConfig(BaseModel):
    """
    Rutas de archivos usadas por la etapa de data prep.
    """
    raw_dataset: str = Field(..., description="Ruta al CSV crudo de entrada.")
    processed_dataset: str = Field(..., description="Ruta al CSV final procesado.")


class ColumnsConfig(BaseModel):
    """
    Nombres de columnas relevantes del dataset.
    """
    id_column: str = Field(..., description="Nombre de la columna de ID cliente.")
    period_column: str = Field(..., description="Nombre de la columna temporal (ej. 'foto_mes').")
    target_column: str = Field(
        "clase_ternaria",
        description="Nombre de la columna objetivo que vamos a generar."
    )


class FullConfig(BaseModel):
    """
    Configuración completa cargada desde config.yaml.
    """
    paths: PathsConfig
    columns: ColumnsConfig
    # Podrías agregar más secciones en el futuro, ej:
    # logic: dict


def load_config(path: str = "config/config.yaml") -> FullConfig:
    """
    Cargar y validar la configuración del proyecto usando Pydantic.

    Parameters
    ----------
    path : str, default="config/config.yaml"
        Ruta al archivo YAML de configuración. Debe contener:
        - paths.raw_dataset
        - paths.processed_dataset
        - columns.id_column
        - columns.period_column
        - columns.target_column (opcional; default 'clase_ternaria')

    Returns
    -------
    FullConfig
        Objeto de configuración validado y tipado.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    ValidationError
        Si faltan claves obligatorias o los tipos no son correctos.
    """
    logger.info(f"Cargando configuración desde {path} ...")

    if not os.path.exists(path):
        logger.error(f"No se encontró el archivo de configuración: {path}")
        raise FileNotFoundError(f"No encontré el archivo de configuración {path}")

    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    try:
        cfg = FullConfig(**raw_cfg)
    except ValidationError as e:
        logger.error("Error validando config.yaml con Pydantic:\n%s", e)
        raise

    logger.debug(f"Config validada: {cfg}")
    return cfg


def yyyymm_add(yyyymm: int, k: int) -> int:
    """
    Sumar meses a un período en formato YYYYMM.

    Dado un entero que representa un período `YYYYMM`, devuelve el período
    que resulta de sumar `k` meses. Soporta overflow de año.

    Ejemplos
    --------
    >>> yyyymm_add(202104, 1)
    202105
    >>> yyyymm_add(202112, 1)
    202201
    >>> yyyymm_add(202112, 2)
    202202
    >>> yyyymm_add(202201, -1)
    202112

    Parameters
    ----------
    yyyymm : int
        Período en formato `YYYYMM`. Por ejemplo, abril 2021 -> 202104.
    k : int
        Cantidad de meses a sumar (puede ser negativa).

    Returns
    -------
    int
        Nuevo período en formato `YYYYMM` luego de sumar `k` meses.

    Notes
    -----
    - Si más adelante la granularidad temporal deja de ser mensual (por ejemplo,
      si `period_column` pasa a representar años tipo `2021`, `2022`, etc.),
      podés reemplazar el cuerpo de esta función con `return yyyymm + k`.
    """
    year = yyyymm // 100
    month = yyyymm % 100

    month = month + k
    while month > 12:
        month -= 12
        year += 1
    while month < 1:
        month += 12
        year -= 1

    return year * 100 + month


def run_data_stage() -> str:
    """
    Ejecutar la etapa de preparación de datos end-to-end.

    Esta función:
    1. Lee el dataset crudo de clientes desde la ruta indicada en config.yaml.
    2. Valida que las columnas mínimas (ID de cliente y período) existan.
    3. Estandariza los nombres de columnas a un esquema interno:
       - `numero_de_cliente`
       - `foto_mes`
    4. Calcula `foto_mes_t1` (mes siguiente) y `foto_mes_t2` (dos meses después).
    5. Usa DuckDB para determinar si cada cliente sigue presente en t+1 y t+2.
    6. Construye la columna objetivo `clase_ternaria` con la lógica:
         - 'BAJA+1'  -> el cliente no aparece en t+1
         - 'BAJA+2'  -> aparece en t+1 pero no en t+2
         - 'CONTINUA'-> sigue apareciendo en t+2
    7. Devuelve las columnas originales más `clase_ternaria`.
    8. Guarda el resultado final en la ruta `paths.processed_dataset` del config.

    Returns
    -------
    str
        Ruta del archivo CSV procesado que se generó (el dataset final listo
        para modelado, con la columna `clase_ternaria` incluida).

    Raises
    ------
    FileNotFoundError
        Si el dataset crudo no existe en la ruta indicada en la configuración.
    ValueError
        Si faltan columnas obligatorias en el dataset crudo.
    ValidationError
        Si la configuración (`config.yaml`) es inválida.
    """

    logger.info("Iniciando etapa de preparación de datos (run_data_stage)...")

    # ------------------------------------------------------------------
    # 0. Load and validate config
    # ------------------------------------------------------------------
    cfg = load_config()

    raw_path = cfg.paths.raw_dataset
    out_path = cfg.paths.processed_dataset

    id_col = cfg.columns.id_column          # ej "numero_de_cliente"
    period_col = cfg.columns.period_column  # ej "foto_mes"
    target_col = cfg.columns.target_column  # ej "clase_ternaria" (no se usa para renombrar, info de referencia)

    logger.info("Ruta de input crudo: %s", raw_path)
    logger.info("Ruta de salida procesado: %s", out_path)
    logger.info("Columna ID: %s | Columna período: %s | Target: %s", id_col, period_col, target_col)

    # ------------------------------------------------------------------
    # 1. Read raw dataset
    # ------------------------------------------------------------------
    logger.info("Leyendo dataset crudo...")
    if not os.path.exists(raw_path):
        logger.error("No se encontró el archivo de entrada: %s", raw_path)
        raise FileNotFoundError(f"No encontré el archivo {raw_path}")

    df_raw_in = pd.read_parquet(raw_path)
    logger.info("Datos crudos leídos. Shape: %s", (df_raw_in.shape,))

    # ------------------------------------------------------------------
    # 2. Validate required columns exist
    # ------------------------------------------------------------------
    required_cols = {id_col, period_col}
    faltantes = required_cols - set(df_raw_in.columns)
    if faltantes:
        logger.error("Faltan columnas requeridas: %s", faltantes)
        raise ValueError(f"Faltan columnas en el CSV crudo: {faltantes}")
    logger.info("Columnas mínimas presentes en el dataset crudo.")

    # ------------------------------------------------------------------
    # 3. Normalize to internal column names
    # ------------------------------------------------------------------
    logger.info("Normalizando nombres internos (numero_de_cliente / foto_mes)...")
    df_std = df_raw_in.rename(columns={
        id_col: "numero_de_cliente",
        period_col: "foto_mes",
    }).copy()

    logger.info("Forzando tipo entero en foto_mes...")
    df_std["foto_mes"] = df_std["foto_mes"].astype(int)

    # ------------------------------------------------------------------
    # 4. Compute future period references (t+1, t+2)
    # ------------------------------------------------------------------
    logger.info("Calculando foto_mes_t1 (t+1) y foto_mes_t2 (t+2)...")
    df_std["foto_mes_t1"] = df_std["foto_mes"].apply(lambda m: yyyymm_add(m, 1))
    df_std["foto_mes_t2"] = df_std["foto_mes"].apply(lambda m: yyyymm_add(m, 2))

    logger.debug("Preview df_std con columnas auxiliares:\n%s", df_std.head())

    # ------------------------------------------------------------------
    # 5. Register with DuckDB
    # ------------------------------------------------------------------
    logger.info("Creando conexión DuckDB en memoria y registrando tabla intermedia...")
    con = duckdb.connect(database=":memory:")
    con.register("competencia_02_crudo_enriq", df_std)
    logger.info("Tabla 'competencia_02_crudo_enriq' registrada en DuckDB.")

    # ------------------------------------------------------------------
    # 6. DuckDB SQL: derive churn status
    # ------------------------------------------------------------------
    logger.info("Ejecutando SQL en DuckDB para construir clase_ternaria...")

    target_sql = r"""
        WITH base AS (
            SELECT
                numero_de_cliente,
                foto_mes,
                foto_mes_t1,
                foto_mes_t2
            FROM competencia_02_crudo_enriq
        ),
        t1 AS (
            SELECT DISTINCT
                numero_de_cliente,
                foto_mes AS foto_mes_t1_real
            FROM competencia_02_crudo_enriq
        ),
        t2 AS (
            SELECT DISTINCT
                numero_de_cliente,
                foto_mes AS foto_mes_t2_real
            FROM competencia_02_crudo_enriq
        ),
        marcado AS (
            SELECT
                b.numero_de_cliente,
                b.foto_mes,

                CASE 
                    WHEN t1.numero_de_cliente IS NOT NULL THEN 1
                    ELSE 0
                END AS esta_t1,

                CASE 
                    WHEN t2.numero_de_cliente IS NOT NULL THEN 1
                    ELSE 0
                END AS esta_t2

            FROM base b
            LEFT JOIN t1
                ON b.numero_de_cliente = t1.numero_de_cliente
               AND b.foto_mes_t1 = t1.foto_mes_t1_real
            LEFT JOIN t2
                ON b.numero_de_cliente = t2.numero_de_cliente
               AND b.foto_mes_t2 = t2.foto_mes_t2_real
        )
        SELECT
            numero_de_cliente,
            foto_mes,
            CASE
                WHEN esta_t1 = 0 THEN 'BAJA+1'
                WHEN esta_t1 = 1 AND esta_t2 = 0 THEN 'BAJA+2'
                ELSE 'CONTINUA'
            END AS clase_ternaria
        FROM marcado
    """

    df_target = con.execute(target_sql).fetch_df()
    logger.info("Target generado. Shape: %s", (df_target.shape,))
    logger.debug("Preview df_target:\n%s", df_target.head())

    # ------------------------------------------------------------------
    # 7. Merge target back into the standardized dataset
    # ------------------------------------------------------------------
    logger.info("Haciendo merge entre dataset estándar y target...")
    df_final_std = df_std.merge(
        df_target,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )
    logger.info("Post-merge. Shape: %s", (df_final_std.shape,))

    # Remover columnas auxiliares internas
    for aux_col in ("foto_mes_t1", "foto_mes_t2"):
        if aux_col in df_final_std.columns:
            del df_final_std[aux_col]

    # ------------------------------------------------------------------
    # 8. Rename columns back to original names
    # ------------------------------------------------------------------
    logger.info("Restaurando nombres de columnas originales...")
    df_final_out = df_final_std.rename(columns={
        "numero_de_cliente": id_col,
        "foto_mes": period_col,
    })

    logger.debug("Preview df_final_out:\n%s", df_final_out.head())

    # ------------------------------------------------------------------
    # 9. Save processed dataset
    # ------------------------------------------------------------------
    logger.info("Guardando dataset final en %s ...", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_final_out.to_parquet(out_path, index=False)
    logger.info("Guardado OK ✅")

    logger.info("Primeras filas del dataset final listo para modelar:")
    logger.info("\n%s", df_final_out.head())

    logger.info("Etapa de preparación de datos completada exitosamente.")
    return out_path


if __name__ == "__main__":
    # Permite ejecutar:
    #   python -m src.data_prep
    output_path = run_data_stage()
    logger.info("🏁 Proceso de data prep completado. Output: %s", output_path)
