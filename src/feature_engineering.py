"""
feature_engineering
====================

Genera variables derivadas (feature engineering) a partir del dataset con target.

Flujo:
1. Carga el dataset procesado con target (paths.processed_dataset en config.yaml).
2. Registra ese dataframe como una tabla DuckDB en memoria (features.base_table_name).
3. Ejecuta, en orden, los archivos SQL listados en config.features.steps.
   - Los primeros scripts pueden hacer CREATE TABLE / TRANSFORM intermedios.
   - El ÚLTIMO script DEBE terminar con un SELECT que devuelva el dataset final.
4. Ese SELECT final se convierte a pandas y se guarda en paths.feature_dataset.

Uso:
    python -m src.feature_engineering
"""

import os
import logging
from typing import List
import pandas as pd
import duckdb
import yaml
from pydantic import BaseModel, Field, ValidationError


# -------------------------------------------------
# Logging
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


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
    Configuración específica de la etapa de feature engineering.
    """
    base_table_name: str = Field(
        "base_clientes",
        description="Nombre lógico con el que se registra el dataset base en DuckDB."
    )
    steps: List[str] = Field(
        ...,
        description=(
            "Lista de rutas a archivos .sql que se ejecutan en orden. "
            "El último archivo debe terminar con un SELECT que devuelva "
            "el dataset final listo para modelar."
        )
    )


class FullConfig(BaseModel):
    """
    FullConfig
    ----------
    Esquema completo esperado en config/config.yaml.
    """
    paths: PathsConfig
    columns: ColumnsConfig
    features: FeaturesConfig
    # logic.* puede existir en el YAML pero no es obligatorio para correr


def load_config(path: str = "config/config.yaml") -> FullConfig:
    """
    Cargar y validar la configuración del proyecto usando Pydantic.

    Parameters
    ----------
    path : str, default="config/config.yaml"
        Ruta al archivo YAML de configuración.

    Returns
    -------
    FullConfig
        Objeto de configuración validado, con rutas de entrada/salida,
        columnas clave y lista de pasos SQL.

    Raises
    ------
    FileNotFoundError
        Si el archivo de configuración no existe.
    ValidationError
        Si faltan claves requeridas o los tipos no matchean el esquema.
    """
    logger.info("Cargando configuración de feature engineering desde %s ...", path)

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

    logger.debug("Config validada (feature engineering): %s", cfg)
    return cfg


def _read_sql_file(sql_path: str) -> str:
    """
    Lee el contenido de un archivo .sql y lo devuelve como string.

    Parameters
    ----------
    sql_path : str
        Ruta al archivo SQL (por ejemplo 'sql/01_base_tables.sql').

    Returns
    -------
    str
        Texto SQL completo.

    Raises
    ------
    FileNotFoundError
        Si el archivo SQL no existe.
    """
    if not os.path.exists(sql_path):
        logger.error("No se encontró el archivo SQL: %s", sql_path)
        raise FileNotFoundError(f"No encontré el archivo SQL {sql_path}")

    with open(sql_path, "r", encoding="utf-8") as f:
        sql_text = f.read()

    return sql_text


def run_feature_engineering() -> str:
    """
    Ejecuta la etapa de feature engineering usando DuckDB y archivos SQL externos.

    Pasos
    -----
    1. Carga el dataset procesado con target (`paths.processed_dataset`).
    2. Lo registra en DuckDB con el nombre `features.base_table_name`.
    3. Para cada archivo SQL en `features.steps`:
       - Lee el .sql
       - Reemplaza placeholders:
            {base_table}   -> features.base_table_name
            {id_col}       -> columns.id_column
            {period_col}   -> columns.period_column
            {target_col}   -> columns.target_column
       - Ejecuta el SQL en DuckDB.
       - Si el SQL termina con un SELECT y devuelve filas, capturamos ese
         resultado como candidato a dataset final.
    4. Toma el último resultado tabular y lo guarda como CSV en
       `paths.feature_dataset`.

    Returns
    -------
    str
        Ruta del CSV final de features listo para modelar.

    Raises
    ------
    FileNotFoundError
        Si falta el dataset base procesado o algún .sql.
    RuntimeError
        Si ninguna query devolvió un SELECT con filas (no hay dataset final).
    """
    logger.info("Iniciando etapa de feature engineering...")

    # 1. Cargar configuración
    cfg = load_config()

    processed_path = cfg.paths.processed_dataset
    feature_out_path = cfg.paths.feature_dataset
    base_tbl = cfg.features.base_table_name
    sql_files = cfg.features.steps

    id_col = cfg.columns.id_column
    period_col = cfg.columns.period_column
    target_col = cfg.columns.target_column

    logger.info("Dataset base procesado (con target): %s", processed_path)
    logger.info("Output final de features: %s", feature_out_path)
    logger.info("Tabla base en DuckDB: %s", base_tbl)
    logger.info("Columnas clave -> id: %s | periodo: %s | target: %s", id_col, period_col, target_col)

    # 2. Cargar dataset base (el que ya tiene clase_ternaria)
    if not os.path.exists(processed_path):
        logger.error("No se encontró el dataset procesado: %s", processed_path)
        raise FileNotFoundError(
            f"No encontré el archivo procesado (con target) {processed_path}"
        )

    df_base = pd.read_csv(processed_path)
    logger.info("Dataset base leído. Shape: %s", (df_base.shape,))
    logger.debug("Primeras filas base:\n%s", df_base.head())

    # 3. Conexión DuckDB y registrar tabla base
    logger.info("Creando conexión DuckDB en memoria y registrando tabla base '%s' ...", base_tbl)
    con = duckdb.connect(database=":memory:")
    con.register(base_tbl, df_base)
    logger.info("Tabla '%s' registrada en DuckDB.", base_tbl)

    # 4. Ejecutar cada archivo SQL en orden
    last_result_df = None

    logger.info("Ejecutando %d scripts SQL de feature engineering...", len(sql_files))
    for i, sql_path in enumerate(sql_files, start=1):
        logger.info("▶ Paso SQL %d/%d: %s", i, len(sql_files), sql_path)

        raw_query = _read_sql_file(sql_path)

        # Reemplazar placeholders con nombres reales
        rendered_query = raw_query.format(
            base_table=base_tbl,
            id_col=id_col,
            period_col=period_col,
            target_col=target_col,
        )

        logger.debug("SQL renderizado (%s):\n%s", sql_path, rendered_query)

        # Ejecutar el SQL contra DuckDB
        result = con.execute(rendered_query)

        # Intentar capturar un resultado tabular (SELECT final)
        try:
            candidate_df = result.fetch_df()
            last_result_df = candidate_df
            logger.info(
                "Paso %d devolvió un resultado tabular. Shape: %s",
                i, (candidate_df.shape,)
            )
        except duckdb.IOException:
            # No es un SELECT con retorno de filas (por ejemplo CREATE TABLE ...)
            logger.info(
                "Paso %d ejecutado. (No devolvió filas directamente, probablemente CREATE/INSERT.)",
                i,
            )

    # 5. Verificar que tengamos dataset final
    if last_result_df is None:
        logger.error(
            "Ningún script SQL devolvió un SELECT final. "
            "El último archivo de features.steps debe terminar con un SELECT."
        )
        raise RuntimeError(
            "No se obtuvo dataset final de features. "
            "Revisá que el último .sql termine con un SELECT."
        )

    logger.info("Dataset final de features obtenido. Shape: %s", (last_result_df.shape,))
    logger.debug("Preview dataset final:\n%s", last_result_df.head())

    # 6. Guardar resultado final enriquecido
    os.makedirs(os.path.dirname(feature_out_path), exist_ok=True)
    last_result_df.to_csv(feature_out_path, index=False)
    logger.info("Features finales guardadas en %s ✅", feature_out_path)

    return feature_out_path


if __name__ == "__main__":
    # Permite ejecutar:
    #   python -m src.feature_engineering
    out_path = run_feature_engineering()
    logger.info("🏁 Feature engineering completado. Output: %s", out_path)
