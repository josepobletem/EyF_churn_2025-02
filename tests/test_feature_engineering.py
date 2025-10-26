import os
import pandas as pd
import pytest
import duckdb
from pydantic import ValidationError

# Importamos el módulo que vamos a testear
import src.feature_engineering as fe


def test_load_config_success(tmp_path):
    """
    load_config() debe devolver un FullConfig válido cuando el YAML
    tiene todas las claves esperadas.
    """

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        """
paths:
  raw_dataset: "data/raw/competencia_01_crudo.csv"
  processed_dataset: "data/processed/competencia_01.csv"
  feature_dataset: "data/processed/competencia_01_features.csv"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column: "clase_ternaria"

features:
  base_table_name: "base_clientes"
  steps:
    - "sql/01_base_tables.sql"
    - "sql/02_feat_numeric.sql"
    - "sql/03_final_model.sql"
""",
        encoding="utf-8",
    )

    cfg = fe.load_config(str(cfg_path))

    assert isinstance(cfg, fe.FullConfig)

    # estas comparaciones tienen que matchear lo que ESCRIBIMOS arriba, literal:
    assert cfg.paths.processed_dataset == "data/processed/competencia_01.csv"
    assert cfg.paths.feature_dataset == "data/processed/competencia_01_features.csv"
    assert cfg.columns.id_column == "numero_de_cliente"
    assert cfg.features.steps[-1].endswith("03_final_model.sql")


def test_load_config_file_not_found():
    """
    load_config() debe levantar FileNotFoundError si el archivo no existe.
    """
    with pytest.raises(FileNotFoundError):
        fe.load_config("no_such_config.yaml")


def test_load_config_validation_error(tmp_path):
    """
    load_config() debe levantar ValidationError si falta una parte requerida,
    por ejemplo `features.steps`.
    """
    bad_cfg_path = tmp_path / "config_bad.yaml"
    bad_cfg_path.write_text(
        """
paths:
  processed_dataset: "data/processed/competencia_01.csv"
  feature_dataset: "data/processed/competencia_01_features.csv"
  raw_dataset: "data/raw/competencia_01_crudo.csv"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column: "clase_ternaria"

features:
  base_table_name: "base_clientes"
  # falta 'steps'
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        fe.load_config(str(bad_cfg_path))


def test_read_sql_file_success(tmp_path):
    """
    _read_sql_file() debe devolver el contenido de un archivo .sql existente.
    """
    sql_path = tmp_path / "foo.sql"
    sql_path.write_text("SELECT 1 AS x;", encoding="utf-8")

    sql_text = fe._read_sql_file(str(sql_path))
    assert "SELECT 1 AS x;" in sql_text


def test_read_sql_file_not_found():
    """
    _read_sql_file() debe levantar FileNotFoundError si el archivo no existe.
    """
    with pytest.raises(FileNotFoundError):
        fe._read_sql_file("nope.sql")


def test_run_feature_engineering_end_to_end(tmp_path, monkeypatch):
    """
    run_feature_engineering() debe:
    - Leer el processed_dataset
    - Registrar la tabla base en DuckDB
    - Ejecutar los .sql en orden (con placeholders reemplazados)
    - Tomar el resultado del último SELECT
    - Guardarlo en feature_dataset
    """

    # ---------------------------------------
    # 1. Creamos un CSV "procesado" dummy
    # ---------------------------------------
    processed_df = pd.DataFrame(
        {
            "numero_de_cliente": [101, 101, 202],
            "foto_mes": [202401, 202402, 202401],
            "clase_ternaria": ["CONTINUA", "BAJA+1", "CONTINUA"],
            # columnas adicionales que podrían usarse en SQL
            "Master_mconsumototal": [1000.0, 1200.0, 500.0],
            "Visa_mconsumototal": [300.0, 200.0, 100.0],
            "Master_cconsumos": [5, 6, 2],
            "Visa_cconsumos": [2, 1, 1],
            "Master_mlimitecompra": [20000.0, 20000.0, 15000.0],
            "Visa_mlimitecompra": [10000.0, 10000.0, 8000.0],
            "ctransferencias_emitidas": [1, 2, 0],
            "ccajas_transacciones": [3, 1, 2],
            "mtransferencias_emitidas": [50.0, 70.0, 0.0],
            "mtransferencias_recibidas": [20.0, 10.0, 5.0],
            "ctrx_quarter": [10, 12, 7],
            "cliente_antiguedad": [24, 25, 10],
        }
    )

    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)

    processed_csv_path = data_dir / "competencia_01.csv"
    processed_df.to_csv(processed_csv_path, index=False)

    # ---------------------------------------
    # 2. Creamos 3 archivos SQL fake
    #    Simulan tu 01/02/03 reales, pero mínimos
    # ---------------------------------------

    # Paso 1:
    # - crea tabla intermedia a partir de {base_table}
    # - y también crea otra tabla derivada con alguna transformación
    sql_1 = """
CREATE OR REPLACE TABLE base_cast AS
SELECT
  *,
  CAST({period_col} AS BIGINT) AS foto_mes_int
FROM {base_table}
;

CREATE OR REPLACE TABLE base_vars AS
SELECT
  *,
  (COALESCE(Master_mconsumototal,0) + COALESCE(Visa_mconsumototal,0)) AS tarjeta_mconsumo_total
FROM base_cast
;

CREATE OR REPLACE TABLE competencia_01 AS
SELECT
  *,
  LAG(tarjeta_mconsumo_total, 1) OVER (
    PARTITION BY {id_col} ORDER BY foto_mes_int
  ) AS lag1_tarjeta_mconsumo_total
FROM base_vars
;
"""
    # Paso 2:
    # - crea otra tabla comp_features a partir de competencia_01
    sql_2 = """
CREATE OR REPLACE TABLE comp_features AS
SELECT
  *,
  (tarjeta_mconsumo_total - lag1_tarjeta_mconsumo_total) AS delta_mconsumo
FROM competencia_01
;
"""

    # Paso 3:
    # - SELECT final (lo que el pipeline debería exportar)
    sql_3 = """
SELECT
  {id_col}        AS cliente_id,
  {period_col}    AS periodo,
  {target_col}    AS target,
  tarjeta_mconsumo_total,
  lag1_tarjeta_mconsumo_total,
  delta_mconsumo
FROM comp_features
;
"""

    sql_dir = tmp_path / "sql"
    sql_dir.mkdir(parents=True, exist_ok=True)

    sql_1_path = sql_dir / "01_base_tables.sql"
    sql_1_path.write_text(sql_1, encoding="utf-8")

    sql_2_path = sql_dir / "02_feat_numeric.sql"
    sql_2_path.write_text(sql_2, encoding="utf-8")

    sql_3_path = sql_dir / "03_final_model.sql"
    sql_3_path.write_text(sql_3, encoding="utf-8")

    # ---------------------------------------
    # 3. Creamos config.yaml temporal que apunte
    #    a estos paths y a los archivos SQL
    # ---------------------------------------
    cfg_path = tmp_path / "config.yaml"
    feature_out_path = data_dir / "competencia_01_features.csv"

    cfg_yaml = f"""
paths:
  raw_dataset: "{(tmp_path / 'data' / 'raw' / 'competencia_01_crudo.csv').as_posix()}"
  processed_dataset: "{processed_csv_path.as_posix()}"
  feature_dataset: "{feature_out_path.as_posix()}"

columns:
  id_column: "numero_de_cliente"
  period_column: "foto_mes"
  target_column: "clase_ternaria"

features:
  base_table_name: "base_clientes"
  steps:
    - "{sql_1_path.as_posix()}"
    - "{sql_2_path.as_posix()}"
    - "{sql_3_path.as_posix()}"
"""

    cfg_path.write_text(cfg_yaml, encoding="utf-8")

    # ---------------------------------------
    # 4. Preparar objeto cfg real una sola vez y luego monkeypatch
    # ---------------------------------------
    real_cfg = fe.load_config(str(cfg_path))

    def _fake_load_config():
        # simplemente devolvemos el objeto ya cargado arriba
        return real_cfg

    monkeypatch.setattr(fe, "load_config", _fake_load_config)

    # ---------------------------------------
    # 5. Ejecutar la función bajo test
    # ---------------------------------------
    out_path = fe.run_feature_engineering()

    # Debe devolver la misma ruta que declaramos en feature_dataset
    assert out_path == feature_out_path.as_posix()
    assert os.path.exists(out_path)

    # ---------------------------------------
    # 6. Validar el CSV final generado
    # ---------------------------------------
    final_df = pd.read_csv(out_path)

    # Columnas esperadas del SELECT final
    for col in [
        "cliente_id",
        "periodo",
        "target",
        "tarjeta_mconsumo_total",
        "lag1_tarjeta_mconsumo_total",
        "delta_mconsumo",
    ]:
        assert col in final_df.columns

    # Debe haber tantas filas como combinaciones únicas (numero_de_cliente, foto_mes)
    expected_rows = (
        processed_df[["numero_de_cliente", "foto_mes"]]
        .drop_duplicates()
        .shape[0]
    )
    assert final_df.shape[0] == expected_rows

    # Target debe venir arrastrado correctamente
    assert "BAJA+1" in final_df["target"].values or "CONTINUA" in final_df["target"].values
