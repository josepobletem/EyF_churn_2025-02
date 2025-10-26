import os
import pandas as pd
import duckdb
import yaml
import tempfile
from types import SimpleNamespace

import src.data_prep as dp


def test_yyyymm_add_basic():
    """
    yyyymm_add should correctly roll over months and years.
    """
    # 2021-04 + 1 month = 2021-05
    assert dp.yyyymm_add(202104, 1) == 202105

    # 2021-12 + 1 month = 2022-01
    assert dp.yyyymm_add(202112, 1) == 202201

    # 2021-12 + 2 months = 2022-02
    assert dp.yyyymm_add(202112, 2) == 202202

    # 2022-01 - 1 month = 2021-12 (negative k also should work)
    assert dp.yyyymm_add(202201, -1) == 202112


def _run_data_stage_like_function(raw_path, out_path, id_col, period_col):
    """
    Helper used in tests to mimic run_data_stage(), but with injected paths and columns.
    Returns the processed dataframe instead of saving to disk.
    """

    # leer crudo
    df_raw_in = pd.read_csv(raw_path)

    # renombrar estándar
    df_std = df_raw_in.rename(columns={
        id_col: "numero_de_cliente",
        period_col: "foto_mes",
    }).copy()

    df_std["foto_mes"] = df_std["foto_mes"].astype(int)

    # agregar t+1 y t+2
    df_std["foto_mes_t1"] = df_std["foto_mes"].apply(lambda m: dp.yyyymm_add(m, 1))
    df_std["foto_mes_t2"] = df_std["foto_mes"].apply(lambda m: dp.yyyymm_add(m, 2))

    # duckdb
    con = duckdb.connect(database=":memory:")
    con.register("competencia_01_crudo_enriq", df_std)

    target_sql = r"""
        WITH base AS (
            SELECT
                numero_de_cliente,
                foto_mes,
                foto_mes_t1,
                foto_mes_t2
            FROM competencia_01_crudo_enriq
        ),
        t1 AS (
            SELECT DISTINCT
                numero_de_cliente,
                foto_mes AS foto_mes_t1_real
            FROM competencia_01_crudo_enriq
        ),
        t2 AS (
            SELECT DISTINCT
                numero_de_cliente,
                foto_mes AS foto_mes_t2_real
            FROM competencia_01_crudo_enriq
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

    # merge final
    df_final_std = df_std.merge(
        df_target,
        on=["numero_de_cliente", "foto_mes"],
        how="left"
    )

    # limpiar auxiliares
    for aux_col in ("foto_mes_t1", "foto_mes_t2"):
        if aux_col in df_final_std.columns:
            del df_final_std[aux_col]

    # volver a nombres originales
    df_final_out = df_final_std.rename(columns={
        "numero_de_cliente": id_col,
        "foto_mes": period_col,
    })

    # en test, NO escribimos a disco, devolvemos df
    return df_final_out


def test_run_data_stage_logic_creates_clase_ternaria():
    """
    We simulate a tiny dataset with 2 customers across multiple months and check
    that clase_ternaria is correctly assigned:
      - BAJA+1 if they vanish next month
      - BAJA+2 if they vanish in 2 months
      - CONTINUA otherwise
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, "raw.csv")
        out_path = os.path.join(tmpdir, "processed.csv")

        # Creamos un dataset sintético
        #
        # cliente A:
        #   mes 202104
        #   mes 202105
        #   mes 202106
        # => siempre sigue -> CONTINUA
        #
        # cliente B:
        #   mes 202104
        #   mes 202105
        #   (no 202106)
        # => aparece en t+1 pero NO en t+2 -> BAJA+2 en 202104
        # para 202105 no está en 202106 => BAJA+1
        toy = pd.DataFrame({
            "numero_de_cliente": [111, 111, 111, 222, 222],
            "foto_mes":          [202104, 202105, 202106, 202104, 202105],
            "otra_var":          [10,     20,     30,     99,     88],
        })

        toy.to_csv(raw_path, index=False)

        # corremos el flujo "like run_data_stage"
        df_proc = _run_data_stage_like_function(
            raw_path=raw_path,
            out_path=out_path,
            id_col="numero_de_cliente",
            period_col="foto_mes",
        )

        # aseguramos que exista la columna target
        assert "clase_ternaria" in df_proc.columns

        # convertimos a dict por fila para inspección:
        rows = df_proc.sort_values(["numero_de_cliente", "foto_mes"]).to_dict(orient="records")

        # Buscamos por cliente/mes
        def get_row(cliente, mes):
            for r in rows:
                if r["numero_de_cliente"] == cliente and r["foto_mes"] == mes:
                    return r
            return None

        r_a_202104 = get_row(111, 202104)
        r_a_202105 = get_row(111, 202105)
        r_a_202106 = get_row(111, 202106)

        r_b_202104 = get_row(222, 202104)
        r_b_202105 = get_row(222, 202105)

        # Cliente 111:
        # 202104 -> tiene 202105 y 202106 => CONTINUA
        # 202105 -> tiene 202106 pero NO 202107 => BAJA+2
        # 202106 -> no tiene 202107 => BAJA+1
        assert r_a_202104["clase_ternaria"] == "CONTINUA"
        assert r_a_202105["clase_ternaria"] == "BAJA+2"
        assert r_a_202106["clase_ternaria"] == "BAJA+1"

        # Cliente 222:
        # 202104 -> tiene 202105 pero NO 202106 => BAJA+2
        # 202105 -> no tiene 202106 => BAJA+1
        assert r_b_202104["clase_ternaria"] == "BAJA+2"
        assert r_b_202105["clase_ternaria"] == "BAJA+1"

