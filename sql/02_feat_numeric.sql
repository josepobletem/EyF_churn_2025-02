-- sql/02_feat_numeric.sql
-- Paso 2: generar comp_features a partir de competencia_01

CREATE OR REPLACE TABLE comp_features AS
WITH win AS (
  SELECT
    c.*,

    -- saldo total tc
    IFNULL(Master_msaldototal, 0) + IFNULL(Visa_msaldototal, 0) AS tc_saldo_total,

    -- Lags de ctrx_quarter
    LAG(ctrx_quarter, 1) OVER (
      PARTITION BY {id_col}
      ORDER BY CAST({period_col} AS INT)
    ) AS lag_1_ctrx_quarter,

    LAG(ctrx_quarter, 2) OVER (
      PARTITION BY {id_col}
      ORDER BY CAST({period_col} AS INT)
    ) AS lag_2_ctrx_quarter,

    LAG(ctrx_quarter, 3) OVER (
      PARTITION BY {id_col}
      ORDER BY CAST({period_col} AS INT)
    ) AS lag_3_ctrx_quarter,

    -- Promedio móvil 3 de ctrx_quarter
    AVG(ctrx_quarter) OVER (
      PARTITION BY {id_col}
      ORDER BY CAST({period_col} AS INT)
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS avg_3_ctrx_quarter,

    -- Rankings por antigüedad dentro del mismo mes
    ROW_NUMBER() OVER (
      PARTITION BY {period_col}
      ORDER BY cliente_antiguedad
    ) AS cliente_antiguedad_2,

    PERCENT_RANK() OVER (
      PARTITION BY {period_col}
      ORDER BY cliente_antiguedad
    ) AS cliente_antiguedad_3,

    CUME_DIST() OVER (
      PARTITION BY {period_col}
      ORDER BY cliente_antiguedad
    ) AS cliente_antiguedad_4,

    NTILE(4) OVER (
      PARTITION BY {period_col}
      ORDER BY cliente_antiguedad
    ) AS cliente_antiguedad_5,

    NTILE(10) OVER (
      PARTITION BY {period_col}
      ORDER BY cliente_antiguedad
    ) AS cliente_antiguedad_6

  FROM competencia_01 c
)
SELECT
  w.*,
  -- deltas de ctrx_quarter usando los lags
  (w.ctrx_quarter - w.lag_1_ctrx_quarter) AS delta_1_ctrx_quarter,
  (w.ctrx_quarter - w.lag_2_ctrx_quarter) AS delta_2_ctrx_quarter
FROM win w
;
