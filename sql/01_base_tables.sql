-- sql/01_base_tables.sql
-- Paso 1: preparar tablas base y primeras features a partir de {base_table}
-- NOTA: {base_table} ya viene registrada desde Python (feature_engineering.py)
--       a partir de paths.processed_dataset. NO usar read_csv_auto acá.

-- 1) Cargar y castear yyyymm -> BIGINT
CREATE OR REPLACE TABLE base_cast AS
SELECT
  *,
  CAST({period_col} AS BIGINT) AS foto_mes_int
FROM {base_table}
;

-- 2) Agregar fecha_mes usando foto_mes_int
CREATE OR REPLACE TABLE base_raw AS
SELECT
  b.*,
  MAKE_DATE(
    CAST(floor(b.foto_mes_int / 100) AS INT),   -- año
    CAST(b.foto_mes_int % 100 AS INT),          -- mes
    1
  ) AS fecha_mes
FROM base_cast b
;

-- 3) Combos base usando columnas reales (consumos tarjeta, límites, actividad)
-- TODO: revisar que estas columnas existan en tu dataset:
--   Master_mconsumototal, Visa_mconsumototal,
--   Master_cconsumos, Visa_cconsumos,
--   Master_mlimitecompra, Visa_mlimitecompra,
--   ctransferencias_emitidas, ccajas_transacciones,
--   mtransferencias_emitidas, mtransferencias_recibidas
CREATE OR REPLACE TABLE base_vars AS
SELECT
  *,
  COALESCE(Master_mconsumototal, 0) + COALESCE(Visa_mconsumototal, 0) AS tarjeta_mconsumo_total,
  COALESCE(Master_cconsumos, 0)     + COALESCE(Visa_cconsumos, 0)     AS tarjeta_cconsumos_total,
  COALESCE(Master_mlimitecompra,0)  + COALESCE(Visa_mlimitecompra,0)  AS tarjeta_mlimite_total,

  -- frecuencia total de actividad del cliente
  COALESCE(Master_cconsumos, 0)
  + COALESCE(Visa_cconsumos, 0)
  + COALESCE(ctransferencias_emitidas, 0)
  + COALESCE(ccajas_transacciones, 0) AS freq_total_tx,

  -- monto total movido
  COALESCE(Master_mconsumototal, 0)
  + COALESCE(Visa_mconsumototal, 0)
  + COALESCE(mtransferencias_emitidas, 0)
  + COALESCE(mtransferencias_recibidas, 0) AS monto_total_tx
FROM base_raw
;

-- 4) Feature engineering (lags, deltas, medias móviles, ratios, acumulados, tendencia)
--    Esta es tu "competencia_01" enriquecida
CREATE OR REPLACE TABLE competencia_01 AS
WITH base AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
    ) AS mes_idx,

    COALESCE(Master_mconsumototal, 0) + COALESCE(Visa_mconsumototal, 0) AS tarjeta_mconsumototal_total,
    COALESCE(Master_cconsumos, 0)     + COALESCE(Visa_cconsumos, 0)     AS tarjeta_cconsumos_total,
    COALESCE(Master_mlimitecompra,0)  + COALESCE(Visa_mlimitecompra,0)  AS tarjeta_mlimitecompra_total
  FROM base_raw
),
win AS (
  SELECT
    b.*,

    -- LAGS
    LAG(tarjeta_mconsumototal_total, 1) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
    ) AS lag1_mconsumo_total,

    LAG(tarjeta_cconsumos_total, 1) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
    ) AS lag1_cconsumos_total,

    -- DELTAS (consumo tarjeta, #consumos)
    (tarjeta_mconsumototal_total
       - LAG(tarjeta_mconsumototal_total) OVER (
           PARTITION BY {id_col}
           ORDER BY foto_mes_int
         )
    ) AS delta_mconsumo_total,

    (tarjeta_cconsumos_total
       - LAG(tarjeta_cconsumos_total) OVER (
           PARTITION BY {id_col}
           ORDER BY foto_mes_int
         )
    ) AS delta_cconsumos_total,

    -- MEDIAS MÓVILES 1/2/3 sobre consumo total tarjeta
    AVG(tarjeta_mconsumototal_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN 0 PRECEDING AND CURRENT ROW
    ) AS mm_1_mconsumo_total,

    AVG(tarjeta_mconsumototal_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    ) AS mm_2_mconsumo_total,

    AVG(tarjeta_mconsumototal_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS mm_3_mconsumo_total,

    -- media móvil 3 periodos para #consumos totales
    AVG(tarjeta_cconsumos_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS mm_3_cconsumos_total,

    -- RATIOS
    CASE WHEN tarjeta_mlimitecompra_total > 0
         THEN tarjeta_mconsumototal_total * 1.0 / tarjeta_mlimitecompra_total
         ELSE 0 END AS ratio_utilizacion_tarjeta,

    CASE WHEN tarjeta_cconsumos_total > 0
         THEN tarjeta_mconsumototal_total * 1.0 / tarjeta_cconsumos_total
         ELSE NULL END AS ratio_ticket_medio,

    -- ACUMULADOS
    SUM(tarjeta_mconsumototal_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS acm_mconsumo_total,

    SUM(tarjeta_cconsumos_total) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS acm_cconsumos_total,

    -- TENDENCIA (regresión lineal del consumo en el tiempo)
    REGR_SLOPE(tarjeta_mconsumototal_total, mes_idx) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS trend_slope_mconsumo_total,

    REGR_INTERCEPT(tarjeta_mconsumototal_total, mes_idx) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS trend_intercept_mconsumo_total

  FROM base b
)
SELECT
  w.*,

  -- Lags/deltas de variables base Visa/Master individuales
  LAG(Visa_mconsumototal,   1) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
  ) AS lag1_Visa_mconsumototal,

  LAG(Master_mconsumototal, 1) OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
  ) AS lag1_Master_mconsumototal,

  (Visa_mconsumototal
     - LAG(Visa_mconsumototal) OVER (
         PARTITION BY {id_col}
         ORDER BY foto_mes_int
       )
  ) AS delta_Visa_mconsumototal,

  (Master_mconsumototal
     - LAG(Master_mconsumototal) OVER (
         PARTITION BY {id_col}
         ORDER BY foto_mes_int
       )
  ) AS delta_Master_mconsumototal

FROM win w
;

-- 5) Versión numérica resumida (competencia_01_num)
--    parecida a tu segunda query (num / num_feats / competencia_01_num)
CREATE OR REPLACE TABLE competencia_01_num AS
WITH
num AS (
  SELECT
    {id_col}              AS numero_de_cliente,
    foto_mes_int,
    fecha_mes,
    tarjeta_mconsumo_total,
    tarjeta_cconsumos_total,
    tarjeta_mlimite_total,
    Visa_mconsumototal,
    Master_mconsumototal,
    Visa_cconsumos,
    Master_cconsumos,
    monto_total_tx,
    freq_total_tx,
    mtransferencias_emitidas,
    mtransferencias_recibidas,
    ccajas_transacciones,
    ctransferencias_emitidas,
    ROW_NUMBER() OVER (
      PARTITION BY {id_col}
      ORDER BY foto_mes_int
    ) AS mes_idx
  FROM base_vars
),
num_feats AS (
  SELECT
    n.*,

    -- LAGS
    LAG(tarjeta_mconsumo_total, 1) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
    ) AS lag1_tarjeta_mconsumo_total,

    LAG(tarjeta_cconsumos_total, 1) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
    ) AS lag1_tarjeta_cconsumos_total,

    LAG(monto_total_tx, 1) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
    ) AS lag1_monto_total_tx,

    LAG(freq_total_tx, 1) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
    ) AS lag1_freq_total_tx,

    -- DELTAS
    (tarjeta_mconsumo_total
       - LAG(tarjeta_mconsumo_total) OVER (
           PARTITION BY numero_de_cliente
           ORDER BY foto_mes_int
         )
    ) AS delta_tarjeta_mconsumo_total,

    (tarjeta_cconsumos_total
       - LAG(tarjeta_cconsumos_total) OVER (
           PARTITION BY numero_de_cliente
           ORDER BY foto_mes_int
         )
    ) AS delta_tarjeta_cconsumos_total,

    -- Growth rate (%)
    CASE
      WHEN ABS(NULLIF(LAG(tarjeta_mconsumo_total) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
          ),0)) > 1e-12
      THEN (tarjeta_mconsumo_total
            - LAG(tarjeta_mconsumo_total) OVER (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes_int
              )
           )
           / ABS(LAG(tarjeta_mconsumo_total) OVER (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes_int
              ))
      ELSE NULL
    END AS gr_tarjeta_mconsumo_total,

    -- Medias móviles 1/2/3
    AVG(tarjeta_mconsumo_total) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 0 PRECEDING AND CURRENT ROW
    ) AS mm1_tarjeta_mconsumo_total,

    AVG(tarjeta_mconsumo_total) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
    ) AS mm2_tarjeta_mconsumo_total,

    AVG(tarjeta_mconsumo_total) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS mm3_tarjeta_mconsumo_total,

    -- Rolling sum/min/max 3
    SUM(freq_total_tx) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS roll3_freq_total_tx,

    MIN(monto_total_tx) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS roll3_min_monto_total_tx,

    MAX(monto_total_tx) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS roll3_max_monto_total_tx,

    -- Ratios
    CASE WHEN tarjeta_mlimite_total > 0
         THEN tarjeta_mconsumo_total * 1.0 / tarjeta_mlimite_total
         ELSE 0 END AS ratio_utilizacion_tarjeta,

    CASE WHEN tarjeta_cconsumos_total > 0
         THEN tarjeta_mconsumo_total * 1.0 / tarjeta_cconsumos_total
         ELSE NULL END AS ticket_medio_tarjeta,

    -- Acumulados
    SUM(tarjeta_mconsumo_total) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS acm_mconsumo_total,

    SUM(tarjeta_cconsumos_total) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS acm_cconsumos_total,

    -- Tendencia
    REGR_SLOPE(tarjeta_mconsumo_total, mes_idx) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS trend_slope_mconsumo_total,

    REGR_INTERCEPT(tarjeta_mconsumo_total, mes_idx) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY foto_mes_int
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS trend_intercept_mconsumo_total

  FROM num n
)
SELECT * FROM num_feats
;
