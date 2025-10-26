-- sql/03_final_model.sql
-- Paso 3: unir comp_features con competencia_01_num para obtener el dataset final
-- IMPORTANTE: el último statement debe ser un SELECT que devuelva filas.

-- Primero: enriquecer con stats de ctrx_quarter en ventana móvil de 3
CREATE OR REPLACE TABLE competencia_01_num_ctrx AS
WITH ventana_num AS (
  SELECT
    numero_de_cliente,
    CAST({period_col} AS INT) AS foto_mes_int_safe,
    {period_col} AS foto_mes,
    ctrx_quarter,
    cliente_antiguedad,

    AVG(ctrx_quarter)  OVER (
      PARTITION BY numero_de_cliente
      ORDER BY CAST({period_col} AS INT)
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS ctrx_quarter_media_3,

    MAX(ctrx_quarter)  OVER (
      PARTITION BY numero_de_cliente
      ORDER BY CAST({period_col} AS INT)
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS ctrx_quarter_max_3,

    MIN(ctrx_quarter)  OVER (
      PARTITION BY numero_de_cliente
      ORDER BY CAST({period_col} AS INT)
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS ctrx_quarter_min_3,

    REGR_SLOPE(ctrx_quarter, cliente_antiguedad) OVER (
      PARTITION BY numero_de_cliente
      ORDER BY CAST({period_col} AS INT)
      ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS ctrx_quarter_slope_3
  FROM competencia_01
)
SELECT * FROM ventana_num
;

-- Ahora unimos todo en comp_features_enriquecida
CREATE OR REPLACE TABLE comp_features_enriquecida AS
SELECT
  cf.*,
  n.ctrx_quarter_media_3,
  n.ctrx_quarter_max_3,
  n.ctrx_quarter_min_3,
  n.ctrx_quarter_slope_3
FROM comp_features cf
LEFT JOIN competencia_01_num_ctrx n
  ON cf.numero_de_cliente = n.numero_de_cliente
 AND cf.{period_col}      = n.foto_mes
;

-- Y devolvemos la tabla final como SELECT (esto es lo que Python guardará en CSV)
SELECT * FROM comp_features_enriquecida
;
