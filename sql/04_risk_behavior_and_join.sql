-- Paso 4: Features de riesgo de churn + dataset final listo para el modelo
-- IMPORTANTE:
--   - Requiere tablas creadas en pasos previos:
--       competencia_01_num        (Paso 1, resumen numérico)
--       comp_features_enriquecida (Paso 3, actividad enriquecida)
--
--   - Este script hace:
--       A) comp_risk_features  -> señales de apagado, stress financiero, volatilidad, fuga de fondos
--       B) dataset_modelo_churn -> JOIN final para entrenar churn


-- ==========================================================
-- A) Tabla comp_risk_features
-- ==========================================================

CREATE OR REPLACE TABLE comp_risk_features AS
WITH base_union AS (
    SELECT
        cf.{id_col}          AS numero_de_cliente,
        cf.{period_col}      AS foto_mes,
        CAST(cf.{period_col} AS INT) AS foto_mes_int,

        -- ---------------------------
        -- Engagement / actividad
        -- ---------------------------
        cf.ctrx_quarter,
        cf.lag_1_ctrx_quarter,
        cf.lag_2_ctrx_quarter,
        cf.lag_3_ctrx_quarter,
        cf.avg_3_ctrx_quarter,
        cf.delta_1_ctrx_quarter,
        cf.delta_2_ctrx_quarter,

        cf.ctrx_quarter_media_3,
        cf.ctrx_quarter_max_3,
        cf.ctrx_quarter_min_3,
        cf.ctrx_quarter_slope_3,

        cf.cliente_antiguedad,
        cf.cliente_antiguedad_2,
        cf.cliente_antiguedad_3,
        cf.cliente_antiguedad_4,
        cf.cliente_antiguedad_5,
        cf.cliente_antiguedad_6,

        -- ---------------------------
        -- Variables financieras de n
        -- ---------------------------

        -- consumo total en tarjeta (legacy: tarjeta_mconsumototal_total)
        n.tarjeta_mconsumo_total     AS tarjeta_mconsumo_total,

        -- cantidad de consumos con tarjeta
        n.tarjeta_cconsumos_total         AS tarjeta_cconsumos_total,

        -- límite de tarjeta (legacy: tarjeta_mlimitecompra_total)
        n.tarjeta_mlimite_total     AS tarjeta_mlimite_total,

        -- frecuencia total de transacciones
        n.freq_total_tx                   AS freq_total_tx,

        -- monto total transaccionado reciente
        -- si tu métrica real tiene otro nombre, usala acá
        n.acm_mconsumo_total             AS monto_total_tx,

        -- rolling / ventanas móviles
        n.mm3_tarjeta_mconsumo_total AS mm3_tarjeta_mconsumo_total,
        n.roll3_freq_total_tx             AS roll3_freq_total_tx,
        n.roll3_min_monto_total_tx        AS roll3_min_monto_total_tx,
        n.roll3_max_monto_total_tx        AS roll3_max_monto_total_tx,

        -- stress financiero
        n.ratio_utilizacion_tarjeta       AS ratio_utilizacion_tarjeta,
        n.ticket_medio_tarjeta            AS ticket_medio_tarjeta,

        -- acumulados históricos
        n.acm_mconsumo_total              AS acm_mconsumo_total,
        n.acm_cconsumos_total             AS acm_cconsumos_total,

        -- tendencia de consumo tarjeta
        n.trend_slope_mconsumo_total      AS trend_slope_mconsumo_total,
        n.trend_intercept_mconsumo_total  AS trend_intercept_mconsumo_total,

        -- flujos de fondos
        n.mtransferencias_emitidas        AS mtransferencias_emitidas,
        n.mtransferencias_recibidas       AS mtransferencias_recibidas,

        -- índice temporal auxiliar / orden temporal
        n.mes_idx                         AS mes_idx
    FROM comp_features_enriquecida cf
    LEFT JOIN competencia_01_num n
      ON cf.{id_col}     = n.numero_de_cliente
     AND CAST(cf.{period_col} AS INT) = n.foto_mes_int
),

win AS (
    SELECT
        b.*,

        ----------------------------------------------------------------
        -- 1) APAGADO DE ACTIVIDAD / DESENGANCHE
        ----------------------------------------------------------------
        CASE
            WHEN ABS(NULLIF(lag_1_ctrx_quarter,0)) > 1e-12 THEN
                (ctrx_quarter - lag_1_ctrx_quarter)
                / ABS(lag_1_ctrx_quarter)
            ELSE NULL
        END AS pct_drop_ctrx_quarter_1m,

        CASE
            WHEN lag_1_ctrx_quarter IS NOT NULL
             AND lag_1_ctrx_quarter > 0
             AND ctrx_quarter < 0.5 * lag_1_ctrx_quarter
            THEN 1 ELSE 0 END AS flag_big_drop_activity,

        CASE
            WHEN delta_1_ctrx_quarter < 0
             AND delta_2_ctrx_quarter < 0
            THEN 1 ELSE 0 END AS flag_two_months_down,

        CASE
            WHEN ctrx_quarter <= 3
             AND COALESCE(lag_1_ctrx_quarter,0) > 3
            THEN 1 ELSE 0 END AS flag_almost_inactive_now,

        ----------------------------------------------------------------
        -- 2) STRESS FINANCIERO / USO DE CRÉDITO
        ----------------------------------------------------------------
        CASE
            WHEN ratio_utilizacion_tarjeta IS NOT NULL
             AND ratio_utilizacion_tarjeta > 0.80
            THEN 1 ELSE 0 END AS flag_high_utilization_tc,

        CASE
            WHEN ticket_medio_tarjeta IS NOT NULL
             AND ticket_medio_tarjeta < 2000
             AND freq_total_tx IS NOT NULL
             AND freq_total_tx > 20
            THEN 1 ELSE 0 END AS flag_low_ticket_high_freq,

        ----------------------------------------------------------------
        -- 3) VOLATILIDAD TRANSACCIONAL
        ----------------------------------------------------------------
        (roll3_max_monto_total_tx - roll3_min_monto_total_tx)
            AS vol_tx_range_roll3,

        CASE
            WHEN roll3_max_monto_total_tx IS NOT NULL
             AND ABS(roll3_max_monto_total_tx) > 1e-12
            THEN (roll3_max_monto_total_tx - roll3_min_monto_total_tx)
                 / ABS(roll3_max_monto_total_tx)
            ELSE NULL
        END AS vol_tx_rel_roll3,

        ----------------------------------------------------------------
        -- 4) SEÑALES DE RETIRO DE FONDOS / BAJA DE INGRESOS
        ----------------------------------------------------------------
        (
            mtransferencias_recibidas
            - LAG(mtransferencias_recibidas) OVER (
                PARTITION BY numero_de_cliente
                ORDER BY foto_mes_int
              )
        ) AS delta_mrecibidas,

        CASE
            WHEN LAG(mtransferencias_recibidas) OVER (
                    PARTITION BY numero_de_cliente
                    ORDER BY foto_mes_int
                 ) > 0
             AND mtransferencias_recibidas <
                 0.5 * LAG(mtransferencias_recibidas) OVER (
                        PARTITION BY numero_de_cliente
                        ORDER BY foto_mes_int
                     )
            THEN 1 ELSE 0 END AS flag_income_drop,

        ----------------------------------------------------------------
        -- 5) ANTIGÜEDAD VS USO ACTUAL
        ----------------------------------------------------------------
        CASE
            WHEN cliente_antiguedad IS NOT NULL
             AND cliente_antiguedad > 0
            THEN ctrx_quarter * 1.0 / cliente_antiguedad
            ELSE NULL
        END AS ctrx_per_antiguedad,

        CASE
            WHEN cliente_antiguedad >= 24
             AND ctrx_quarter <= 3
            THEN 1 ELSE 0 END AS flag_old_low_usage,

        ----------------------------------------------------------------
        -- 6) TENDENCIA DE CONSUMO TARJETA
        ----------------------------------------------------------------
        CASE
            WHEN trend_slope_mconsumo_total IS NOT NULL
             AND trend_slope_mconsumo_total < 0
            THEN 1 ELSE 0 END AS flag_trend_neg_consume_tc
    FROM base_union b
)

SELECT * FROM win
;


-- ==========================================================
-- B) Dataset final listo para entrenar churn
-- ==========================================================

CREATE OR REPLACE TABLE dataset_modelo_churn AS
SELECT
    f.*,
    r.pct_drop_ctrx_quarter_1m,
    r.flag_big_drop_activity,
    r.flag_two_months_down,
    r.flag_almost_inactive_now,
    r.flag_high_utilization_tc,
    r.flag_low_ticket_high_freq,
    r.vol_tx_range_roll3,
    r.vol_tx_rel_roll3,
    r.delta_mrecibidas,
    r.flag_income_drop,
    r.ctrx_per_antiguedad,
    r.flag_old_low_usage,
    r.flag_trend_neg_consume_tc,

    -- columnas base que paso 5 sigue usando o hace LAG sobre ellas
    r.ticket_medio_tarjeta,
    r.tarjeta_mconsumo_total,
    r.tarjeta_mlimite_total,
    r.freq_total_tx,
    r.monto_total_tx,
    r.mtransferencias_recibidas,
    r.mtransferencias_emitidas,
    r.ratio_utilizacion_tarjeta
FROM comp_features_enriquecida f
LEFT JOIN comp_risk_features r
  ON f.{id_col}     = r.numero_de_cliente
 AND f.{period_col} = r.foto_mes
;

-- ==========================================================
-- Output final del paso 4:
-- ==========================================================
SELECT * FROM dataset_modelo_churn
;
