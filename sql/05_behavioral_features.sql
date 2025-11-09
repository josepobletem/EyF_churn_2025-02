-- Paso 5: agregar features avanzadas de estabilidad, estrés, productos, engagement histórico
-- INPUT  : dataset_modelo_churn (creada en el paso 4)
-- OUTPUT : dataset_modelo_churn_enriquecido
--
-- Este script:
--   - Usa ventanas por cliente ordenadas por foto_mes_int.
--   - Agrega volatilidad, estrés financiero, señales de abandono prolongado,
--     relación ingresos/gastos, dependencia de tarjeta, etc.
--
-- IMPORTANTE:
--   dataset_modelo_churn debe tener al menos:
--     numero_de_cliente
--     {period_col}              (string tipo '202104')
--     foto_mes_int              (INT tipo 202104)
--     fecha_mes                 (DATE)
--     ctrx_quarter
--     cliente_antiguedad
--     tarjeta_mconsumo_total
--     tarjeta_cconsumos_total
--     tarjeta_mlimite_total
--     tc_saldo_total
--     freq_total_tx
--     monto_total_tx
--     mtransferencias_emitidas
--     mtransferencias_recibidas
--     ratio_utilizacion_tarjeta
--     ticket_medio_tarjeta
--     ... y el resto generado en pasos previos.


CREATE OR REPLACE TABLE dataset_modelo_churn_enriquecido AS
WITH base AS (
    SELECT
        d.*,

        --------------------------------------------------------------------
        -- LAGS útiles para variaciones (cambios de comportamiento)
        --------------------------------------------------------------------
        LAG(ratio_utilizacion_tarjeta, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_ratio_utilizacion_tarjeta,

        LAG(ticket_medio_tarjeta, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_ticket_medio_tarjeta,

        LAG(freq_total_tx, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_freq_total_tx,

        LAG(ctrx_quarter, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_ctrx_quarter,

        LAG(tarjeta_mconsumo_total, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_tarjeta_mconsumo_total,

        LAG(monto_total_tx, 1) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        ) AS lag_monto_total_tx,


        --------------------------------------------------------------------
        -- HISTÓRICO HASTA EL MES ACTUAL (cumulative windows)
        --------------------------------------------------------------------
        MAX(ctrx_quarter) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hist_max_ctrx_quarter,

        AVG(ctrx_quarter) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hist_avg_ctrx_quarter,

        AVG(tarjeta_mconsumo_total) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS hist_avg_tarjeta_mconsumo_total,


        --------------------------------------------------------------------
        -- ROLLING 3 MESES (variabilidad / volatilidad / estabilidad)
        --------------------------------------------------------------------
        STDDEV(ctrx_quarter) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS std_ctrx_quarter_3,

        AVG(ctrx_quarter) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS avg_ctrx_quarter_3,

        STDDEV(tarjeta_mconsumo_total) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS std_mconsumo_tarjeta_3,

        AVG(tarjeta_mconsumo_total) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS avg_mconsumo_tarjeta_3,

        SUM(CASE WHEN ctrx_quarter < 3 THEN 1 ELSE 0 END) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS low_activity_count_3,  -- cuántos de los últimos 3 periodos estuvo casi inactivo


        --------------------------------------------------------------------
        -- TRANSFERENCIAS COMO PROXY DE INGRESOS / RETIRO DE FONDOS
        --------------------------------------------------------------------
        CASE
            WHEN mtransferencias_emitidas IS NOT NULL
             AND mtransferencias_emitidas > 0
            THEN mtransferencias_recibidas * 1.0 / mtransferencias_emitidas
            ELSE NULL
        END AS ratio_ingresos_gastos,  -- cuánto entra vs cuánto sale

        -- flag: no recibió plata en los últimos 3 meses (posible migración sueldo)
        SUM(
            CASE
                WHEN mtransferencias_recibidas IS NULL OR mtransferencias_recibidas = 0
                THEN 1 ELSE 0
            END
        ) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS no_deposit_3m_count,


        --------------------------------------------------------------------
        -- STRESS FINANCIERO
        --------------------------------------------------------------------
        CASE
            WHEN tarjeta_mlimite_total IS NOT NULL
             AND tarjeta_mlimite_total > 0
            THEN (tc_saldo_total * 1.0 / tarjeta_mlimite_total)
            ELSE NULL
        END AS ratio_saldo_sobre_limite_tc,

        CASE
            WHEN tarjeta_mlimite_total IS NOT NULL
             AND tarjeta_mlimite_total > 0
             AND tc_saldo_total * 1.0 / tarjeta_mlimite_total > 0.95
            THEN 1 ELSE 0
        END AS flag_maxed_credit,  -- prácticamente al tope de la línea


        --------------------------------------------------------------------
        -- ENGAGEMENT RELATIVO A SU HISTORIA
        --------------------------------------------------------------------
        CASE
            WHEN hist_max_ctrx_quarter IS NOT NULL
             AND hist_max_ctrx_quarter > 0
            THEN ctrx_quarter * 1.0 / hist_max_ctrx_quarter
            ELSE NULL
        END AS pct_ctrx_vs_histmax,  -- actividad actual vs el máximo histórico

        CASE
            WHEN hist_avg_tarjeta_mconsumo_total IS NOT NULL
             AND hist_avg_tarjeta_mconsumo_total > 0
            THEN tarjeta_mconsumo_total * 1.0 / hist_avg_tarjeta_mconsumo_total
            ELSE NULL
        END AS pct_cons_actual_vs_hist_avg,  -- consumo tarjeta hoy vs su propio promedio histórico


        --------------------------------------------------------------------
        -- CAMBIO DE COMPORTAMIENTO (DIFERENCIAS vs MES ANTERIOR)
        --------------------------------------------------------------------
        (ratio_utilizacion_tarjeta - LAG(ratio_utilizacion_tarjeta) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_ratio_utilizacion,

        (ticket_medio_tarjeta - LAG(ticket_medio_tarjeta) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_ticket_medio,

        (freq_total_tx - LAG(freq_total_tx) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_freq_tx,

        (ctrx_quarter - LAG(ctrx_quarter) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_ctrx_quarter,

        (tarjeta_mconsumo_total - LAG(tarjeta_mconsumo_total) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_tarjeta_mconsumo_total,

        (monto_total_tx - LAG(monto_total_tx) OVER (
            PARTITION BY numero_de_cliente
            ORDER BY foto_mes_int
        )) AS delta_monto_total_tx,


        --------------------------------------------------------------------
        -- INTERACCIONES / RATIOS COMPUESTOS
        --------------------------------------------------------------------
        CASE
            WHEN freq_total_tx IS NOT NULL
             AND freq_total_tx > 0
            THEN tarjeta_mconsumo_total * 1.0 / freq_total_tx
            ELSE NULL
        END AS ratio_cons_vs_freq,      -- gasto tarjeta / cantidad de tx totales

        CASE
            WHEN ctrx_quarter IS NOT NULL
             AND ctrx_quarter > 0
            THEN tarjeta_mconsumo_total * 1.0 / ctrx_quarter
            ELSE NULL
        END AS ratio_cons_vs_ctrx,      -- gasto tarjeta / tx_trimestrales

        CASE
            WHEN monto_total_tx IS NOT NULL
             AND monto_total_tx > 0
            THEN tarjeta_mconsumo_total * 1.0 / monto_total_tx
            ELSE NULL
        END AS ratio_credito_vs_total,  -- qué % del flujo total es tarjeta de crédito


        --------------------------------------------------------------------
        -- ANTIGÜEDAD Y DESENGANCHE LARGO PLAZO
        --------------------------------------------------------------------
        CASE
            WHEN cliente_antiguedad < 6
            THEN 1 ELSE 0
        END AS flag_cliente_reciente,

        CASE
            WHEN cliente_antiguedad >= 24
             AND ctrx_quarter <= 3
            THEN 1 ELSE 0
        END AS flag_veterano_apagado,  -- similar a flag_old_low_usage pero mantenemos para modelo


        --------------------------------------------------------------------
        -- SEÑAL DE INACTIVIDAD PROLONGADA:
        -- ¿cuántos de los últimos 3 meses estuvo "casi apagado"?
        --------------------------------------------------------------------
        CASE
            WHEN low_activity_count_3 >= 3 THEN 1 ELSE 0
        END AS flag_inactividad_prolongada_3m

    FROM dataset_modelo_churn d
),

final AS (
    SELECT
        b.*,

        ----------------------------------------------------------------
        -- Derivados de volatilidad: coeficiente de variación (CV)
        ----------------------------------------------------------------
        CASE
            WHEN b.avg_ctrx_quarter_3 IS NOT NULL
             AND b.avg_ctrx_quarter_3 > 0
            THEN b.std_ctrx_quarter_3 / b.avg_ctrx_quarter_3
            ELSE NULL
        END AS cv_ctrx_quarter_3,

        CASE
            WHEN b.avg_mconsumo_tarjeta_3 IS NOT NULL
             AND b.avg_mconsumo_tarjeta_3 > 0
            THEN b.std_mconsumo_tarjeta_3 / b.avg_mconsumo_tarjeta_3
            ELSE NULL
        END AS cv_mconsumo_tarjeta_3

    FROM base b
)

SELECT * FROM final
;

-- ==========================================================
-- Output final del paso 5:
-- ==========================================================
SELECT * FROM dataset_modelo_churn_enriquecido
;