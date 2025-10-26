"""
pipeline
========

Orquesta las etapas principales del flujo de datos del proyecto de churn:

1. data_prep.run_data_stage()
   - Lee el dataset crudo (paths.raw_dataset en config.yaml)
   - Genera el dataset con target "clase_ternaria"
   - Escribe paths.processed_dataset (ej: data/processed/competencia_01.csv)

2. feature_engineering.run_feature_engineering()
   - Lee el dataset procesado con target
   - Ejecuta las queries SQL definidas en config/features.steps
     (01_base_tables.sql, 02_feat_numeric.sql, 03_final_model.sql)
   - Escribe paths.feature_dataset (ej: data/processed/competencia_01_features.csv)

Uso:
    python -m src.pipeline
"""

import logging

from src import data_prep
from src import feature_engineering


# -------------------------------------------------
# Logging config (coherente con los otros módulos)
# -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Ejecuta el pipeline end-to-end de preparación de datos
    y generación de features.

    Etapas:
    1. run_data_stage() -> genera dataset con target.
    2. run_feature_engineering() -> genera dataset final con features.

    Returns
    -------
    None
        (Pero loggea las rutas de salida producidas por cada etapa.)
    """
    logger.info("🚀 Iniciando pipeline completo EyF Churn...")

    # 1. Stage de datos (target/clase_ternaria)
    logger.info("▶ Etapa 1: data prep (creación de target)...")
    final_csv_target = data_prep.run_data_stage()
    logger.info("✅ Dataset con target listo: %s", final_csv_target)

    # 2. Feature engineering (lag, ratios, etc. vía DuckDB SQL)
    logger.info("▶ Etapa 2: feature engineering...")
    final_csv_features = feature_engineering.run_feature_engineering()
    logger.info("✅ Dataset final con features listo: %s", final_csv_features)

    logger.info("🏁 Pipeline completado con éxito.")


if __name__ == "__main__":
    main()
