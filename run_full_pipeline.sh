#!/usr/bin/env bash
set -euo pipefail

echo "=== [1/4] Activando entorno Python ==========================="

# Si vas a correr esto en una VM Linux:
#   source .venv/bin/activate
#
# En Windows con Git Bash, lo más común es:
#   source .venv/Scripts/activate
#
# Para que el script sea portable, vamos a detectar carpeta:
if [ -d ".venv/bin" ]; then
    # layout tipo Linux / WSL
    source .venv/bin/activate
elif [ -d ".venv/Scripts" ]; then
    # layout típico Windows venv
    source .venv/Scripts/activate
else
    echo "⚠️ No se encontró el entorno virtual .venv. Asegurate de crearlo antes de correr este script."
    echo "   Ejemplo:"
    echo "      python -m venv .venv"
    echo "      source .venv/bin/activate   (Linux/WSL)"
    echo "      .venv\\Scripts\\Activate.ps1 (PowerShell)"
    exit 1
fi

echo "Python usado:"
python --version
echo ""

echo "=== [2/4] Generando datasets ================================"
echo "-> Paso A: data_prep (target / clase_ternaria)"
python -m src.data_prep

echo "-> Paso B: feature_engineering (features numéricas, lags, ratios...)"
python -m src.feature_engineering

echo ""
echo "=== [3/4] Buscando hiperparámetros =========================="
echo "-> optimizer: Optuna / LightGBM, guarda best_params.yaml y best_model.pkl"
python -m src.optimizer

echo ""
echo "=== [4/4] Entrenando modelo final ==========================="
echo "-> trainer: reentrena con TODO el dataset de features usando best_params.yaml"
python -m src.trainer

echo ""
echo "🏁 Pipeline completo OK"
