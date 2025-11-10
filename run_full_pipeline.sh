#!/usr/bin/env bash
# ==========================================
# run_full_pipeline.sh
# ==========================================
# Flags soportados:
#   --ensemble            -> usa src.trainer_ensemble_and_predict (entrena y/o predice)
#   --mes=YYYYMM          -> override del mes a predecir (ej: --mes=202106)
#   --threshold=0.025     -> override del umbral (ej: --threshold=0.0025)
#   --train-only          -> con --ensemble: sólo entrena (no predice)
#   -h | --help           -> muestra ayuda
#
# Ejemplos:
#   ./run_full_pipeline.sh --ensemble --mes=202106 --threshold=0.025
#   ./run_full_pipeline.sh --mes=202106                    # pipeline clásico
#   ./run_full_pipeline.sh --ensemble --train-only         # sólo entrenamiento ensamble
#
# Requiere un venv en .venv/ con Python y dependencias.

set -euo pipefail

# ---------- helpers ----------
usage() {
  cat <<'EOF'
Uso:
  run_full_pipeline.sh [--ensemble] [--train-only] [--mes YYYYMM] [--threshold FLOAT]

Flags:
  --ensemble            Usa el pipeline de ensamble (trainer_ensemble_and_predict)
  --train-only          Con --ensemble: entrena sin hacer scoring
  --mes=YYYYMM          Mes a puntuar (también admite "--mes YYYYMM")
  --threshold=FLOAT     Umbral de decisión para scoring (también admite "--threshold FLOAT")
  -h, --help            Muestra esta ayuda
EOF
}

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ---------- defaults ----------
ENSEMBLE=0
TRAIN_ONLY=0
SCORE_MES="202106"
SCORE_THRESHOLD="0.01"

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ensemble) ENSEMBLE=1; shift ;;
    --train-only) TRAIN_ONLY=1; shift ;;
    --mes=*)
      SCORE_MES="${1#*=}"; shift ;;
    --mes)
      [[ $# -ge 2 ]] || { echo "[ERROR] Falta valor para --mes"; exit 1; }
      SCORE_MES="$2"; shift 2 ;;
    --threshold=*)
      SCORE_THRESHOLD="${1#*=}"; shift ;;
    --threshold)
      [[ $# -ge 2 ]] || { echo "[ERROR] Falta valor para --threshold"; exit 1; }
      SCORE_THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Flag desconocida: $1"
      usage
      exit 1 ;;
  esac
done

# ---------- entorno ----------
log "=== [1/5] Activando entorno Python ==========================="
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
else
  echo "⚠️  No se encontró el entorno virtual .venv."
  echo "    Crealo antes de correr este script:"
  echo "      python -m venv .venv"
  echo "      source .venv/bin/activate"
  exit 1
fi

log "Python usado:"
python --version >/dev/null

# ---------- Paso 2: Data prep / Features ----------
log "=========================================================="
log "=== [2/5] Generando datasets ============================="
log "=========================================================="

log "-> Paso A: data_prep (target / clase_ternaria)"
log python -m src.data_prep

log "-> Paso B: feature_engineering (features numéricas, lags, ratios...)"
log python -m src.feature_engineering

# ---------- Paso 3: Optimizer ----------
log "=========================================================="
log "=== [3/5] Buscando hiperparametros ======================="
log "=========================================================="

python -m src.optimizer

# ---------- Paso 4/5: Ensamble o clásico ----------
: <<'__DISABLED__'
if [[ "$ENSEMBLE" -eq 1 ]]; then
  if [[ "$TRAIN_ONLY" -eq 1 ]]; then
    log "=========================================================="
    log "=== [4/4] Entrenando ENSAMBLE (sin scoring) =============="
    log "=========================================================="
    python -m src.trainer_ensemble_and_predict --train
  else
    log "=========================================================="
    log "=== [4/4] Entrenar ENSAMBLE + Scoring en una llamada ====="
    log "=========================================================="
    log "-> Entrenando y prediciendo mes ${SCORE_MES} (thr ${SCORE_THRESHOLD})"
    python -m src.trainer_ensemble_and_predict --train --mes "${SCORE_MES}" --threshold "${SCORE_THRESHOLD}"
  fi
else
  # Flujo clásico
  log "=========================================================="
  log "=== [4/5] Entrenando modelo final (clásico) =============="
  log "=========================================================="
  python -m src.trainer

  log "=========================================================="
  log "=== [5/5] Scoring / Prediccion mensual (clásico) ========="
  log "=========================================================="
  log "-> Predict mes ${SCORE_MES} con threshold ${SCORE_THRESHOLD}"
  python -m src.predict --mes "${SCORE_MES}" --threshold "${SCORE_THRESHOLD}"
fi
__DISABLED__

echo
log "OK: ejecutados data_prep, feature_engineering y optimizer (resto comentado)."
