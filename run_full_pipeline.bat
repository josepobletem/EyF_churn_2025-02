@echo off
REM ================================
REM run_full_pipeline.bat
REM ================================
REM Comportamiento similar al bash:
REM - Activa el entorno virtual .venv
REM - Ejecuta los pasos del pipeline
REM - Se detiene si hay error

REM Hacer que cualquier error corte la ejecución
REM (en batch usamos IF ERRORLEVEL después de cada paso)

echo === [1/4] Activando entorno Python ===========================

REM Detectar entorno virtual .venv
IF EXIST ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) ELSE (
    echo ⚠️ No se encontró el entorno virtual .venv.
    echo    Asegurate de crearlo antes de correr este script.
    echo    Ejemplo:
    echo       python -m venv .venv
    echo       .venv\Scripts\activate.bat
    exit /b 1
)

echo Python usado:
python --version
IF ERRORLEVEL 1 (
    echo [ERROR] Python no responde. Abortando.
    exit /b 1
)
echo.

echo === [2/4] Generando datasets ================================
echo -> Paso A: data_prep (target / clase_ternaria)
echo python -m src.data_prep
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.data_prep
echo     exit /b 1
echo )

echo -> Paso B: feature_engineering (features numéricas, lags, ratios...)
echo python -m src.feature_engineering
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.feature_engineering
echo     exit /b 1
echo )

echo.
echo === [3/4] Buscando hiperparámetros ==========================
echo -> optimizer: Optuna / LightGBM, guarda best_params.yaml y best_model.pkl
echo python -m src.optimizer
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.optimizer
echo     exit /b 1
echo )

echo.
echo === [4/4] Entrenando modelo final ===========================
echo -> trainer: reentrena con TODO el dataset de features usando best_params.yaml
echo python -m src.trainer
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.trainer
echo     exit /b 1
echo )

echo ==========================================================
echo === [5/5] Scoring / Prediccion mensual ==================
echo ==========================================================

REM Ajusta estos parámetros si queres scorar otro mes / umbral
set SCORE_MES=202106
set SCORE_THRESHOLD=0.0039

echo -> Predict mes %SCORE_MES% con threshold %SCORE_THRESHOLD%
python -m src.predict --mes %SCORE_MES% --threshold %SCORE_THRESHOLD%
IF ERRORLEVEL 1 (
    echo [ERROR] Fallo en src.predict
    exit /b 1
)
echo.

echo.
echo Pipeline completo OK

REM Fin normal
exit /b 0
