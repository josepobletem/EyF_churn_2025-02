@echo off
REM ================================
REM run_full_pipeline.bat
REM ================================
REM Flags soportados:
REM   --ensemble            -> usa src.trainer_ensemble_and_predict (entrena y/o predice)
REM   --mes=YYYYMM          -> override del mes a predecir (ej: --mes=202106)
REM   --threshold=0.025     -> override del umbral (ej: --threshold=0.0025)
REM   --train-only          -> con --ensemble: entrena y NO predice
REM Ejemplos:
REM   run_full_pipeline.bat --ensemble --mes=202106 --threshold=0.025
REM   run_full_pipeline.bat --mes=202106                    (pipeline clásico)
REM   run_full_pipeline.bat --ensemble --train-only         (solo entrenamiento ensamble)

REM ---------------- Args parsing ----------------
setlocal ENABLEDELAYEDEXPANSION
set ENSEMBLE=0
set TRAIN_ONLY=0
set CUSTOM_MES=
set CUSTOM_THRESHOLD=

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--ensemble" set ENSEMBLE=1& shift & goto parse_args
if /I "%~1"=="--train-only" set TRAIN_ONLY=1& shift & goto parse_args
for /f "tokens=1,2 delims==" %%A in ("%~1") do (
  if /I "%%~A"=="--mes" set CUSTOM_MES=%%~B
  if /I "%%~A"=="--threshold" set CUSTOM_THRESHOLD=%%~B
)
shift
goto parse_args
:args_done

REM ---------------- Entorno ----------------
echo === [1/5] Activando entorno Python ===========================
IF EXIST ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) ELSE (
    echo ⚠️ No se encontró el entorno virtual .venv.
    echo    Asegurate de crearlo antes de correr este script.
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

REM ---------------- Paso 2: Data prep / Features ----------------
echo ==========================================================
echo === [2/5] Generando datasets =============================
echo ==========================================================
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

REM ---------------- Paso 3: Optimizer ----------------
echo ==========================================================
echo === [3/5] Buscando hiperparametros =======================
echo ==========================================================
python -m src.optimizer
IF ERRORLEVEL 1 (
    echo [ERROR] Fallo en src.optimizer
    exit /b 1
)

REM ---------------- Paso 4 y 5 combinados (solo ENSAMBLE) ----------------
REM Si usás --ensemble y NO usás --train-only, haremos UNA sola llamada que entrena y predice.
REM En modo clásico (sin --ensemble), se mantiene la lógica anterior (trainer + predict).

REM Defaults de scoring (override por flags)
set SCORE_MES=202106
set SCORE_THRESHOLD=0.01
if defined CUSTOM_MES set SCORE_MES=%CUSTOM_MES%
if defined CUSTOM_THRESHOLD set SCORE_THRESHOLD=%CUSTOM_THRESHOLD%

echo ==========================================================
echo === [4/4] Entrenando ENSAMBLE (sin scoring) ==============
echo ==========================================================
echo python -m src.trainer_ensemble_and_predict --train
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.trainer_ensemble_and_predict --train
echo     exit /b 1
echo )

echo ==========================================================
echo === [4/4] Entrenar ENSAMBLE + Scoring en una llamada =====
echo ==========================================================
echo -> Entrenando y prediciendo mes %SCORE_MES% (thr %SCORE_THRESHOLD%)
echo python -m src.trainer_ensemble_and_predict --train --mes %SCORE_MES% --threshold %SCORE_THRESHOLD%
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.trainer_ensemble_and_predict --train --mes %SCORE_MES%
echo     exit /b 1
echo )

REM ===== Flujo clásico (trainer + predict) =====
echo ==========================================================
echo === [4/5] Entrenando modelo final (clásico) ==============
echo ==========================================================
echo python -m src.trainer
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.trainer
echo     exit /b 1
echo )

echo ==========================================================
echo === [5/5] Scoring / Prediccion mensual (clásico) =========
echo ==========================================================
echo -> Predict mes %SCORE_MES% con threshold %SCORE_THRESHOLD%
echo n -m src.predict --mes %SCORE_MES% --threshold %SCORE_THRESHOLD%
echo IF ERRORLEVEL 1 (
echo     echo [ERROR] Fallo en src.predict
echo     exit /b 1
echo )


 echo(
 echo Pipeline completo OK
 exit /b 0
)
