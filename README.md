# 🧠 EyF Churn 2025 — Full ML Pipeline

Pipeline completo de modelado de churn con **LightGBM**, **Optuna** y **métrica de negocio personalizada**, inspirado en el enfoque de la **competencia EyF (UBA)**.

---

## 📂 Estructura del Proyecto

```
EyF_churn_2025-02/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                # datos crudos originales
│   ├── processed/          # datos limpios + target + features finales combinadas
│   ├── features/           # (opcional) dumps intermedios de features
│   ├── test/               # datos de holdout / scoring (mes test_month)
│   └── README_data.md      # (opcional) descripción de datasets
├── models/
│   ├── best_model.pkl          # modelo candidato encontrado por optimizer
│   ├── best_params.yaml        # hiperparámetros óptimos + metadata
│   ├── final_model.pkl         # modelo final reentrenado en train_months
│   └── final_metrics.yaml      # métricas in-sample finales
├── sql/
│   ├── 01_base_tables.sql
│   ├── 02_feat_numeric.sql
│   ├── 03_final_join.sql
│   ├── 04_risk_behavior_and_join.sql
│   └── 05_behavioral_features.sql
├── notebook/
│   ├── append_month_09_comp_03.ipynb
│   ├── Driff_outliers_calculate.ipynb
│   └── validate_comp_03.ipynb
├── src/
│   ├── data_prep.py            # genera dataset procesado con target
│   ├── feature_engineering.py  # arma features en DuckDB/SQL
│   ├── optimizer.py            # Optuna + LightGBM + gan_eval
│   ├── trainer.py              # reentrena modelo final con best_params.yaml
│   ├── predict.py              # predice modelo entrenado con best_params.yaml
│   ├── trainer_ensemble_and_predict_gcp.py
│   ├── trainer_ensemble_and_predict.py
│   ├── trainer_zlgbm_canaritos.py # comp_02
│   ├── feature_engineering_polars_comp03.py  # Arma features con polars
│   ├── feature_engineering_polars_EM_MAX_MIN_comp03.py # Arma features con polars
│   └── trainer_zlgbm_canaritos_undersampling_polar_comp03.py # entrena ensamblando y predice
├── tests/
│   ├── test_data_prep.py           # prueba carga y procesamiento inicial
│   ├── test_feature_engineering.py # prueba consultas SQL y features generadas
│   ├── test_optimizer.py           # prueba búsqueda de hiperparámetros
│   ├── test_trainer.py             # prueba entrenamiento final y guardado de modelo
│   └── conftest.py (opcional)      # configuración común de pytest (fixtures)
├── run_full_pipeline.bat
├── run_full_pipeline.sh
├── README.md
├── Makefile
└── requirements.txt

```

---

## 🚀 Descripción del Pipeline

El pipeline consta de **4 etapas principales** ejecutadas secuencialmente:

### **1️⃣ Data Preparation (`src/data_prep.py`)**
- Limpieza, selección y formateo del dataset base (`competencia_01.csv`, `competencia_02_crudo.csv`, `competencia_03_crudo_append.parquet`).
- Generación de columna target `clase_ternaria` y sus variantes binarias.
- Exporta dataset procesado:  
  `data/processed/competencia_01.csv`
  `gs://jose_poblete_bukito3/eyf/processed/competencia_02.parquet`
  `gs://jose_poblete_bukito3/eyf/processed/competencia_03.parquet`

### **2️⃣ Feature Engineering (`src/feature_engineering.py`)**
- Usa **DuckDB** y **SQL modular** (`sql/*.sql`) para construir features.
- Ejecuta pasos automáticos:
  - `01_base_tables.sql`
  - `02_feat_numeric.sql`
  - `03_final_join.sql`
  - `04_risk_behavior_and_join.sql`
  - `05_behavioral_features.sql`

- Genera dataset final:
  `data/processed/competencia_01_features.csv`
  `gs://jose_poblete_bukito3/eyf/processed/competencia_02_features_new.parquet`
  `gs://jose_poblete_bukito3/eyf/processed/competencia_03_features_polar.parquet`

💡 Si el script lanza un error tipo:
```
FileNotFoundError: No se encontró el archivo SQL sql/03_final_join.sql
```
→ asegurate de crear ese archivo SQL con la unión final de features.

---

### **3️⃣ Optuna + LightGBM (`src/optimizer.py`)**

Busca los **mejores hiperparámetros** mediante **Optuna**, optimizando una métrica de negocio propia (`gan_eval`).

#### 🔹 Lógica de partición:
```python
mes_train = [202101, 202102, 202103]
mes_test = 202104
```

#### 🔹 Métrica de negocio:
```python
ganancia_acierto = 780000
costo_estimulo = 20000

def lgb_gan_eval(y_pred, data):
    weight = data.get_weight()
    ganancia = np.where(weight == 1.00002, ganancia_acierto, 0)              - np.where(weight < 1.00002, costo_estimulo, 0)
    ganancia = np.cumsum(ganancia[np.argsort(y_pred)[::-1]])
    return 'gan_eval', np.max(ganancia), True
```

#### 🔹 Espacio de búsqueda (Optuna):
Explora un rango amplio de parámetros:

```python
params = {
    "learning_rate": trial.suggest_float(5e-4, 0.2, log=True),
    "num_leaves": trial.suggest_int(16, 512),
    "max_depth": trial.suggest_int(-1, 16),
    "min_data_in_leaf": trial.suggest_int(5, 2000),
    "feature_fraction": trial.suggest_float(0.4, 1.0),
    "bagging_fraction": trial.suggest_float(0.5, 1.0),
    "bagging_freq": trial.suggest_int(0, 10),
    "min_split_gain": trial.suggest_float(0.0, 1.0),
    "lambda_l1": trial.suggest_float(1e-8, 10.0, log=True),
    "lambda_l2": trial.suggest_float(1e-8, 10.0, log=True),
    "scale_pos_weight": trial.suggest_float(0.5, 10.0, log=True),
}
```

#### 🔹 Evaluación en CV:
- `nfold = 5`  
- `num_boost_round = 5000`  
- `early_stopping_rounds = 200`

LightGBM imprime algo como:
```
[500] valid's gan_eval: 1.7424e+08 + 1.09826e+07
```
👉 Promedio ± desviación entre los 5 folds.

---

### **4️⃣ Entrenamiento final (`src/trainer.py`)**
- Usa los **mejores hiperparámetros** encontrados por Optuna.
- Reentrena con todos los datos de entrenamiento.
- Guarda:
  - `models/best_model.pkl`
  - `models/best_params.yaml`

---

## ⚙️ Ejecución del Pipeline

### 🔸 En Windows
```bash
run_full_pipeline.bat
```

### 🔸 En Linux / WSL
```bash
bash run_full_pipeline.sh
```

### Ambos scripts hacen para comp 01:

1. Activan el entorno `.venv`.
2. Ejecutan:
   - `python -m src.data_prep`
   - `python -m src.feature_engineering`
   - `python -m src.optimizer`
   - `python -m src.trainer`
  
### Para competencia 02 `run_full_pipeline.sh` :

1. Activan el entorno `.venv`.
2. Instalar zlgbm:
```
###################zlgbm ########################
cd
rm -rf  LightGBM
git clone --recursive  https://github.com/dmecoyfin/LightGBM

source  ~/.venv/bin/activate
pip uninstall --yes lightgbm

# instalacion Python
cd  ~/LightGBM
sh ./build-python.sh  install
```

3. Ejecutan:
   - `python -m src.data_prep`
   - `python -m src.feature_engineering`
   - `python -m src.trainer_zlgbm_canaritos`
  
### Para competencia 03 `run_full_pipeline.sh` :

1. Activan el entorno `.venv`.
2. Instalar zlgbm:
```
###################zlgbm ########################
cd
rm -rf  LightGBM
git clone --recursive  https://github.com/dmecoyfin/LightGBM

source  ~/.venv/bin/activate
pip uninstall --yes lightgbm

# instalacion Python
cd  ~/LightGBM
sh ./build-python.sh  install
```

3. Ejecutan:
   - `python -m src.data_prep`
   - `python -m src.feature_engineering_polars_comp03.py`
   - `python -m src.trainer_zlgbm_canaritos_undersampling_polar_comp03`

---

## 🧩 Configuración (`config/config.yaml`)

Ejemplo de configuración mínima:

```yaml
paths:
  # dataset crudo original (sin target todavía)
  #raw_dataset: "data/raw/competencia_01_crudo.csv"
  #raw_dataset: "data/raw/competencia_02_crudo.csv"
  raw_dataset: "data/raw/competencia_03_crudo_append.parquet"

  # dataset con target churn/class (output de data_prep.py)
  #processed_dataset: "data/processed/competencia_01.csv"
  #processed_dataset: "data/processed/competencia_02.csv"
  #processed_dataset: "gs://jose_poblete_bukito3/eyf/processed/competencia_02.parquet"
  processed_dataset: "gs://jose_poblete_bukito3/eyf/processed/competencia_03.parquet"

  # dataset final con features listo para entrenar (output de feature_engineering.py)
  #feature_dataset: "data/processed/competencia_01_features_new.csv"
  #feature_dataset: "data/processed/competencia_02_features_new.parquet"
  #feature_dataset:   "gs://jose_poblete_bukito3/eyf/features/competencia_02_features_new.parquet"
  #feature_dataset:   "gs://jose_poblete_bukito3/eyf/features/competencia_03_features.parquet"
  feature_dataset:   "gs://jose_poblete_bukito3/eyf/features/competencia_03_features_polar.parquet"
  #feature_dataset:   "gs://jose_poblete_bukito3/eyf/features/competencia_03_features_polar_EM_max_min.parquet"


columns:
  # identificador único del cliente
  id_column: "numero_de_cliente"

  # periodo tipo YYYYMM (por ejemplo 202104)
  period_column: "foto_mes"

  # target creado en data_prep.py: BAJA+1 / BAJA+2 / CONTINUA
  target_column: "clase_ternaria"
  
  # target para el optimizador
  binary_target_col: "clase_binaria2"
  peso_col: "clase_peso"
  binary_target_gan: "clase_binaria1"

logic:
  # Documentación de negocio de churn
  churn_definition: |
    CASE
      WHEN esta_t1 = 0 THEN 'BAJA+1'
      WHEN esta_t1 = 1 AND esta_t2 = 0 THEN 'BAJA+2'
      ELSE 'CONTINUA'
    END
  time_granularity: "mes"

features:
  # nombre con el que vamos a registrar el dataset base en DuckDB
  # (es el processed_dataset leído por Python)
  base_table_name: "base_clientes"

  # orden de ejecución de los SQL
  steps:
    - "sql/01_base_tables.sql"
    - "sql/02_feat_numeric.sql"
    - "sql/03_final_model.sql"
    - "sql/04_risk_behavior_and_join.sql"
    - "sql/05_behavioral_features.sql"

train:
  # ya los tenías (pueden quedar aunque no se usen aquí)
  n_models: 50
  seed: 12345
  seeds: [509963, 464939, 782911, 213713, 811157, 502717,
          736421, 592889, 463301, 889147, 219787,
          713409, 982331, 627519, 854773, 431287,
          918445, 376201, 745691, 269553, 1009,
          804221, 912557, 664981, 335729, 589043,
          721843, 477119, 839467, 909311, 552883,
          934211, 128993, 671521, 347899, 558913,
          203, 307, 409, 503, 607,
          701, 809, 907, 1103, 1201,
          1301, 1409, 1501, 1601, 1709,
          1801, 1901, 2003, 101, 915773,
          28455, 638251, 970433, 542719, 881077,
          309641, 752389, 194723, 82651, 46839,
          903227, 255719, 619847, 784203, 331907,
          867541, 592301, 447983, 713521, 988163,
          524197,230101,231019, 675439, 812129, 456701
          ]
  decision_threshold: 0.025

  # ⚙️ NUEVO PARA ESTE SCRIPT DE ZLightGBM CON CANARITOS
  #models_dir: "gs://jose_poblete_bukito3/eyf/zlgbm"  # donde se guardan modelo y árboles
  #kaggle_dir: "gs://jose_poblete_bukito3/eyf/kaggle" # donde se guardan archivos Kaggle

  # Path para bayesinan optimization
  # models_dir: "gs://jose_poblete_bukito3/eyf/models"  

  train_months: [
                 #201907, #Sacar no undersampling
                 #201908, #Sacar no undersampling
                 #201909, #Sacar no undersampling
                 201910, #drift
                 201911, 
                 201912,
                 202001,
                 202002, #zlgbm se saca
                 202003, #zlgbm se saca
                 202004, 
                 202005, 
                 #202006, #drift y outliers
                 #202007, # outliers
                 202008, 
                 202009,
                 202010, 
                 202011,
                 #202012, # outliers
                 202101, 
                 202102,
                 202103,
                 202104,
                 202105,
                 #202106, # HASTA AQUI COMP 02
                 #202107 # Hasta aqui comp 03
                 ]

  future_months: [202107]   # como pide la consigna

  qcanaritos: 20                   # cantidad de canaritos
  experimento: "zlgbm_canarios_v1_comp03"  # sufijo para nombre KA...
  top_n_kaggle: 11500               # cantidad de envíos = 1

```

## 🏆 Competencia 03 (comp_03) — Feature Engineering y Entrenamiento Especializado

Para la Competencia 03 se utilizan scripts específicos que reemplazan al pipeline estándar de feature engineering y entrenamiento. Estos scripts están optimizados para manejar drift, outliers, lags avanzados, percentiles, suavizamientos exponenciales, canaritos, semillero extendido y undersampling controlado.

### ⚙️ 1️⃣ Feature Engineering para Comp_03


Script: src/feature_engineering_polars_comp03.py

Ejecuta:


```bash
python -m src.feature_engineering_polars_comp03
```

Este módulo realiza:

 - Generación de features usando Polars (alta velocidad y bajo consumo de memoria).

 - Agregados avanzados:
 
    - Lags adicionales

    - Lags de segundo orden

    - Diferencias (deltas)

 - Filtrado automático de meses conflictivos (drift y outliers detectados previamente).

 - Exportación del dataset final a:


```bash
gs://jose_poblete_bukito3/eyf/features/competencia_03_features_polar.parquet
```

### 🤖 2️⃣ Entrenamiento y Predicción Comp_03

Script: src/trainer_zlgbm_canaritos_undersampling_polar_comp03.py

Ejecuta:

```bash
python -m src.trainer_zlgbm_canaritos_undersampling_polar_comp03
```

Este script incorpora:

 - ZLightGBM con mejoras robustas.

 - Ensamble de múltiples modelos mediante semillas distintas.

 - Undersampling inteligente en meses seleccionados.

 - Canaritos para regularización por ruido.

 - Uso del dataset generado por Polars.

 - Optimización automática basada en la métrica de negocio gan_eval.

Entrega como salida:

 - Modelos entrenados
 
 - Predicciones para el mes futuro (202107)

 - Archivo final para Kaggle:

```bash
pred_zlgbm_{zcfg.experimento}_{top_n}_202105_minleaf50_20can_50seed_22month_gb01_prestamosless_driff1_0203less_final_Julio_semillero_undersampling6_polar_drift_out_less.csv
```

### 🧩 Resumen Comp_03

| Etapa                    | Script                                                  | Descripción                                          |
| ------------------------ | ------------------------------------------------------- | ---------------------------------------------------- |
| Feature Engineering      | `feature_engineering_polars_comp03.py`                  | Construcción avanzada de features con Polars         |
| Entrenamiento + Ensamble | `trainer_zlgbm_canaritos_undersampling_polar_comp03.py` | Modelo ZLGBM + undersampling + canaritos + semillero |


---

## 📊 Métricas y Logs

Los resultados de CV se muestran en consola, por ejemplo:

```
[1000] valid's gan_eval: 1.80624e+08 + 1.04733e+07
[1500] valid's gan_eval: 1.84200e+08 + 7.59254e+06
CV OK. Mejor gan_eval(mean)=184.2M ± 7.6M en iter=1500
```

Los artefactos se guardan en:
- `models/best_model.pkl`
- `models/best_params.yaml`

---

## 🧠 Interpretación de la Métrica `gan_eval`

| Valor | Significado |
|--------|--------------|
| `1.842e+08` | Ganancia promedio (media entre folds) |
| `± 7.592e+06` | Variabilidad entre folds |
| Intervalo estimado | [176M, 192M] |
| Variabilidad relativa | ~4.1% → modelo estable |

---

## 🛠️ Dependencias Principales

| Librería | Uso |
|-----------|------|
| `pandas` | Manipulación de datos |
| `numpy` | Cálculos numéricos |
| `lightgbm` | Modelo de boosting |
| `optuna` | Optimización bayesiana |
| `duckdb` | Feature engineering con SQL |
| `pyyaml` | Configuración |
| `pydantic` | Validación de config |
| `logging` | Trazabilidad de pipeline |

---

## 💾 Resultados esperados

Al finalizar, el pipeline genera:
- Dataset enriquecido (`data/processed/competencia_01_features.csv`)
- Modelo LightGBM ajustado a la métrica de negocio (`models/best_model.pkl`)
- Parámetros óptimos (`models/best_params.yaml`)
- Logs detallados en salida estándar.

---

## 👤 Autor

**José Poblete M.**  
Data Scientist & MLOps Engineer  
Facultad de Ciencias Naturales — UBA.
