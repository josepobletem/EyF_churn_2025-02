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
├── src/
│   ├── data_prep.py            # genera dataset procesado con target
│   ├── feature_engineering.py  # arma features en DuckDB/SQL
│   ├── optimizer.py            # Optuna + LightGBM + gan_eval
│   ├── trainer.py              # reentrena modelo final con best_params.yaml
│   ├── predict.py              # predice modelo entrenado con best_params.yaml
│   ├── trainer_ensemble_and_predict_gcp.py
│   ├── trainer_ensemble_and_predict.py
│   └── trainer_zlgbm_canaritos.py   
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
- Limpieza, selección y formateo del dataset base (`competencia_01.csv`).
- Generación de columna target `clase_ternaria` y sus variantes binarias.
- Exporta dataset procesado:  
  `data/processed/competencia_01.csv`

### **2️⃣ Feature Engineering (`src/feature_engineering.py`)**
- Usa **DuckDB** y **SQL modular** (`sql/*.sql`) para construir features.
- Ejecuta pasos automáticos:
  - `01_base_tables.sql`
  - `02_feat_numeric.sql`
  - `03_final_join.sql`
- Genera dataset final:
  `data/processed/competencia_01_features.csv`

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

Ambos scripts hacen:
1. Activan el entorno `.venv`.
2. Ejecutan:
   - `python -m src.data_prep`
   - `python -m src.feature_engineering`
   - `python -m src.optimizer`
   - `python -m src.trainer`

---

## 🧩 Configuración (`config/config.yaml`)

Ejemplo de configuración mínima:

```yaml
paths:
  # dataset crudo original (sin target todavía)
  #raw_dataset: "data/raw/competencia_01_crudo.csv"
  raw_dataset: "data/raw/competencia_02_crudo.csv"

  # dataset con target churn/class (output de data_prep.py)
  #processed_dataset: "data/processed/competencia_01.csv"
  #processed_dataset: "data/processed/competencia_02.csv"
  processed_dataset: "gs://jose_poblete_bukito3/eyf/processed/competencia_02.parquet"

  # dataset final con features listo para entrenar (output de feature_engineering.py)
  #feature_dataset: "data/processed/competencia_01_features_new.csv"
  #feature_dataset: "data/processed/competencia_02_features_new.parquet"
  feature_dataset:   "gs://jose_poblete_bukito3/eyf/features/competencia_02_features_new.parquet"
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
  n_models: 5
  seed: 12345
  seeds: [464939, 782911, 213713, 811157, 502717, 203, 307, 409, 503, 607, 701, 809, 907, 1009,
          1103, 1201, 1301, 1409, 1501, 1601, 1709, 1801, 1901, 2003,
          782911, 101, 213713]
  decision_threshold: 0.025

  # ⚙️ NUEVO PARA ESTE SCRIPT
  models_dir: "gs://jose_poblete_bukito3/eyf/zlgbm"  # donde se guardan modelo y árboles
  kaggle_dir: "gs://jose_poblete_bukito3/eyf/kaggle" # donde se guardan archivos Kaggle

  train_months: [ #201905, 201906,
                 201907,
                 201908, 
                 201909, 
                 201910,
                 201911, 
                 201912,
                 202001,
                 #202002, 202003,
                 202004, 
                 202005, 
                 202006,
                 202007,
                 202008, 
                 202009, 202010, 202011, 202012,
                 202101, 
                 202102,
                 202103,
                 202104,
                 202105,
                 202106]

  future_months: [202108]   # como pide la consigna

  qcanaritos: 5                   # cantidad de canaritos
  experimento: "zlgbm_canarios_v1"  # sufijo para nombre KA...
  top_n_kaggle: 11500               # cantidad de envíos = 1

```

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
