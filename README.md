# 🧠 EyF Churn 2025 — Full ML Pipeline

Pipeline completo de modelado de churn con **LightGBM**, **Optuna** y **métrica de negocio personalizada**, inspirado en el enfoque de la **competencia EyF (UBA)**.

---

## 📂 Estructura del Proyecto

```
EyF_churn_2025-02/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── models/
│   ├── best_model.pkl
│   └── best_params.yaml
├── sql/
│   ├── 01_base_tables.sql
│   ├── 02_feat_numeric.sql
│   └── 03_final_join.sql
├── src/
│   ├── data_prep.py
│   ├── feature_engineering.py
│   ├── optimizer.py
│   ├── trainer.py
│   └── utils/
├── run_full_pipeline.bat
├── run_full_pipeline.sh
└── README.md
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
  raw_dataset: data/raw/competencia_01.csv
  processed_dataset: data/processed/competencia_01.csv
  feature_dataset: data/processed/competencia_01_features.csv

columns:
  id_column: numero_de_cliente
  period_column: foto_mes
  target_column_full: clase_ternaria
  target_binary_col: clase_binaria2
  peso_col: clase_peso

train:
  train_months: [202101, 202102, 202103]
  test_month: 202104
  drop_cols: ["lag_3_ctrx_quarter"]
  n_estimators: 5000
  nfold: 5
  seed: 12345
  n_trials: 30
  ganancia_acierto: 780000.0
  costo_estimulo: 20000.0
  weight_baja2: 1.00002
  weight_baja1: 1.00001
  weight_continua: 1.0
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