# Diseño — Interconnect: predicción de churn

Fecha: 2026-10-17 · Estado: aprobado en brainstorming, pendiente de revisión escrita.

Las reglas de negocio, datos y evaluación viven en `CLAUDE.md` y no se repiten aquí; este documento fija las decisiones que el `CLAUDE.md` deja abiertas.

## 1. Decisiones de entorno

| Tema | Decisión |
|---|---|
| Ejecución | Desarrollo local; el notebook también debe poder subirse a la plataforma del curso. |
| Ruta de datos | `resolve_data_dir()`: usa `/datasets/final_provider/` si existe; si no, `final_provider/` relativo a la raíz del proyecto. Al subir a la plataforma se sube la carpeta `final_provider/` junto al notebook. |
| Entorno | `.venv` dentro del proyecto + `requirements.txt` con versiones fijadas a lo realmente instalado. Python 3.14 (`py`). |
| Boosting | LightGBM. Si no instala en Python 3.14, sustituir por `HistGradientBoostingClassifier` (categóricas nativas) y documentarlo. |
| Control de versiones | `git init` local; repo **público** en GitHub (nombre propuesto `interconnect-churn`), creado con `gh` previa confirmación del nombre. Los 4 CSV (~1 MB, dataset público) se incluyen en el repo. |

## 2. Estructura

```
proyecto final/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .gitignore                # .venv/, __pycache__/, .ipynb_checkpoints/, .pytest_cache/
├── final_provider/           # contract.csv, personal.csv, internet.csv, phone.csv
├── docs/superpowers/specs/   # este documento
├── docs/plans/               # plan de implementación
├── src/__init__.py
├── src/data.py
├── src/features.py
├── tests/test_data.py
├── tests/test_features.py
└── interconnect_churn.ipynb
```

## 3. Módulos `src/`

Todas las funciones son puras (no mutan la entrada; devuelven un DataFrame nuevo).

### `src/data.py`
- `resolve_data_dir() -> Path` — ver tabla de entorno.
- `load_raw(data_dir) -> dict[str, pd.DataFrame]` — claves `contract`, `personal`, `internet`, `phone`.
- `merge_tables(raw) -> pd.DataFrame` — `contract` LEFT JOIN `personal`, `internet`, `phone` por `customerID`. Lanza `ValueError` si el resultado no tiene el mismo número de filas que `contract` o si hay `customerID` duplicados.

### `src/features.py`
- `fill_missing_services(df)` — columnas de `internet.csv` (`InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) con NaN → `"No internet"`; `MultipleLines` NaN → `"No phone"`. No elimina filas.
- `add_target(df)` — `churn = 1` si `EndDate != "No"`, si no `0`.
- `add_tenure(df, cutoff)` — `tenure_days = EndDate − BeginDate`; para activos (`EndDate == "No"`) se usa `cutoff`. El valor de `cutoff` se confirma en el EDA (se espera `2020-02-01`) y se pasa explícitamente.
- `clean_total_charges(df)` — `pd.to_numeric(errors="coerce")` y relleno con **0**. Justificación: los vacíos corresponden a clientes cuya `BeginDate` coincide con la fecha de corte y aún no han sido facturados; se verifica en el EDA antes de aplicar.
- `add_engineered(df)` — `n_services` (número de servicios adicionales con valor `"Yes"` + teléfono + internet), `has_internet`, `has_phone`.
- `build_features(df, cutoff)` — aplica en orden: `fill_missing_services` → `add_target` → `add_tenure` → `clean_total_charges` → `add_engineered`, y elimina `BeginDate`, `EndDate`, `customerID`. Ninguna feature se deriva del año/mes de `EndDate`.

La razón `TotalCharges / MonthlyCharges` no va en `src/` por defecto: se evalúa en el EDA (es casi colineal con `tenure`) y solo se añade si aporta.

### Tests (pytest, TDD)
DataFrames pequeños construidos a mano. Casos mínimos:
- cliente solo-teléfono → columnas de internet = `"No internet"`; cliente solo-internet → `MultipleLines = "No phone"`;
- `merge_tables` conserva todas las filas de `contract` y falla ante duplicados;
- `EndDate = "No"` → `churn = 0`; fecha → `churn = 1`;
- cliente activo → `tenure_days` calculado hasta `cutoff`; cliente dado de baja → hasta `EndDate`;
- `TotalCharges = " "` → `0.0`;
- `build_features` no deja `BeginDate`, `EndDate` ni `customerID`;
- las funciones no mutan su entrada.

## 4. Notebook `interconnect_churn.ipynb`

Textos y comentarios en español; `random_state=12345` en todo.

**Etapa 1 — Plan de trabajo** (markdown): unión y nulos, target, preprocesamiento por variable, modelos.

**Etapa 2.1 — EDA:** carga, tipos, duplicados, nulos antes/después del merge, confirmación de la fecha de corte (máximo de `BeginDate`/`EndDate`), balance de clases, distribuciones de `MonthlyCharges`, `TotalCharges`, `tenure_days` por churn, tasa de churn por `Type`, `InternetService`, `PaymentMethod`, anomalías (`TotalCharges` en blanco).

**Etapa 2.2 — Preprocesamiento:** `build_features`; split estratificado por `churn` 60/20/20 (train/valid/test). Dos `ColumnTransformer` dentro de `Pipeline`, ajustados solo con train:
- Regresión Logística: `OneHotEncoder(handle_unknown="ignore")` + `StandardScaler` en numéricas.
- Random Forest: `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`, numéricas sin escalar.
- LightGBM: categóricas como dtype `category` nativo.

**Etapa 2.3 — Baseline (sin balanceo):** LogReg, RandomForest, LightGBM con parámetros por defecto. Tabla con AUC-ROC y Recall (umbral 0.5) en validación.

**Etapa 2.4 — Optimización:** los mismos tres modelos con balanceo (`class_weight="balanced"`; en LightGBM `is_unbalance=True` o `scale_pos_weight`) + `RandomizedSearchCV` sobre train con `StratifiedKFold(n_splits=5, shuffle=True, random_state=12345)`, `scoring="roc_auc"`, 30–50 iteraciones por modelo. Tabla comparativa baseline vs. optimizado: AUC-ROC y Recall en validación + AUC medio de CV.

**Umbral:** sobre validación, con la curva precisión-recall, se elige el umbral que alcanza Recall ≈ 0.80 y se reporta la precisión resultante.

**Prueba final:** el modelo con mejor AUC en validación se reentrena con train + valid (mismos hiperparámetros y umbral) y se evalúa **una sola vez** en test: AUC-ROC, Recall con el umbral elegido, matriz de confusión. Gráfico simple de importancia de features.

**Etapa 3 — Informe para negocio:** modelo elegido y por qué; AUC-ROC y Recall en test; traducción del Recall ("de cada 100 clientes que se van, detectamos X; marketing contactaría a Y de cada 100 marcados que realmente se iban") y recomendación de uso.

## 5. Manejo de anomalías

- AUC en validación ≥ ~0.98 o brecha grande CV↔validación → sospecha de fuga; aplicar `superpowers:systematic-debugging` antes de seguir.
- AUC final < 0.88 → revisar features y preprocesamiento antes de ampliar la búsqueda de hiperparámetros; considerar CatBoost solo entonces.

## 6. Flujo de trabajo

Cada etapa se cierra con commit y requiere confirmación del usuario para pasar a la siguiente:

1. Setup: git, `.gitignore`, `.venv`, `requirements.txt`, README.
2. `src/` + `tests/` con TDD; `pytest` en verde.
3. Notebook: Etapa 1, 2.1, 2.2; ejecución completa.
4. Notebook: Etapa 2.3, 2.4; ejecución completa.
5. Prueba final + Etapa 3; ejecución limpia con `jupyter nbconvert --execute`.
6. GitHub: confirmar nombre, `gh repo create` público, push.

Verificación: ninguna métrica se escribe a mano; las cifras del informe se redactan después de ver la salida real de test. Revisión con `superpowers:requesting-code-review` al cerrar cada etapa.

## 7. Fuera de alcance

Despliegue, API, dashboard, SHAP extenso, CatBoost (salvo que LightGBM no alcance la meta).
