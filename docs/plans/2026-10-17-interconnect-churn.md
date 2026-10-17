# Interconnect Churn — Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir el notebook `interconnect_churn.ipynb` que predice la baja de clientes de Interconnect (AUC-ROC en test ≥ 0.88), apoyado en funciones probadas en `src/`.

**Architecture:** La carga, la unión y las features viven en `src/data.py` y `src/features.py`, que son funciones puras probadas con pytest (TDD). El notebook importa esas funciones y hace EDA, split 60/20/20, modelos (LogReg / RandomForest / LightGBM), balanceo + `RandomizedSearchCV`, elección de umbral en validación, una única evaluación en test e informe para negocio.

**Tech Stack:** Python 3.14 (`py`), `.venv`, pandas, numpy, scikit-learn, lightgbm, scipy, matplotlib, seaborn, pytest, jupyter/nbconvert, nbformat.

Spec: `docs/superpowers/specs/2026-10-17-interconnect-churn-design.md`. Reglas: `CLAUDE.md`.

## Global Constraints

- Todo el texto del notebook, los comentarios y los mensajes al usuario van en **español**.
- `random_state=12345` en todo (constante `RANDOM_STATE = 12345`).
- Unión: `contract` LEFT JOIN `personal`, `internet`, `phone` por `customerID`; los NaN tras la unión indican ausencia de servicio (`"No internet"` / `"No phone"`) y no se eliminan filas.
- Target: `churn = 1` si `EndDate` es una fecha, `0` si `EndDate == "No"`.
- `tenure_days = EndDate − BeginDate`; para clientes activos se usa la fecha de corte `2020-02-01` (verificada: máx. `BeginDate` = 2020-02-01).
- Después de crear `churn` y `tenure_days` se eliminan `BeginDate`, `EndDate` y `customerID`. Ninguna feature sale del año o mes de `EndDate`.
- `TotalCharges`: `pd.to_numeric(errors="coerce")` y relleno con 0. Los 11 vacíos tienen `BeginDate == 2020-02-01`.
- Split estratificado por `churn`, 60/20/20, `random_state=12345`. **El test se usa una sola vez**, al final.
- Encoders y escaladores se ajustan solo con train, dentro de `Pipeline` / `ColumnTransformer`.
- Métrica principal: AUC-ROC en test (meta ≥ 0.88). Secundaria: Recall, con el umbral reportado.
- **Nunca se escribe una métrica a mano**: las cifras del texto salen de la ejecución de esta sesión.
- Para pasar de etapa (marcado como **CHECKPOINT**) hay que pedir confirmación al usuario y hacer antes la revisión con `superpowers:requesting-code-review`.
- Todos los comandos se ejecutan desde la raíz del proyecto: `D:\Curso Ciencia de Datos\proyecto final` (en Bash: `/d/Curso Ciencia de Datos/proyecto final`).
- Commits con la línea final `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `requirements.txt` | Dependencias con versión fijada |
| `pytest.ini` | `pythonpath = .` para que `tests/` importe `src` |
| `README.md` | Qué es el proyecto, instalación y ejecución |
| `src/__init__.py` | Paquete vacío |
| `src/data.py` | `resolve_data_dir`, `load_raw`, `merge_tables` |
| `src/features.py` | `fill_missing_services`, `add_target`, `add_tenure`, `clean_total_charges`, `add_engineered`, `build_features` |
| `tests/test_data.py` | Tests de `src/data.py` |
| `tests/test_features.py` | Tests de `src/features.py` |
| `interconnect_churn.ipynb` | Entregable (Etapas 1–3) |

---

### Task 1: Entorno y esqueleto del proyecto

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `README.md`, `src/__init__.py`

**Interfaces:**
- Produces: `.venv` con todas las dependencias; los comandos `.venv/Scripts/python -m pytest` y `.venv/Scripts/jupyter` funcionan.

- [ ] **Step 1: Crear el entorno virtual e instalar dependencias**

```bash
cd "/d/Curso Ciencia de Datos/proyecto final"
py -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install pandas numpy scikit-learn scipy lightgbm matplotlib seaborn pytest jupyter nbconvert nbformat ipykernel
```

Si `lightgbm` falla al instalar en Python 3.14, **parar** y aplicar el fallback del spec: sustituir LightGBM por `sklearn.ensemble.HistGradientBoostingClassifier(categorical_features="from_dtype")` en las Tasks 6–7 y avisar al usuario.

- [ ] **Step 2: Verificar las importaciones**

Run: `.venv/Scripts/python -c "import pandas, numpy, sklearn, scipy, lightgbm, matplotlib, seaborn, pytest, nbformat; print('ok', pandas.__version__, sklearn.__version__, lightgbm.__version__)"`
Expected: `ok <versiones>`

- [ ] **Step 3: Escribir `requirements.txt` con las versiones reales**

```bash
.venv/Scripts/python - <<'EOF'
from importlib.metadata import version
pkgs = ["pandas", "numpy", "scikit-learn", "scipy", "lightgbm", "matplotlib",
        "seaborn", "pytest", "jupyter", "nbconvert", "nbformat", "ipykernel"]
open("requirements.txt", "w").write("".join(f"{p}=={version(p)}\n" for p in pkgs))
EOF
cat requirements.txt
```

- [ ] **Step 4: Crear `pytest.ini` y `src/__init__.py`**

`pytest.ini`:
```ini
[pytest]
pythonpath = .
testpaths = tests
```

`src/__init__.py`: archivo vacío.

- [ ] **Step 5: Crear `README.md`**

~~~markdown
# Interconnect — Predicción de churn

Proyecto final del curso de Ciencia de Datos. Modelo de clasificación binaria que predice qué clientes de la telecom Interconnect se darán de baja, para que marketing les ofrezca promociones a tiempo.

**Métrica principal:** AUC-ROC en test (meta ≥ 0.88). **Secundaria:** Recall.

## Estructura

- `final_provider/` — datos crudos (`contract`, `personal`, `internet`, `phone`).
- `src/` — carga, unión e ingeniería de features (probado en `tests/`).
- `interconnect_churn.ipynb` — EDA, modelado, evaluación e informe para negocio.
- `docs/` — diseño y plan de implementación.

## Cómo ejecutar

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/jupyter notebook interconnect_churn.ipynb
```

En la plataforma del curso: subir el notebook junto con las carpetas `src/` y `final_provider/`. Si existe `/datasets/final_provider/`, el notebook usa esa ruta automáticamente.
~~~

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini README.md src/__init__.py
git commit -m "chore: set up venv requirements, pytest config and README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `src/data.py` — carga y unión

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: nada.
- Produces:
  - `TABLES: tuple[str, ...] = ("contract", "personal", "internet", "phone")`
  - `resolve_data_dir(platform_dir: Path = PLATFORM_DIR, local_dir: Path = LOCAL_DIR) -> Path`
  - `load_raw(data_dir: Path) -> dict[str, pd.DataFrame]` (claves = `TABLES`)
  - `merge_tables(raw: dict[str, pd.DataFrame]) -> pd.DataFrame` (lanza `ValueError` si hay IDs duplicados o si cambia el número de filas)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_data.py`:
```python
import pandas as pd
import pytest

from src.data import TABLES, load_raw, merge_tables, resolve_data_dir


def make_raw():
    return {
        "contract": pd.DataFrame({"customerID": ["A", "B", "C"],
                                  "MonthlyCharges": [10.0, 20.0, 30.0]}),
        "personal": pd.DataFrame({"customerID": ["A", "B", "C"],
                                  "gender": ["Male", "Female", "Male"]}),
        # B no tiene internet
        "internet": pd.DataFrame({"customerID": ["A", "C"],
                                  "InternetService": ["DSL", "Fiber optic"]}),
        # C no tiene teléfono
        "phone": pd.DataFrame({"customerID": ["A", "B"],
                               "MultipleLines": ["No", "Yes"]}),
    }


def test_merge_keeps_all_contract_rows():
    df = merge_tables(make_raw())
    assert list(df["customerID"]) == ["A", "B", "C"]


def test_merge_leaves_nan_for_missing_services():
    df = merge_tables(make_raw()).set_index("customerID")
    assert pd.isna(df.loc["B", "InternetService"])
    assert pd.isna(df.loc["C", "MultipleLines"])
    assert df.loc["A", "InternetService"] == "DSL"


def test_merge_raises_on_duplicate_ids():
    raw = make_raw()
    raw["internet"] = pd.concat([raw["internet"], raw["internet"].iloc[[0]]])
    with pytest.raises(ValueError):
        merge_tables(raw)


def test_resolve_prefers_platform_dir(tmp_path):
    platform, local = tmp_path / "platform", tmp_path / "local"
    platform.mkdir()
    assert resolve_data_dir(platform, local) == platform


def test_resolve_falls_back_to_local_dir(tmp_path):
    platform, local = tmp_path / "platform", tmp_path / "local"
    assert resolve_data_dir(platform, local) == local


def test_load_raw_reads_all_tables(tmp_path):
    for name in TABLES:
        pd.DataFrame({"customerID": ["A"]}).to_csv(tmp_path / f"{name}.csv", index=False)
    raw = load_raw(tmp_path)
    assert set(raw) == set(TABLES)
    assert all(len(t) == 1 for t in raw.values())


def test_real_data_merge_has_one_row_per_contract():
    raw = load_raw(resolve_data_dir())
    df = merge_tables(raw)
    assert len(df) == len(raw["contract"]) == 7043
    assert df["customerID"].is_unique
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: error de colección `ModuleNotFoundError: No module named 'src.data'`.

- [ ] **Step 3: Implementar `src/data.py`**

```python
"""Carga y unión de las tablas de Interconnect."""
from pathlib import Path

import pandas as pd

PLATFORM_DIR = Path("/datasets/final_provider")
LOCAL_DIR = Path(__file__).resolve().parent.parent / "final_provider"
TABLES = ("contract", "personal", "internet", "phone")


def resolve_data_dir(platform_dir: Path = PLATFORM_DIR, local_dir: Path = LOCAL_DIR) -> Path:
    """Usa la ruta de la plataforma si existe; si no, la carpeta local del proyecto."""
    return platform_dir if Path(platform_dir).exists() else Path(local_dir)


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Lee los cuatro CSV y los devuelve en un diccionario por nombre de tabla."""
    return {name: pd.read_csv(Path(data_dir) / f"{name}.csv") for name in TABLES}


def merge_tables(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """contract LEFT JOIN personal, internet, phone por customerID."""
    for name in TABLES:
        if raw[name]["customerID"].duplicated().any():
            raise ValueError(f"customerID duplicado en la tabla '{name}'")
    contract = raw["contract"]
    df = contract
    for name in TABLES[1:]:
        df = df.merge(raw[name], on="customerID", how="left")
    if len(df) != len(contract):
        raise ValueError("La unión cambió el número de filas de contract")
    return df
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: add data loading and left-join merge with integrity checks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `src/features.py` — limpieza, target y tenure

**Files:**
- Create: `src/features.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: el DataFrame unido con las columnas de `merge_tables` (los nombres originales de los CSV).
- Produces:
  - `INTERNET_COLS: list[str]` (7 columnas de `internet.csv`) y `ADDON_COLS: list[str]` (las 6 columnas de servicios adicionales)
  - `fill_missing_services(df) -> pd.DataFrame`
  - `add_target(df) -> pd.DataFrame` (agrega `churn: int`)
  - `add_tenure(df, cutoff: str) -> pd.DataFrame` (agrega `tenure_days: int`)
  - `clean_total_charges(df) -> pd.DataFrame` (`TotalCharges: float`)

- [ ] **Step 1: Escribir los tests que fallan**

`tests/test_features.py`:
```python
import numpy as np
import pandas as pd

from src.features import (
    add_target,
    add_tenure,
    clean_total_charges,
    fill_missing_services,
)

CUTOFF = "2020-02-01"


def merged_sample():
    """A: internet + teléfono, activo. B: solo teléfono, dado de baja. C: solo internet, nuevo."""
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "BeginDate": ["2019-01-01", "2019-06-01", "2020-02-01"],
        "EndDate": ["No", "2019-12-01 00:00:00", "No"],
        "Type": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
        "MonthlyCharges": [70.0, 20.0, 50.0],
        "TotalCharges": ["840.0", "120.0", " "],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "InternetService": ["Fiber optic", np.nan, "DSL"],
        "OnlineSecurity": ["Yes", np.nan, "No"],
        "OnlineBackup": ["No", np.nan, "No"],
        "DeviceProtection": ["No", np.nan, "No"],
        "TechSupport": ["No", np.nan, "No"],
        "StreamingTV": ["Yes", np.nan, "No"],
        "StreamingMovies": ["Yes", np.nan, "No"],
        "MultipleLines": ["Yes", "No", np.nan],
    })


def test_fill_missing_services_marks_absent_internet():
    df = fill_missing_services(merged_sample()).set_index("customerID")
    assert df.loc["B", "InternetService"] == "No internet"
    assert df.loc["B", "StreamingMovies"] == "No internet"
    assert df.loc["A", "InternetService"] == "Fiber optic"


def test_fill_missing_services_marks_absent_phone():
    df = fill_missing_services(merged_sample()).set_index("customerID")
    assert df.loc["C", "MultipleLines"] == "No phone"
    assert df.loc["A", "MultipleLines"] == "Yes"


def test_fill_missing_services_keeps_rows_and_leaves_no_nan():
    df = fill_missing_services(merged_sample())
    assert len(df) == 3
    assert not df.isna().any().any()


def test_add_target():
    df = add_target(merged_sample())
    assert list(df["churn"]) == [0, 1, 0]


def test_add_tenure_uses_cutoff_for_active_and_enddate_for_churned():
    df = add_tenure(merged_sample(), CUTOFF)
    # A: 2019-01-01 -> 2020-02-01; B: 2019-06-01 -> 2019-12-01; C: mismo día
    assert list(df["tenure_days"]) == [396, 183, 0]


def test_clean_total_charges_converts_blank_to_zero():
    df = clean_total_charges(merged_sample())
    assert df["TotalCharges"].tolist() == [840.0, 120.0, 0.0]
    assert pd.api.types.is_float_dtype(df["TotalCharges"])


def test_functions_do_not_mutate_input():
    df = merged_sample()
    before = df.copy()
    fill_missing_services(df)
    add_target(df)
    add_tenure(df, CUTOFF)
    clean_total_charges(df)
    pd.testing.assert_frame_equal(df, before)
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: error de colección `ModuleNotFoundError: No module named 'src.features'`.

- [ ] **Step 3: Implementar `src/features.py` (primera parte)**

```python
"""Limpieza, target e ingeniería de features para el churn de Interconnect."""
import pandas as pd

ADDON_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
INTERNET_COLS = ["InternetService", *ADDON_COLS]


def fill_missing_services(df: pd.DataFrame) -> pd.DataFrame:
    """NaN tras la unión = el cliente no tiene ese servicio."""
    out = df.copy()
    out[INTERNET_COLS] = out[INTERNET_COLS].fillna("No internet")
    out["MultipleLines"] = out["MultipleLines"].fillna("No phone")
    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """churn = 1 si EndDate es una fecha, 0 si es 'No'."""
    out = df.copy()
    out["churn"] = (out["EndDate"] != "No").astype(int)
    return out


def add_tenure(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Días de permanencia; los clientes activos se miden hasta la fecha de corte."""
    out = df.copy()
    begin = pd.to_datetime(out["BeginDate"], format="ISO8601")
    end = pd.to_datetime(out["EndDate"].mask(out["EndDate"] == "No"), format="ISO8601")
    end = end.fillna(pd.Timestamp(cutoff))
    out["tenure_days"] = (end - begin).dt.days.astype(int)
    return out


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """TotalCharges vacío = cliente que ingresó en la fecha de corte y aún no paga: 0."""
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    return out
```

- [ ] **Step 4: Ejecutar los tests y confirmar que pasan**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add service fill, churn target, tenure and TotalCharges cleaning

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `src/features.py` — features nuevas y `build_features`

**Files:**
- Modify: `src/features.py` (agregar al final)
- Test: `tests/test_features.py` (agregar tests)

**Interfaces:**
- Consumes: las funciones de la Task 3 y `load_raw`, `merge_tables`, `resolve_data_dir` de la Task 2 (en el test de integración).
- Produces:
  - `add_engineered(df) -> pd.DataFrame`: agrega `has_internet: int`, `has_phone: int` y `n_services: int`. Requiere que antes se haya aplicado `fill_missing_services`.
  - `DROP_COLS = ["customerID", "BeginDate", "EndDate"]`
  - `build_features(df, cutoff: str) -> pd.DataFrame`: recibe la salida de `merge_tables` y devuelve las features más `churn`, sin `DROP_COLS` y sin NaN.

- [ ] **Step 1: Agregar los tests que fallan a `tests/test_features.py`**

Sustituir el import del inicio por:
```python
from src.data import load_raw, merge_tables, resolve_data_dir
from src.features import (
    DROP_COLS,
    add_engineered,
    add_target,
    add_tenure,
    build_features,
    clean_total_charges,
    fill_missing_services,
)
```

Agregar al final:
```python
def test_add_engineered_counts_services():
    df = add_engineered(fill_missing_services(merged_sample()))
    # A: 3 servicios adicionales + teléfono + internet; B: solo teléfono; C: solo internet
    assert list(df["n_services"]) == [5, 1, 1]
    assert list(df["has_internet"]) == [1, 0, 1]
    assert list(df["has_phone"]) == [1, 1, 0]


def test_build_features_drops_leaky_and_id_columns():
    df = build_features(merged_sample(), CUTOFF)
    for col in DROP_COLS:
        assert col not in df.columns
    assert {"churn", "tenure_days", "n_services"} <= set(df.columns)
    assert not df.isna().any().any()


def test_build_features_does_not_mutate_input():
    df = merged_sample()
    before = df.copy()
    build_features(df, CUTOFF)
    pd.testing.assert_frame_equal(df, before)


def test_build_features_on_real_data():
    df = build_features(merge_tables(load_raw(resolve_data_dir())), CUTOFF)
    assert len(df) == 7043
    assert not df.isna().any().any()
    assert set(df["churn"].unique()) == {0, 1}
    assert (df["tenure_days"] >= 0).all()
```

- [ ] **Step 2: Ejecutar los tests y confirmar que fallan**

Run: `.venv/Scripts/python -m pytest tests/test_features.py -v`
Expected: error de colección `ImportError: cannot import name 'DROP_COLS'`.

- [ ] **Step 3: Agregar la implementación al final de `src/features.py`**

```python
DROP_COLS = ["customerID", "BeginDate", "EndDate"]


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    """Indicadores de servicio y número total de servicios contratados."""
    out = df.copy()
    out["has_internet"] = (out["InternetService"] != "No internet").astype(int)
    out["has_phone"] = (out["MultipleLines"] != "No phone").astype(int)
    out["n_services"] = (
        (out[ADDON_COLS] == "Yes").sum(axis=1) + out["has_internet"] + out["has_phone"]
    ).astype(int)
    return out


def build_features(df: pd.DataFrame, cutoff: str) -> pd.DataFrame:
    """Aplica toda la preparación y quita ID y fechas (para evitar fugas)."""
    out = fill_missing_services(df)
    out = add_target(out)
    out = add_tenure(out, cutoff)
    out = clean_total_charges(out)
    out = add_engineered(out)
    return out.drop(columns=DROP_COLS)
```

- [ ] **Step 4: Ejecutar toda la suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: 18 passed (7 de data + 11 de features).

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: add engineered service features and build_features pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: CHECKPOINT.** Hacer la revisión con `superpowers:requesting-code-review` sobre las Tasks 1–4, mostrar la salida de `pytest` y pedir confirmación para pasar al notebook.

---

### Task 5: Notebook — Etapa 1 (plan), EDA y preprocesamiento

**Files:**
- Create: `interconnect_churn.ipynb`

**Interfaces:**
- Consumes: `resolve_data_dir`, `load_raw`, `merge_tables` (Task 2) y `build_features` (Task 4).
- Produces: variables del notebook que usan las Tasks 6–7:
  - `RANDOM_STATE`, `CUTOFF`, `df`
  - `X_train, X_valid, X_test, y_train, y_valid, y_test`
  - `cat_cols, num_cols`, `ohe_prep`, `ord_prep`
  - `X_train_lgb, X_valid_lgb, X_test_lgb`
  - `evaluate(model, X, y, threshold=0.5) -> dict`

**Cómo construir el notebook:** las celdas se crean con `nbformat` mediante un script de un solo uso en el scratchpad (no se versiona). El script crea `nbformat.v4.new_notebook()`, agrega las celdas en el orden de abajo (`new_markdown_cell` / `new_code_cell`) y guarda con `nbformat.write`. En las Tasks 6–7 se carga el notebook con `nbformat.read`, se agregan celdas al final y se guarda.

- [ ] **Step 1: Celdas de la Etapa 1 y el setup**

Markdown:
```markdown
# Interconnect: predicción de la baja de clientes (churn)

Interconnect quiere detectar con anticipación a los clientes que se darán de baja para ofrecerles promociones y planes especiales. Es un problema de **clasificación binaria**: ¿el cliente se dará de baja pronto? (`churn` = 1 / 0).

**Métrica principal:** AUC-ROC en test (meta ≥ 0.88). **Secundaria:** Recall, porque un falso negativo (un cliente que se va y el modelo no lo detecta) es ingreso perdido.

## Etapa 1: Plan de trabajo

1. **Unión y nulos.** `contract` LEFT JOIN `personal`, `internet` y `phone` por `customerID`. Un NaN después de la unión significa que el cliente no tiene ese servicio, no que falte el dato. Se rellena con `"No internet"` o `"No phone"` y no se elimina ninguna fila.
2. **Target.** `churn = 1` si `EndDate` es una fecha y `0` si es `"No"`.
3. **Preprocesamiento por variable.**
   - `BeginDate`/`EndDate` se convierten en `tenure_days`. Para los clientes activos se usa la fecha de corte, que se verifica en el EDA. Después se eliminan las fechas para evitar fugas de información.
   - `TotalCharges` es texto: se convierte a número y los vacíos se rellenan con 0.
   - Features nuevas: `has_internet`, `has_phone` y `n_services`.
   - Categóricas: One-Hot para la Regresión Logística, con las numéricas escaladas; OrdinalEncoder para Random Forest; categóricas nativas para LightGBM.
4. **Modelos.** Split estratificado 60/20/20 (train/validación/test).
   - Baseline sin balanceo: Regresión Logística, Random Forest y LightGBM.
   - Optimización: balanceo de clases (`class_weight` / `is_unbalance`) más `RandomizedSearchCV` con CV estratificada de 5 folds y `scoring="roc_auc"`.
   - Se elige el umbral en validación para alcanzar un Recall de alrededor de 0.80.
   - El modelo final se reentrena con train + validación y se evalúa **una sola vez** en test.
```

Código:
```python
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.data import load_raw, merge_tables, resolve_data_dir
from src.features import build_features

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")
RANDOM_STATE = 12345
```

- [ ] **Step 2: Celdas del EDA**

Markdown: `## Etapa 2.1: Análisis exploratorio`

Código (carga, tipos y duplicados):
```python
data_dir = resolve_data_dir()
raw = load_raw(data_dir)
print("Datos desde:", data_dir)
for name, table in raw.items():
    print(f"\n=== {name}: {table.shape[0]} filas, {table.shape[1]} columnas, "
          f"customerID duplicados: {table['customerID'].duplicated().sum()}, "
          f"filas duplicadas: {table.duplicated().sum()}")
    table.info()
    display(table.head(3))
```

Código (fecha de corte):
```python
contract = raw["contract"]
begin_max = pd.to_datetime(contract["BeginDate"]).max()
end_max = pd.to_datetime(contract.loc[contract["EndDate"] != "No", "EndDate"]).max()
print("Último BeginDate:", begin_max.date(), "| Último EndDate:", end_max.date())
print("\nDistribución de EndDate entre los clientes que se fueron:")
print(contract.loc[contract["EndDate"] != "No", "EndDate"].value_counts().sort_index())
CUTOFF = "2020-02-01"
assert begin_max == pd.Timestamp(CUTOFF), "La fecha de corte no coincide con los datos"
```

Código (anomalía de `TotalCharges`):
```python
total_num = pd.to_numeric(contract["TotalCharges"], errors="coerce")
blank = contract[total_num.isna()]
print(f"TotalCharges no numérico: {len(blank)} filas")
display(blank[["customerID", "BeginDate", "EndDate", "MonthlyCharges", "TotalCharges"]])
```

Markdown (debe coincidir con la salida; ajustar si difiere):
```markdown
Todos los `TotalCharges` en blanco son de clientes con `BeginDate` igual a la fecha de corte (2020-02-01): acaban de contratar y todavía no se les ha facturado. Por eso se rellenan con **0**, que es el monto real acumulado. Usar `MonthlyCharges` supondría un pago que aún no ocurrió.
```

Código (unión y nulos):
```python
merged = merge_tables(raw)
print("Filas tras la unión:", len(merged))
print("\nNulos por columna tras la unión (= servicio no contratado):")
print(merged.isna().sum()[lambda s: s > 0])
```

Código (features y balance de clases):
```python
df = build_features(merged, CUTOFF)
df.info()
print("\nBalance de clases:")
print(df["churn"].value_counts().to_frame("clientes").assign(proporción=df["churn"].value_counts(normalize=True).round(3)))
```

Código (distribuciones numéricas):
```python
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col in zip(axes, ["MonthlyCharges", "TotalCharges", "tenure_days"]):
    sns.histplot(data=df, x=col, hue="churn", stat="density", common_norm=False, bins=40, ax=ax)
    ax.set_title(f"{col} por churn")
plt.tight_layout()
plt.show()
```

Código (tasa de churn por categoría):
```python
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for ax, col in zip(axes, ["Type", "InternetService", "PaymentMethod"]):
    df.groupby(col)["churn"].mean().sort_values().plot.barh(ax=ax)
    ax.set_title(f"Tasa de churn por {col}")
    ax.set_xlabel("tasa de churn")
plt.tight_layout()
plt.show()
```

Código (¿aporta `TotalCharges / MonthlyCharges`?):
```python
ratio = df["TotalCharges"] / df["MonthlyCharges"]
print("Correlación entre TotalCharges/MonthlyCharges y tenure_days:",
      round(np.corrcoef(ratio, df["tenure_days"])[0, 1], 4))
```

- [ ] **Step 3: Celdas del preprocesamiento**

Markdown: `## Etapa 2.2: Preprocesamiento y división de los datos`

Código (split 60/20/20):
```python
X = df.drop(columns="churn")
y = df["churn"]
X_trval, X_test, y_trval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trval, y_trval, test_size=0.25, stratify=y_trval, random_state=RANDOM_STATE)
for name, part, target in [("train", X_train, y_train), ("valid", X_valid, y_valid), ("test", X_test, y_test)]:
    print(f"{name}: {part.shape}, churn = {target.mean():.3f}")
```

Código (encoders):
```python
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
print("Categóricas:", cat_cols)
print("Numéricas:", num_cols)

# Regresión logística: One-Hot (sin orden artificial) + escalado (coeficientes comparables y convergencia).
ohe_prep = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
# Random Forest: los árboles no necesitan escalado y OrdinalEncoder evita multiplicar columnas.
ord_prep = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
    ("num", "passthrough", num_cols),
])
# LightGBM: categóricas nativas. Las categorías se fijan solo con train.
cat_dtypes = {c: pd.CategoricalDtype(sorted(X_train[c].unique())) for c in cat_cols}
X_train_lgb, X_valid_lgb, X_test_lgb = (part.astype(cat_dtypes) for part in (X_train, X_valid, X_test))
```

Código (función de evaluación):
```python
def evaluate(model, X, y, threshold=0.5):
    """AUC-ROC (con probabilidades) y Recall/Precisión con el umbral dado."""
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "AUC-ROC": roc_auc_score(y, proba),
        "Recall": recall_score(y, pred),
        "Precisión": precision_score(y, pred),
    }
```

- [ ] **Step 4: Ejecutar el notebook completo**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 interconnect_churn.ipynb`
Expected: termina sin error. Revisar en las salidas que haya 7043 filas, que el `assert` de `CUTOFF` pase, que los 11 `TotalCharges` en blanco tengan `BeginDate` 2020-02-01 y que el churn en los tres splits sea parecido.

- [ ] **Step 5: Agregar la celda de conclusiones del EDA**

Agregar una celda markdown `### Conclusiones del EDA` después de la celda de la razón `TotalCharges/MonthlyCharges`. Debe tener 4–6 viñetas con lo que **muestran las salidas de la ejecución**: proporción de churn, efecto de `Type`, `InternetService`, `PaymentMethod` y `tenure_days`, rango de fechas de baja y decisión sobre la razón. Si la correlación es ≥ 0.95, la razón no se agrega por ser redundante con `tenure_days`. Ejecutar de nuevo con el comando del Step 4.

- [ ] **Step 6: Commit**

```bash
git add interconnect_churn.ipynb
git commit -m "feat(notebook): add work plan, EDA and preprocessing

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: CHECKPOINT.** Revisión con `superpowers:requesting-code-review`. Presentar al usuario los hallazgos del EDA (sacados de las salidas) y pedir confirmación para pasar a modelado.

---

### Task 6: Notebook — baseline y optimización

**Files:**
- Modify: `interconnect_churn.ipynb` (celdas agregadas al final)

**Interfaces:**
- Consumes: las variables de la Task 5.
- Produces:
  - `DATA: dict[str, tuple]` con `nombre -> (X_train, X_valid, X_test)` en el formato que acepta cada modelo
  - `candidates: dict[tuple[str, str], estimator]` con clave `(modelo, etapa)` y etapa en `{"baseline", "optimizado"}`
  - `results: pd.DataFrame` con columnas `modelo`, `etapa`, `AUC-ROC CV`, `AUC-ROC`, `Recall`, `Precisión`
  - `searches: dict[str, RandomizedSearchCV]`

- [ ] **Step 1: Celdas del baseline**

Markdown:
```markdown
## Etapa 2.3: Modelos baseline (sin balanceo)

Tres modelos con parámetros por defecto, entrenados en train y evaluados en validación con umbral 0.5.
```

Código:
```python
def make_logreg(**kw):
    return Pipeline([("prep", clone(ohe_prep)),
                     ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, **kw))])

def make_rf(n_jobs=-1, **kw):
    return Pipeline([("prep", clone(ord_prep)),
                     ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, **kw))])

def make_lgbm(n_jobs=-1, **kw):
    return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=n_jobs, verbose=-1, **kw)

DATA = {
    "LogisticRegression": (X_train, X_valid, X_test),
    "RandomForest": (X_train, X_valid, X_test),
    "LightGBM": (X_train_lgb, X_valid_lgb, X_test_lgb),
}
candidates, rows = {}, []
for name, model in {"LogisticRegression": make_logreg(), "RandomForest": make_rf(), "LightGBM": make_lgbm()}.items():
    X_tr, X_va, _ = DATA[name]
    model.fit(X_tr, y_train)
    candidates[(name, "baseline")] = model
    rows.append({"modelo": name, "etapa": "baseline", "AUC-ROC CV": np.nan, **evaluate(model, X_va, y_valid)})
results = pd.DataFrame(rows)
results.round(4)
```

- [ ] **Step 2: Celdas de la optimización**

Markdown:
```markdown
## Etapa 2.4: Balanceo de clases + búsqueda de hiperparámetros

El churn es la clase minoritaria (cerca de 26%), así que se aplica **ponderación de clases** (`class_weight="balanced"`; en LightGBM, `is_unbalance=True`). Así se penalizan más los errores sobre los clientes que se van, sin duplicar ni descartar filas. La búsqueda usa `RandomizedSearchCV` con CV estratificada de 5 folds sobre train y `scoring="roc_auc"`.
```

Código:
```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
search_spaces = {
    "LogisticRegression": (
        make_logreg(class_weight="balanced"),
        {"model__C": loguniform(1e-3, 1e2)},
        30,
    ),
    "RandomForest": (
        make_rf(n_jobs=1, class_weight="balanced"),
        {"model__n_estimators": randint(200, 801),
         "model__max_depth": [None, 6, 8, 10, 12, 16, 20],
         "model__min_samples_leaf": randint(1, 21),
         "model__max_features": ["sqrt", "log2", 0.5]},
        30,
    ),
    "LightGBM": (
        make_lgbm(n_jobs=1, is_unbalance=True, subsample_freq=1),
        {"n_estimators": randint(200, 1501),
         "learning_rate": loguniform(0.01, 0.2),
         "num_leaves": randint(8, 65),
         "min_child_samples": randint(5, 61),
         "subsample": uniform(0.6, 0.4),
         "colsample_bytree": uniform(0.6, 0.4),
         "reg_lambda": loguniform(1e-3, 10)},
        50,
    ),
}
searches = {}
for name, (estimator, params, n_iter) in search_spaces.items():
    X_tr, X_va, _ = DATA[name]
    search = RandomizedSearchCV(estimator, params, n_iter=n_iter, scoring="roc_auc",
                                cv=cv, random_state=RANDOM_STATE, n_jobs=-1)
    search.fit(X_tr, y_train)
    searches[name] = search
    candidates[(name, "optimizado")] = search.best_estimator_
    rows.append({"modelo": name, "etapa": "optimizado", "AUC-ROC CV": search.best_score_,
                 **evaluate(search.best_estimator_, X_va, y_valid)})
    print(f"{name}: mejores parámetros {search.best_params_}")
results = pd.DataFrame(rows)
```

Código (tabla comparativa):
```python
results.pivot_table(index="modelo", columns="etapa",
                    values=["AUC-ROC CV", "AUC-ROC", "Recall", "Precisión"]).round(4)
```

- [ ] **Step 3: Ejecutar el notebook completo**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 interconnect_churn.ipynb`
Expected: termina sin error y la tabla tiene 6 filas (3 modelos × 2 etapas).
**Revisión de anomalías:** si algún AUC de validación es ≥ 0.98, o la diferencia entre `AUC-ROC CV` y `AUC-ROC` de un mismo modelo es mayor que 0.03, **parar** y aplicar `superpowers:systematic-debugging` (sospecha de fuga de datos).

- [ ] **Step 4: Agregar la celda de análisis de la comparación**

Agregar una celda markdown `### Comparación baseline vs. optimizado` con 3–5 viñetas sacadas de la tabla ejecutada: qué modelo gana en AUC, cuánto cambia el Recall con el balanceo y si la CV coincide con validación. Ejecutar de nuevo.

- [ ] **Step 5: Commit**

```bash
git add interconnect_churn.ipynb
git commit -m "feat(notebook): add baseline models and balanced hyperparameter search

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: CHECKPOINT.** Revisión con `superpowers:requesting-code-review`. Presentar la tabla comparativa (valores ejecutados) y pedir confirmación para la prueba final en test.

---

### Task 7: Notebook — umbral, prueba final e informe para negocio

**Files:**
- Modify: `interconnect_churn.ipynb` (celdas agregadas al final)

**Interfaces:**
- Consumes: `DATA`, `candidates`, `results`, `evaluate`, los splits y `y_*` de las Tasks 5–6.
- Produces: `best_key`, `threshold`, `final_model`, `test_metrics: dict` y la Etapa 3 en markdown.

- [ ] **Step 1: Celdas de selección de modelo y umbral**

Markdown:
```markdown
## Selección del modelo y del umbral

Se elige el candidato con mayor AUC-ROC en validación. Después, con su curva de precisión-recall en validación, se busca el umbral más alto que todavía da un Recall ≥ 0.80. Así se detecta a la mayoría de los clientes que se van y se contacta al menor número posible de clientes que se quedan.
```

Código:
```python
best_row = results.loc[results["AUC-ROC"].idxmax()]
best_key = (best_row["modelo"], best_row["etapa"])
best_model = candidates[best_key]
X_tr, X_va, X_te = DATA[best_key[0]]
print("Modelo elegido:", best_key)

TARGET_RECALL = 0.80
proba_valid = best_model.predict_proba(X_va)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_valid, proba_valid)
threshold = thresholds[recall[:-1] >= TARGET_RECALL].max()
print(f"Umbral elegido: {threshold:.3f}")
print("Validación con ese umbral:", {k: round(v, 4) for k, v in evaluate(best_model, X_va, y_valid, threshold).items()})

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(recall, precision)
ax.axvline(TARGET_RECALL, ls="--", c="gray")
ax.set_xlabel("Recall")
ax.set_ylabel("Precisión")
ax.set_title(f"Curva precisión-recall en validación ({best_key[0]})")
plt.show()
```

- [ ] **Step 2: Celdas de la prueba final (test, una sola vez)**

Markdown:
```markdown
## Prueba final en test

El modelo elegido se reentrena con train + validación, con los mismos hiperparámetros y el mismo umbral, y se evalúa **una única vez** en el conjunto de prueba.
```

Código:
```python
final_model = clone(best_model)
final_model.fit(pd.concat([X_tr, X_va]), pd.concat([y_train, y_valid]))
test_metrics = evaluate(final_model, X_te, y_test, threshold)
print("Resultados en TEST:", {k: round(v, 4) for k, v in test_metrics.items()})
print(f"Umbral: {threshold:.3f}")

pred_test = (final_model.predict_proba(X_te)[:, 1] >= threshold).astype(int)
ConfusionMatrixDisplay.from_predictions(y_test, pred_test, display_labels=["Se queda", "Se va"])
plt.title("Matriz de confusión en test")
plt.show()
```

Código (importancia de features):
```python
if isinstance(final_model, Pipeline):
    estimator = final_model.named_steps["model"]
    names = final_model.named_steps["prep"].get_feature_names_out()
else:
    estimator, names = final_model, X_te.columns
values = (estimator.feature_importances_ if hasattr(estimator, "feature_importances_")
          else np.abs(estimator.coef_[0]))
pd.Series(values, index=names).sort_values().tail(15).plot.barh(figsize=(7, 5))
plt.title(f"Importancia de features: {best_key[0]}")
plt.show()
```

- [ ] **Step 3: Ejecutar el notebook completo**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 interconnect_churn.ipynb`
Expected: termina sin error. Anotar en la salida los valores exactos de AUC-ROC, Recall, Precisión y umbral en test, y los cuatro valores de la matriz de confusión. Si AUC-ROC en test < 0.88, **parar** y avisar al usuario antes de seguir (no tocar el test de nuevo para ajustar).

- [ ] **Step 4: Escribir la Etapa 3 con los números de la salida**

Agregar al final una celda markdown con esta estructura, reemplazando cada `‹…›` por el valor exacto impreso en el Step 3 (la matriz de confusión da VP, FN, FP y VN):

```markdown
## Etapa 3: Informe para negocio

**Modelo elegido:** ‹modelo› (‹etapa›). ‹1–2 frases de por qué: mejor AUC-ROC en validación, comparación con los demás según la tabla›.

**Resultados en test** (datos nunca usados para entrenar ni ajustar):

| Métrica | Valor |
|---|---|
| AUC-ROC | ‹auc› |
| Recall | ‹recall› |
| Precisión | ‹precisión› |
| Umbral | ‹umbral› |

**¿Qué significa?**
- De cada 100 clientes que realmente se van, el modelo detecta a ‹round(recall*100)›.
- De cada 100 clientes que el modelo marca como "en riesgo", ‹round(precisión*100)› realmente se iban a ir. El resto recibiría una promoción sin necesitarla: es el costo de no perder a los demás.
- En el conjunto de test (‹n_test› clientes): ‹VP› bajas detectadas a tiempo, ‹FN› bajas no detectadas y ‹FP› clientes contactados sin necesidad.

**Recomendación de uso:** ejecutar el modelo cada mes sobre los clientes activos y enviar a marketing la lista de clientes por encima del umbral, ordenada por probabilidad. Si la campaña tiene presupuesto limitado, subir el umbral: se contacta a menos gente con más precisión, a cambio de menos Recall. ‹1 frase sobre los factores principales según la importancia de features›.
```

Después del reemplazo, `grep -c "‹" interconnect_churn.ipynb` debe dar `0`.

- [ ] **Step 5: Ejecución final limpia y verificación**

Run: `.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 interconnect_churn.ipynb && .venv/Scripts/python -m pytest -q`
Expected: el notebook se ejecuta sin error, los tests pasan y los números de la Etapa 3 coinciden con las salidas impresas de esta última ejecución (las semillas fijas hacen la ejecución determinista; si algo difiere, corregir el texto).

- [ ] **Step 6: Commit**

```bash
git add interconnect_churn.ipynb
git commit -m "feat(notebook): add threshold selection, final test evaluation and business report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 7: CHECKPOINT.** Revisión con `superpowers:requesting-code-review` y `superpowers:verification-before-completion`. Presentar al usuario las métricas de test y pedir confirmación para subir a GitHub.

---

### Task 8: Publicar en GitHub

**Files:** ninguno.

- [ ] **Step 1: Confirmar con el usuario** el nombre del repo (propuesto: `interconnect-churn`) y que será **público**.

- [ ] **Step 2: Verificar la autenticación de `gh`**

Run: `gh auth status`
Expected: sesión activa. Si no la hay, pedir al usuario que ejecute `! gh auth login`.

- [ ] **Step 3: Crear el repo y hacer push**

```bash
gh repo create interconnect-churn --public --source . --remote origin --push \
  --description "Predicción de churn para Interconnect (proyecto final Ciencia de Datos)"
```

- [ ] **Step 4: Verificar**

Run: `git status -sb && gh repo view --json visibility,url`
Expected: `master...origin/master` sin cambios pendientes; el repo aparece como público.
