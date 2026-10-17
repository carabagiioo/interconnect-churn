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
