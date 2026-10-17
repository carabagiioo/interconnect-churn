# Interconnect — Predicción de churn

Proyecto final del curso de Ciencia de Datos. Modelo de clasificación binaria que predice qué clientes de la telecom Interconnect se darán de baja, para que marketing les ofrezca promociones a tiempo.

**Métrica principal:** AUC-ROC en test (meta ≥ 0.88). **Secundaria:** Recall.

## Estructura

- `final_provider/` — datos crudos (`contract`, `personal`, `internet`, `phone`).
- `src/` — carga, unión e ingeniería de features (probado en `tests/`).
- `interconnect_churn.ipynb` — EDA, modelado, evaluación e informe para negocio.
- `interconnect_churn_plataforma.ipynb` — la misma versión en un solo archivo (sin depender de `src/`) para la plataforma.
- `docs/` — diseño y plan de implementación.

## Cómo ejecutar

```bash
py -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pytest
.venv/Scripts/jupyter notebook interconnect_churn.ipynb
```

### En la plataforma del curso

Usar `interconnect_churn_plataforma.ipynb`: es el mismo notebook con las funciones de `src/` incluidas en una celda, así que funciona como un único archivo. En JupyterLab: **File → Open from URL** con

```
https://raw.githubusercontent.com/carabagiioo/interconnect-churn/master/interconnect_churn_plataforma.ipynb
```

Si existe `/datasets/final_provider/`, el notebook usa esa ruta automáticamente; si no, busca la carpeta `final_provider/` junto al notebook.
