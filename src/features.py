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
