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
