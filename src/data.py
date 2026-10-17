"""Carga y unión de las tablas de Interconnect."""
from pathlib import Path

import pandas as pd

PLATFORM_DIR = Path("/datasets/final_provider")
LOCAL_DIR = Path(__file__).resolve().parent.parent / "final_provider"
TABLES = ("contract", "personal", "internet", "phone")


def resolve_data_dir(platform_dir: Path = PLATFORM_DIR, local_dir: Path = LOCAL_DIR) -> Path:
    """Usa la ruta de la plataforma si existe; si no, la carpeta local del proyecto."""
    return Path(platform_dir) if Path(platform_dir).exists() else Path(local_dir)


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
