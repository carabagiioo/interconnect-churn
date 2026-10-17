import numpy as np
import pandas as pd
import pytest

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
    add_engineered(fill_missing_services(df))
    pd.testing.assert_frame_equal(df, before)


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


def test_add_target_rejects_missing_enddate():
    df = merged_sample()
    df.loc[0, "EndDate"] = np.nan
    with pytest.raises(ValueError):
        add_target(df)


def test_add_tenure_rejects_begin_after_cutoff():
    df = merged_sample()
    df.loc[2, "BeginDate"] = "2020-03-01"
    with pytest.raises(ValueError):
        add_tenure(df, CUTOFF)
