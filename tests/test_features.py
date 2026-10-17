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
