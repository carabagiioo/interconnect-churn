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
