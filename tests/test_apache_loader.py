"""Unit tests for the APACHE loader using a tiny synthetic CSV."""
import pandas as pd
import pytest

from agents.severity import apache_loader as loader


@pytest.fixture
def apache_csv(tmp_path, monkeypatch):
    path = tmp_path / "apachePatientResult.csv"
    pd.DataFrame([
        {"patientunitstayid": 141765, "apacheversion": "IV", "apachescore": 47,
         "predictedicumortality": 0.008},
        {"patientunitstayid": 141765, "apacheversion": "IVa", "apachescore": 47,
         "predictedicumortality": 0.012},
        {"patientunitstayid": 156906, "apacheversion": "IV", "apachescore": 18,
         "predictedicumortality": 0.006},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(loader, "APACHE_PATH", path)
    return path


def test_loader_returns_both_versions(apache_csv):
    result = loader.load_apache(141765)
    assert set(result) == {"IV", "IVa"}
    assert result["IV"]["mortality_probability"] == pytest.approx(0.008)
    assert result["IVa"]["score"] == 47


def test_loader_handles_missing_episode(apache_csv):
    assert loader.load_apache(999999) == {}


def test_loader_does_not_expose_actual_outcome(apache_csv):
    result = loader.load_apache(141765)
    assert all(set(version) == {"version", "score", "mortality_probability"}
               for version in result.values())



def _write_apache(tmp_path, monkeypatch, rows):
    path = tmp_path / "apachePatientResult.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    monkeypatch.setattr(loader, "APACHE_PATH", path)
    return path


def test_loader_ignores_invalid_probability_rows(tmp_path, monkeypatch):
    _write_apache(tmp_path, monkeypatch, [
        {"patientunitstayid": 141765, "apacheversion": "IV", "apachescore": 47,
         "predictedicumortality": -1.0},
        {"patientunitstayid": 141765, "apacheversion": "IVa", "apachescore": 47,
         "predictedicumortality": 0.012},
    ])
    result = loader.load_apache(141765)
    assert set(result) == {"IVa"}


def test_loader_ignores_unknown_apache_version(tmp_path, monkeypatch):
    _write_apache(tmp_path, monkeypatch, [
        {"patientunitstayid": 141765, "apacheversion": "III", "apachescore": 47,
         "predictedicumortality": 0.012},
    ])
    assert loader.load_apache(141765) == {}


def test_loader_rejects_duplicate_version(tmp_path, monkeypatch):
    _write_apache(tmp_path, monkeypatch, [
        {"patientunitstayid": 141765, "apacheversion": "IV", "apachescore": 47,
         "predictedicumortality": 0.012},
        {"patientunitstayid": 141765, "apacheversion": "IV", "apachescore": 47,
         "predictedicumortality": 0.013},
    ])
    with pytest.raises(ValueError, match="Duplicate APACHE version"):
        loader.load_apache(141765)


def test_loader_missing_csv_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "APACHE_PATH", tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_apache(141765)
