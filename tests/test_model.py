from datetime import date

from garage_copilot.ml import FEATURES, generate_dataset, vehicle_features
from garage_copilot.ml.model import MotRiskModel
from garage_copilot.vehicles.models import MotDefect, MotTest, Vehicle


def _vehicle(year: int, mileage: int, advisories: int, failures: int) -> Vehicle:
    tests = [
        MotTest(completed=date(2025, 1, 1), mileage=mileage - 100, passed=False)
        for _ in range(failures)
    ]
    tests.append(
        MotTest(
            completed=date(2026, 1, 1),
            mileage=mileage,
            passed=True,
            defects=[MotDefect(text=f"advisory {i}") for i in range(advisories)],
        )
    )
    return Vehicle(
        registration="AB12 CDE",
        make="Ford",
        model="Focus",
        year=year,
        fuel="diesel",
        last_service=date(2025, 6, 1),
        mot_tests=tests,
    )


def test_dataset_shape_and_realistic_fail_rate():
    df = generate_dataset(n=2000, seed=3)
    assert list(df.columns[:-1]) == FEATURES
    assert 0.2 < df["failed"].mean() < 0.45  # UK cars fail roughly a third of first MOT attempts


def test_model_beats_chance(risk_model):
    assert risk_model.metrics["roc_auc"] > 0.75


def test_risk_ranks_old_worn_cars_higher(risk_model):
    today = date(2026, 10, 28)
    new = risk_model.score(_vehicle(2022, 20_000, 0, 0), today)
    old = risk_model.score(_vehicle(2011, 160_000, 3, 3), today)
    assert old.probability > new.probability
    assert old.band == "high" and old.recommend_pre_mot_check
    assert any("advisories" in r for r in old.reasons)


def test_vehicle_features_single_row():
    row = vehicle_features(_vehicle(2016, 90_000, 2, 1), today=date(2026, 10, 28))
    assert row.shape == (1, len(FEATURES))
    assert row.loc[0, "advisories_last_test"] == 2
    assert row.loc[0, "months_since_service"] == 16


def test_save_and_load_roundtrip(tmp_path, risk_model):
    path = tmp_path / "m.joblib"
    risk_model.save(path)
    loaded = MotRiskModel.load(path)
    v = _vehicle(2015, 100_000, 1, 1)
    assert loaded.score(v).probability == risk_model.score(v).probability
    assert loaded.metrics == risk_model.metrics
