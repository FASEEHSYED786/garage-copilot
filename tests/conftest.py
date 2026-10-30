from datetime import date
from pathlib import Path

import pytest

from garage_copilot.config import Settings
from garage_copilot.core import Copilot
from garage_copilot.ml.model import MotRiskModel

TODAY = date(2026, 10, 28)  # a Wednesday, so date logic is deterministic


@pytest.fixture(scope="session")
def risk_model() -> MotRiskModel:
    return MotRiskModel.train_default(n=3000, seed=1)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(model_path=tmp_path / "model.joblib")


@pytest.fixture
def copilot(settings: Settings, risk_model: MotRiskModel) -> Copilot:
    return Copilot.build(settings, today=lambda: TODAY, risk_model=risk_model)
