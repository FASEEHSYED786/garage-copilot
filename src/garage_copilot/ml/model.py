"""MOT failure risk model: a scikit-learn pipeline with persistence and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from garage_copilot.ml.dataset import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    generate_dataset,
    vehicle_features,
)
from garage_copilot.vehicles.models import Vehicle


@dataclass
class RiskScore:
    probability: float
    band: str
    reasons: list[str] = field(default_factory=list)

    @property
    def recommend_pre_mot_check(self) -> bool:
        return self.band == "high"


class MotRiskModel:
    HIGH = 0.45
    MEDIUM = 0.25

    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or self._build_pipeline()
        self.metrics: dict[str, float] = {}

    @staticmethod
    def _build_pipeline() -> Pipeline:
        preprocess = ColumnTransformer(
            [
                ("num", "passthrough", NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ]
        )
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0)
        return Pipeline([("prep", preprocess), ("clf", clf)])

    def fit(self, df: pd.DataFrame, test_size: float = 0.2, seed: int = 0) -> dict[str, float]:
        x_train, x_test, y_train, y_test = train_test_split(
            df[FEATURES], df[TARGET], test_size=test_size, random_state=seed, stratify=df[TARGET]
        )
        self.pipeline.fit(x_train, y_train)
        proba = self.pipeline.predict_proba(x_test)[:, 1]
        self.metrics = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "fail_rate": float(df[TARGET].mean()),
            "n_train": float(len(x_train)),
        }
        return self.metrics

    @classmethod
    def train_default(cls, n: int = 6000, seed: int = 42) -> MotRiskModel:
        model = cls()
        model.fit(generate_dataset(n=n, seed=seed))
        return model

    def score(self, vehicle: Vehicle, today: date | None = None) -> RiskScore:
        features = vehicle_features(vehicle, today)
        probability = float(self.pipeline.predict_proba(features)[0, 1])
        if probability >= self.HIGH:
            band = "high"
        elif probability >= self.MEDIUM:
            band = "medium"
        else:
            band = "low"
        return RiskScore(round(probability, 3), band, _reasons(features.iloc[0]))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "metrics": self.metrics}, path)

    @classmethod
    def load(cls, path: Path) -> MotRiskModel:
        # joblib uses pickle: only load model files you created yourself.
        payload = joblib.load(path)
        model = cls(payload["pipeline"])
        model.metrics = payload.get("metrics", {})
        return model

    @classmethod
    def load_or_train(cls, path: Path) -> MotRiskModel:
        if Path(path).exists():
            return cls.load(path)
        model = cls.train_default()
        model.save(path)
        return model


def _reasons(row: pd.Series) -> list[str]:
    """Plain-English drivers a service adviser can repeat to a customer."""
    reasons = []
    if row["advisories_last_test"] >= 2:
        reasons.append(f"{int(row['advisories_last_test'])} advisories on the last MOT")
    if row["previous_failures"] >= 1:
        reasons.append(f"{int(row['previous_failures'])} previous MOT failure(s)")
    if row["age_years"] >= 10:
        reasons.append(f"{int(row['age_years'])} years old")
    if row["mileage"] >= 100_000:
        reasons.append(f"high mileage ({int(row['mileage']):,} miles)")
    if row["months_since_service"] >= 15:
        reasons.append(f"{int(row['months_since_service'])} months since last service")
    return reasons
