"""Synthetic MOT outcome data and feature extraction.

The generator encodes simple, plausible relationships (older, higher-mileage cars with
recent advisories fail more often). It exists so the pipeline can be trained and tested
end to end without real customer data. Swap `generate_dataset` for the DVSA anonymised
MOT dataset to train on real outcomes.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from garage_copilot.vehicles.models import Vehicle

NUMERIC_FEATURES = [
    "age_years",
    "mileage",
    "miles_per_year",
    "previous_failures",
    "advisories_last_test",
    "months_since_service",
]
CATEGORICAL_FEATURES = ["fuel"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "failed"

_FUEL_EFFECT = {"petrol": 0.0, "diesel": 0.25, "hybrid": -0.2, "electric": -0.45}


def generate_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fuel = rng.choice(list(_FUEL_EFFECT), size=n, p=[0.45, 0.33, 0.14, 0.08])
    age = rng.integers(3, 20, size=n)
    miles_per_year = rng.normal(9000, 3500, size=n).clip(1500, 30000)
    mileage = (age * miles_per_year * rng.uniform(0.85, 1.15, size=n)).astype(int)
    previous_failures = rng.poisson(0.15 * age)
    advisories = rng.poisson(0.4 + 0.12 * age)
    months_since_service = rng.integers(0, 30, size=n)

    logit = (
        -5.2
        + 0.13 * age
        + 0.000006 * mileage
        + 0.35 * previous_failures
        + 0.45 * advisories
        + 0.04 * months_since_service
        + np.vectorize(_FUEL_EFFECT.get)(fuel)
        + rng.normal(0, 0.35, size=n)
    )
    failed = rng.random(n) < 1 / (1 + np.exp(-logit))

    return pd.DataFrame(
        {
            "age_years": age,
            "mileage": mileage,
            "miles_per_year": miles_per_year.round().astype(int),
            "previous_failures": previous_failures,
            "advisories_last_test": advisories,
            "months_since_service": months_since_service,
            "fuel": fuel,
            TARGET: failed.astype(int),
        }
    )


def vehicle_features(vehicle: Vehicle, today: date | None = None) -> pd.DataFrame:
    """Turn a looked-up vehicle into the single-row frame the model expects."""
    today = today or date.today()
    age = max(today.year - vehicle.year, 0)
    latest = vehicle.latest_mot
    mileage = vehicle.current_mileage
    if vehicle.last_service:
        months = (
            (today.year - vehicle.last_service.year) * 12 + today.month - vehicle.last_service.month
        )
    else:
        months = 12
    row = {
        "age_years": age,
        "mileage": mileage,
        "miles_per_year": int(mileage / age) if age else mileage,
        "previous_failures": sum(1 for t in vehicle.mot_tests if not t.passed),
        "advisories_last_test": len(latest.advisories) if latest else 0,
        "months_since_service": max(months, 0),
        "fuel": vehicle.fuel,
    }
    return pd.DataFrame([row], columns=FEATURES)
