from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

FuelType = Literal["petrol", "diesel", "hybrid", "electric"]


class MotDefect(BaseModel):
    text: str
    kind: Literal["advisory", "minor", "major", "dangerous"] = "advisory"


class MotTest(BaseModel):
    completed: date
    mileage: int = Field(ge=0)
    passed: bool
    defects: list[MotDefect] = Field(default_factory=list)

    @property
    def advisories(self) -> list[MotDefect]:
        return [d for d in self.defects if d.kind in ("advisory", "minor")]

    @property
    def failures(self) -> list[MotDefect]:
        return [d for d in self.defects if d.kind in ("major", "dangerous")]


class Vehicle(BaseModel):
    registration: str
    make: str
    model: str
    year: int
    fuel: FuelType
    colour: str | None = None
    audio_system: str | None = None
    last_service: date | None = None
    mot_tests: list[MotTest] = Field(default_factory=list)

    @property
    def latest_mot(self) -> MotTest | None:
        return max(self.mot_tests, key=lambda t: t.completed, default=None)

    @property
    def current_mileage(self) -> int:
        latest = self.latest_mot
        return latest.mileage if latest else 0

    def mot_due(self) -> date | None:
        """Next MOT due date: one year after the latest test.

        Cars with no test yet are due on the third anniversary of first registration
        (simplified to 1 January of year + 3 because the sample data stores the year only).
        """
        latest = self.latest_mot
        if latest:
            return _add_years(latest.completed, 1)
        return date(self.year + 3, 1, 1)

    def summary(self) -> str:
        due = self.mot_due()
        text = (
            f"{self.registration}: {self.year} {self.make} {self.model} ({self.fuel}), "
            f"{self.current_mileage:,} miles"
        )
        return f"{text}, MOT due {due:%d %b %Y}" if due else text


def _add_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # 29 February
        return d.replace(year=d.year + years, day=28)
