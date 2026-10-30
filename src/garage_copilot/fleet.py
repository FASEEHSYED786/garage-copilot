"""Workshop bookings and courtesy car allocation (in-memory)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

JobType = Literal["mot", "service", "repair", "audio_fit"]

# Jobs longer than four hours qualify for a courtesy car (see booking policy).
JOB_HOURS: dict[str, float] = {"mot": 1.0, "service": 5.0, "repair": 6.0, "audio_fit": 2.5}


@dataclass(frozen=True)
class CourtesyCar:
    registration: str
    model: str


@dataclass
class Booking:
    id: int
    registration: str
    job: JobType
    day: date
    courtesy_car: CourtesyCar | None = None
    notes: list[str] = field(default_factory=list)


class FleetService:
    def __init__(self, cars: list[CourtesyCar], max_jobs_per_day: int = 6):
        self.cars = cars
        self.max_jobs_per_day = max_jobs_per_day
        self.bookings: list[Booking] = []
        self._ids = itertools.count(1)

    @classmethod
    def with_sample_fleet(cls) -> FleetService:
        return cls(
            [
                CourtesyCar("CC23 AAA", "Ford Fiesta"),
                CourtesyCar("CC23 BBB", "Toyota Yaris Hybrid"),
                CourtesyCar("CC24 CCC", "Kia Ceed"),
            ]
        )

    def available_cars(self, day: date) -> list[CourtesyCar]:
        taken = {b.courtesy_car for b in self.bookings if b.day == day and b.courtesy_car}
        return [c for c in self.cars if c not in taken]

    def is_day_full(self, day: date) -> bool:
        return sum(1 for b in self.bookings if b.day == day) >= self.max_jobs_per_day

    def next_open_day(self, start: date) -> date:
        day = start
        for _ in range(60):
            if day.weekday() < 5 and not self.is_day_full(day):
                return day
            day += timedelta(days=1)
        raise RuntimeError("No open days in the next 60 days")

    def book(
        self, registration: str, job: JobType, day: date, want_courtesy_car: bool = False
    ) -> Booking:
        if day.weekday() >= 5:
            raise ValueError("The workshop only takes bookings Monday to Friday")
        if self.is_day_full(day):
            raise ValueError(f"{day:%a %d %b} is fully booked")
        booking = Booking(next(self._ids), registration, job, day)
        if want_courtesy_car:
            if JOB_HOURS.get(job, 0) <= 4:
                booking.notes.append("Courtesy cars are only provided for jobs over four hours.")
            elif cars := self.available_cars(day):
                booking.courtesy_car = cars[0]
            else:
                booking.notes.append("No courtesy car free that day.")
        self.bookings.append(booking)
        return booking
