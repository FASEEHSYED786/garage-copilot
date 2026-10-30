"""Vehicle lookup by registration.

`VehicleLookup` is the interface the rest of the app depends on. Two adapters ship:

* `MockVehicleLookup` reads sample vehicles from JSON (default, no keys needed).
* `DvsaMotHistoryClient` calls the DVSA MOT History API. Field mapping follows the
  public API's documented shape; check it against the current DVSA docs before
  relying on it in production.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import httpx

from garage_copilot.vehicles.models import MotDefect, MotTest, Vehicle

_REG_CLEAN = re.compile(r"[^A-Z0-9]")
# Current-style UK plates (e.g. AB19 CDE). Older formats are accepted by `normalise`
# but only current-style ones are recognised inside free text.
UK_REG_PATTERN = re.compile(r"\b([A-Z]{2}[0-9]{2}\s?[A-Z]{3})\b", re.IGNORECASE)


class VehicleNotFoundError(LookupError):
    pass


def normalise_registration(raw: str) -> str:
    """Upper-case, strip spaces and punctuation, then format current-style plates as 'AB19 CDE'."""
    reg = _REG_CLEAN.sub("", raw.upper())
    if not 2 <= len(reg) <= 8:
        raise ValueError("Not a valid UK registration")
    if len(reg) == 7 and reg[:2].isalpha() and reg[2:4].isdigit() and reg[4:].isalpha():
        return f"{reg[:4]} {reg[4:]}"
    return reg


def find_registration(text: str) -> str | None:
    match = UK_REG_PATTERN.search(text)
    return normalise_registration(match.group(1)) if match else None


class VehicleLookup(Protocol):
    def get(self, registration: str) -> Vehicle: ...


class MockVehicleLookup:
    def __init__(self, vehicles: list[Vehicle]):
        self._by_reg = {normalise_registration(v.registration): v for v in vehicles}

    @classmethod
    def from_file(cls, path: Path) -> MockVehicleLookup:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Vehicle.model_validate(item) for item in raw])

    def get(self, registration: str) -> Vehicle:
        reg = normalise_registration(registration)
        try:
            return self._by_reg[reg]
        except KeyError:
            raise VehicleNotFoundError(f"No vehicle found for {reg}") from None

    def all(self) -> list[Vehicle]:
        return list(self._by_reg.values())


class DvsaMotHistoryClient:
    """Thin adapter over the DVSA MOT History API (trade endpoint)."""

    BASE_URL = "https://history.mot.api.gov.uk/v1/trade/vehicles/registration/"

    def __init__(self, api_key: str, access_token: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=10.0)
        self._headers = {
            "x-api-key": api_key,
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def get(self, registration: str) -> Vehicle:
        reg = normalise_registration(registration)
        resp = self._client.get(self.BASE_URL + reg.replace(" ", ""), headers=self._headers)
        if resp.status_code == 404:
            raise VehicleNotFoundError(f"No vehicle found for {reg}")
        resp.raise_for_status()
        return self._to_vehicle(resp.json(), reg)

    @staticmethod
    def _to_vehicle(data: dict[str, Any], reg: str) -> Vehicle:
        tests = [
            MotTest(
                completed=_parse_date(t["completedDate"]),
                mileage=int(t.get("odometerValue") or 0),
                passed=str(t.get("testResult", "")).upper() == "PASSED",
                defects=[
                    MotDefect(text=d.get("text", ""), kind=_defect_kind(d.get("type", "")))
                    for d in t.get("defects", [])
                ],
            )
            for t in data.get("motTests", [])
        ]
        first_used = data.get("firstUsedDate") or data.get("registrationDate")
        year = _parse_date(first_used).year if first_used else date.today().year
        return Vehicle(
            registration=reg,
            make=str(data.get("make", "UNKNOWN")).title(),
            model=str(data.get("model", "UNKNOWN")).title(),
            year=year,
            fuel=_fuel(data.get("fuelType", "")),
            colour=data.get("primaryColour"),
            mot_tests=tests,
        )


def _parse_date(value: str) -> date:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d", "%Y.%m.%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date: {value!r}")


def _defect_kind(raw: str) -> str:
    raw = raw.lower()
    if raw in ("dangerous", "major", "minor"):
        return raw
    if raw in ("fail", "prs"):
        return "major"
    return "advisory"


def _fuel(raw: str) -> str:
    raw = raw.lower()
    if "electric" in raw and "hybrid" not in raw:
        return "electric"
    if "hybrid" in raw:
        return "hybrid"
    return "diesel" if "diesel" in raw else "petrol"
