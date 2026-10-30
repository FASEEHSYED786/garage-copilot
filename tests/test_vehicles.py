import httpx
import pytest

from garage_copilot.config import Settings
from garage_copilot.vehicles import (
    DvsaMotHistoryClient,
    MockVehicleLookup,
    VehicleNotFoundError,
    find_registration,
    normalise_registration,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ab19cde", "AB19 CDE"),
        (" AB19 CDE ", "AB19 CDE"),
        ("ab-19-cde", "AB19 CDE"),
        ("A123BCD", "A123BCD"),
    ],
)
def test_normalise_registration(raw, expected):
    assert normalise_registration(raw) == expected


def test_normalise_rejects_empty():
    with pytest.raises(ValueError):
        normalise_registration(" - ")


def test_find_registration_in_text():
    assert find_registration("Can you book ab19cde in on Friday?") == "AB19 CDE"
    assert find_registration("no plate here") is None


def test_mock_lookup_reads_sample_data():
    lookup = MockVehicleLookup.from_file(Settings().vehicles_file)
    golf = lookup.get("ab19 cde")
    assert golf.make == "Volkswagen"
    assert golf.mot_tests, "sample vehicles carry MOT history"
    assert golf.current_mileage == golf.latest_mot.mileage
    assert golf.mot_due() > golf.latest_mot.completed


def test_mock_lookup_unknown_vehicle():
    lookup = MockVehicleLookup.from_file(Settings().vehicles_file)
    with pytest.raises(VehicleNotFoundError):
        lookup.get("ZZ99 ZZZ")


def test_dvsa_client_maps_response():
    payload = {
        "registration": "AB12CDE",
        "make": "FORD",
        "model": "FIESTA",
        "fuelType": "Petrol",
        "primaryColour": "Blue",
        "firstUsedDate": "2012-03-01",
        "motTests": [
            {
                "completedDate": "2025-03-02T10:00:00.000Z",
                "testResult": "FAILED",
                "odometerValue": "84000",
                "defects": [{"text": "Brake disc worn", "type": "MAJOR"}],
            },
            {
                "completedDate": "2025-03-05T10:00:00.000Z",
                "testResult": "PASSED",
                "odometerValue": "84010",
                "defects": [{"text": "Tyre worn close to limit", "type": "ADVISORY"}],
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/AB12CDE")
        assert request.headers["x-api-key"] == "key"
        return httpx.Response(200, json=payload)

    client = DvsaMotHistoryClient(
        "key", "token", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    v = client.get("ab12 cde")
    assert (v.make, v.model, v.year, v.fuel) == ("Ford", "Fiesta", 2012, "petrol")
    assert v.latest_mot.passed and v.current_mileage == 84010
    assert v.mot_tests[0].failures[0].text == "Brake disc worn"


def test_dvsa_client_404():
    client = DvsaMotHistoryClient(
        "key",
        "token",
        client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(404))),
    )
    with pytest.raises(VehicleNotFoundError):
        client.get("AB12 CDE")
