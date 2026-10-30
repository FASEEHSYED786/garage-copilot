import pytest
from fastapi.testclient import TestClient

from garage_copilot.api import create_app


@pytest.fixture
def client(copilot) -> TestClient:
    return TestClient(create_app(copilot))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm"] == "offline"
    assert body["knowledge_chunks"] > 10


def test_vehicle_lookup(client):
    resp = client.get("/vehicles/ab19cde")
    assert resp.status_code == 200
    assert resp.json()["make"] == "Volkswagen"
    assert client.get("/vehicles/ZZ99ZZZ").status_code == 404


def test_mot_risk(client):
    body = client.get("/vehicles/RS14ABC/mot-risk").json()
    assert body["band"] in {"low", "medium", "high"}
    assert 0 <= body["probability"] <= 1


def test_ask(client):
    body = client.post("/ask", json={"question": "What is the courtesy car excess?"}).json()
    assert "£250" in body["answer"]
    assert body["citations"]


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_agent_endpoint(client):
    body = client.post("/agent", json={"message": "Book AB19 CDE for an MOT next week"}).json()
    assert "Booked mot" in body["reply"]
    assert [s["tool"] for s in body["steps"]] == ["lookup_vehicle", "mot_risk", "book_job"]


def test_fleet_availability(client):
    body = client.get("/fleet/availability", params={"day": "2026-11-02"}).json()
    assert len(body["available"]) == 3


def test_bad_registration_is_400(client):
    assert client.get("/vehicles/" + "A" * 20).status_code == 400


def test_api_key_required_when_set(copilot):
    from dataclasses import replace

    copilot.settings = replace(copilot.settings, api_key="s3cret")
    client = TestClient(create_app(copilot))
    assert client.get("/health").status_code == 401
    assert client.get("/health", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health", headers={"X-API-Key": "s3cret"}).status_code == 200
