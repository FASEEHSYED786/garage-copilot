import json

import pytest

from garage_copilot.cli import main


@pytest.fixture(autouse=True)
def _model_path(tmp_path, monkeypatch):
    monkeypatch.setenv("GARAGE_MODEL_PATH", str(tmp_path / "model.joblib"))


def test_cli_ask(capsys):
    assert main(["ask", "What is the courtesy car excess?"]) == 0
    out = capsys.readouterr().out
    assert "£250" in out and "Courtesy car terms" in out


def test_cli_eval(capsys):
    assert main(["eval"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["hit@3"] >= 0.9


def test_cli_vehicle_and_risk(capsys):
    assert main(["vehicle", "ab19cde"]) == 0
    assert '"make": "Volkswagen"' in capsys.readouterr().out
    assert main(["risk", "RS14ABC"]) == 0
    assert '"band"' in capsys.readouterr().out


def test_cli_agent(capsys):
    assert main(["agent", "Is RS14 ABC likely to fail its MOT?"]) == 0
    assert "tools used: lookup_vehicle, mot_risk" in capsys.readouterr().out
