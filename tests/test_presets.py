import pytest

from garage_copilot.config import Settings
from garage_copilot.core import Copilot
from garage_copilot.presets import PRESETS, get_preset
from tests.conftest import TODAY


def build(preset: str, tmp_path, risk_model) -> Copilot:
    settings = Settings(preset=preset, model_path=tmp_path / "m.joblib")
    return Copilot.build(settings, today=lambda: TODAY, risk_model=risk_model)


def test_unknown_preset():
    with pytest.raises(ValueError, match="Choose from"):
        get_preset("bakery")


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_builds(name, tmp_path, risk_model):
    cp = build(name, tmp_path, risk_model)
    assert cp.tools.names == [t for t in cp.tools.names if t in PRESETS[name].tools]
    assert len(cp.rag.store) > 0


def test_audio_fitter_only_knows_audio_and_booking(tmp_path, risk_model):
    cp = build("audio_fitter", tmp_path, risk_model)
    titles = {h.chunk.doc_title for h in cp.rag.store.search("courtesy car excess", k=10)}
    assert "Courtesy car terms" not in titles
    assert "mot_risk" not in cp.tools


def test_fleet_desk_cannot_book_jobs(tmp_path, risk_model):
    cp = build("fleet_hire", tmp_path, risk_model)
    reply = cp.agent.run("Book AB19 CDE for an MOT and is a courtesy car free tomorrow?").reply
    assert "Booked" not in reply
    assert "Free courtesy cars" in reply
