"""Runtime settings, read from environment variables.

Everything defaults to an offline mode so the project runs (and its tests pass)
without any API keys. Real providers are opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"


@dataclass(frozen=True)
class Settings:
    preset: str = "garage"
    llm_provider: str = "offline"  # offline | openai | anthropic
    llm_model: str | None = None
    openai_api_key: str | None = field(default=None, repr=False)
    anthropic_api_key: str | None = field(default=None, repr=False)

    vehicle_source: str = "mock"  # mock | dvsa
    dvsa_api_key: str | None = field(default=None, repr=False)
    dvsa_access_token: str | None = field(default=None, repr=False)

    api_key: str | None = field(default=None, repr=False)

    knowledge_dir: Path = DATA_DIR / "knowledge"
    vehicles_file: Path = DATA_DIR / "vehicles.json"
    model_path: Path = Path(".garage_copilot") / "mot_risk.joblib"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            preset=env.get("GARAGE_PRESET", "garage").lower(),
            llm_provider=env.get("GARAGE_LLM_PROVIDER", "offline").lower(),
            llm_model=env.get("GARAGE_LLM_MODEL"),
            openai_api_key=env.get("OPENAI_API_KEY"),
            anthropic_api_key=env.get("ANTHROPIC_API_KEY"),
            vehicle_source=env.get("GARAGE_VEHICLE_SOURCE", "mock").lower(),
            dvsa_api_key=env.get("DVSA_API_KEY"),
            dvsa_access_token=env.get("DVSA_ACCESS_TOKEN"),
            api_key=env.get("GARAGE_API_KEY"),
            knowledge_dir=Path(env.get("GARAGE_KNOWLEDGE_DIR", DATA_DIR / "knowledge")),
            model_path=Path(
                env.get("GARAGE_MODEL_PATH", Path(".garage_copilot") / "mot_risk.joblib")
            ),
        )
