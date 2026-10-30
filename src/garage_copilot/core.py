"""Wires the components together. One place to swap adapters (mock vs DVSA, offline vs LLM)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from garage_copilot.agent.agent import OpenAIToolAgent, RuleBasedAgent
from garage_copilot.agent.tools import ToolRegistry, build_tools
from garage_copilot.config import Settings
from garage_copilot.fleet import FleetService
from garage_copilot.llm import build_llm
from garage_copilot.ml.model import MotRiskModel
from garage_copilot.rag.assistant import RagAssistant
from garage_copilot.vehicles.lookup import DvsaMotHistoryClient, MockVehicleLookup, VehicleLookup


@dataclass
class Copilot:
    settings: Settings
    lookup: VehicleLookup
    rag: RagAssistant
    risk_model: MotRiskModel
    fleet: FleetService
    tools: ToolRegistry
    agent: RuleBasedAgent | OpenAIToolAgent

    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        today: Callable[[], date] = date.today,
        risk_model: MotRiskModel | None = None,
    ) -> Copilot:
        settings = settings or Settings.from_env()
        lookup = _build_lookup(settings)
        rag = RagAssistant.from_directory(settings.knowledge_dir, llm=build_llm(settings))
        model = risk_model or MotRiskModel.load_or_train(settings.model_path)
        fleet = FleetService.with_sample_fleet()
        tools = build_tools(lookup, model, rag, fleet, today=today)
        agent: RuleBasedAgent | OpenAIToolAgent
        if settings.llm_provider == "openai" and settings.openai_api_key:
            agent = OpenAIToolAgent(
                tools, settings.openai_api_key, settings.llm_model or "gpt-4o-mini", today=today
            )
        else:
            agent = RuleBasedAgent(tools, today=today)
        return cls(settings, lookup, rag, model, fleet, tools, agent)


def _build_lookup(settings: Settings) -> VehicleLookup:
    if settings.vehicle_source == "dvsa":
        if not (settings.dvsa_api_key and settings.dvsa_access_token):
            raise ValueError("GARAGE_VEHICLE_SOURCE=dvsa needs DVSA_API_KEY and DVSA_ACCESS_TOKEN")
        return DvsaMotHistoryClient(settings.dvsa_api_key, settings.dvsa_access_token)
    return MockVehicleLookup.from_file(settings.vehicles_file)
