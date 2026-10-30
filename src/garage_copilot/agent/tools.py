"""Tools the agent can call. Each tool has a JSON schema so the same registry works for the
offline planner and for LLM tool calling."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from garage_copilot.fleet import FleetService
from garage_copilot.ml.model import MotRiskModel
from garage_copilot.rag.assistant import RagAssistant
from garage_copilot.vehicles.lookup import VehicleLookup, VehicleNotFoundError


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    func: Callable[..., dict[str, Any]]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    ok: bool
    output: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"ok": self.ok, **self.output}, default=str)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def call(self, name: str, **args: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(name, args, False, {"error": f"Unknown tool {name}"})
        try:
            return ToolResult(name, args, True, self._tools[name].func(**args))
        except (VehicleNotFoundError, ValueError, TypeError) as exc:
            return ToolResult(name, args, False, {"error": str(exc)})


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def build_tools(
    lookup: VehicleLookup,
    risk_model: MotRiskModel,
    rag: RagAssistant,
    fleet: FleetService,
    today: Callable[[], date] = date.today,
) -> ToolRegistry:
    reg_param = {
        "registration": {"type": "string", "description": "UK registration, e.g. AB19 CDE"}
    }

    def lookup_vehicle(registration: str) -> dict[str, Any]:
        v = lookup.get(registration)
        latest = v.latest_mot
        return {
            "registration": v.registration,
            "vehicle": f"{v.year} {v.make} {v.model}",
            "fuel": v.fuel,
            "audio_system": v.audio_system,
            "mileage": v.current_mileage,
            "mot_due": v.mot_due(),
            "last_mot_passed": latest.passed if latest else None,
            "last_mot_advisories": [d.text for d in latest.advisories] if latest else [],
        }

    def mot_risk(registration: str) -> dict[str, Any]:
        score = risk_model.score(lookup.get(registration), today())
        return {
            "probability": score.probability,
            "band": score.band,
            "reasons": score.reasons,
            "recommend_pre_mot_check": score.recommend_pre_mot_check,
        }

    def search_knowledge(question: str) -> dict[str, Any]:
        answer = rag.ask(question)
        return {"answer": answer.text, "citations": answer.citations}

    def check_courtesy_cars(day: str) -> dict[str, Any]:
        d = date.fromisoformat(day)
        return {
            "day": d,
            "available": [f"{c.model} ({c.registration})" for c in fleet.available_cars(d)],
        }

    def book_job(
        registration: str, job: str, day: str, courtesy_car: bool = False
    ) -> dict[str, Any]:
        v = lookup.get(registration)
        requested = date.fromisoformat(day)
        actual = fleet.next_open_day(requested)
        booking = fleet.book(v.registration, job, actual, want_courtesy_car=courtesy_car)  # type: ignore[arg-type]
        return {
            "booking_id": booking.id,
            "registration": booking.registration,
            "job": booking.job,
            "day": booking.day,
            "moved_from": requested if actual != requested else None,
            "courtesy_car": (
                f"{booking.courtesy_car.model} ({booking.courtesy_car.registration})"
                if booking.courtesy_car
                else None
            ),
            "notes": booking.notes,
        }

    return ToolRegistry(
        [
            Tool(
                "lookup_vehicle",
                "Look up a vehicle and its MOT history by registration.",
                _obj(reg_param, ["registration"]),
                lookup_vehicle,
            ),
            Tool(
                "mot_risk",
                "Predict the chance a vehicle fails its next MOT, with reasons.",
                _obj(reg_param, ["registration"]),
                mot_risk,
            ),
            Tool(
                "search_knowledge",
                "Answer a question from the workshop knowledge base (MOT, servicing, "
                "courtesy car terms, audio fitting, booking policy). Returns citations.",
                _obj({"question": {"type": "string"}}, ["question"]),
                search_knowledge,
            ),
            Tool(
                "check_courtesy_cars",
                "List courtesy cars free on a day (YYYY-MM-DD).",
                _obj({"day": {"type": "string", "format": "date"}}, ["day"]),
                check_courtesy_cars,
            ),
            Tool(
                "book_job",
                "Book a vehicle into the workshop. Moves to the next open weekday if needed.",
                _obj(
                    {
                        **reg_param,
                        "job": {
                            "type": "string",
                            "enum": ["mot", "service", "repair", "audio_fit"],
                        },
                        "day": {"type": "string", "format": "date"},
                        "courtesy_car": {"type": "boolean"},
                    },
                    ["registration", "job", "day"],
                ),
                book_job,
            ),
        ]
    )
