"""FastAPI service. Run with: uvicorn --factory garage_copilot.api:create_app"""

from __future__ import annotations

import secrets
from datetime import date
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from garage_copilot import __version__
from garage_copilot.core import Copilot
from garage_copilot.vehicles.lookup import VehicleNotFoundError
from garage_copilot.vehicles.models import Vehicle


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    citations: list[str]


class AgentRequest(BaseModel):
    message: str = Field(min_length=2, max_length=1000)


class AgentStepOut(BaseModel):
    tool: str
    args: dict[str, Any]
    ok: bool


class AgentResponseOut(BaseModel):
    reply: str
    steps: list[AgentStepOut]


class RiskOut(BaseModel):
    registration: str
    probability: float
    band: str
    reasons: list[str]
    recommend_pre_mot_check: bool


def create_app(copilot: Copilot | None = None) -> FastAPI:
    state: dict[str, Copilot] = {}

    def cp() -> Copilot:
        if "copilot" not in state:
            state["copilot"] = copilot or Copilot.build()
        return state["copilot"]

    def require_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = cp().settings.api_key
        if expected and not secrets.compare_digest(x_api_key or "", expected):
            raise HTTPException(401, "Missing or invalid X-API-Key")

    app = FastAPI(
        title="Garage Copilot",
        version=__version__,
        description="Vehicle lookup, workshop knowledge RAG, MOT risk prediction and a "
        "tool-using agent for garages and fleet operators.",
        dependencies=[Depends(require_key)],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        c = cp()
        return {
            "status": "ok",
            "version": __version__,
            "llm": c.settings.llm_provider,
            "vehicle_source": c.settings.vehicle_source,
            "knowledge_chunks": len(c.rag.store),
            "model_auc": c.risk_model.metrics.get("roc_auc"),
        }

    def get_vehicle(registration: str) -> Vehicle:
        try:
            return cp().lookup.get(registration)
        except VehicleNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/vehicles/{registration}")
    def vehicle(registration: str) -> dict[str, Any]:
        v = get_vehicle(registration)
        return {**v.model_dump(), "mot_due": v.mot_due()}

    @app.get("/vehicles/{registration}/mot-risk", response_model=RiskOut)
    def mot_risk(registration: str) -> RiskOut:
        v = get_vehicle(registration)
        s = cp().risk_model.score(v)
        return RiskOut(
            registration=v.registration,
            probability=s.probability,
            band=s.band,
            reasons=s.reasons,
            recommend_pre_mot_check=s.recommend_pre_mot_check,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        answer = cp().rag.ask(req.question)
        return AskResponse(answer=answer.text, citations=answer.citations)

    @app.get("/fleet/availability")
    def availability(day: date) -> dict[str, Any]:
        cars = cp().fleet.available_cars(day)
        return {"day": day, "available": [c.__dict__ for c in cars]}

    @app.post("/agent", response_model=AgentResponseOut)
    def agent(req: AgentRequest) -> AgentResponseOut:
        result = cp().agent.run(req.message)
        return AgentResponseOut(
            reply=result.reply,
            steps=[AgentStepOut(tool=s.tool, args=s.args, ok=s.ok) for s in result.steps],
        )

    return app
