"""The Garage Copilot agent.

Two planners share one tool registry:

* `RuleBasedAgent` parses intent with simple rules. It is deterministic, needs no API key
  and is what the tests exercise.
* `OpenAIToolAgent` lets an LLM choose tools through OpenAI-style function calling.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import httpx

from garage_copilot.agent.tools import ToolRegistry, ToolResult
from garage_copilot.vehicles.lookup import find_registration

_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


@dataclass
class AgentResponse:
    reply: str
    steps: list[ToolResult] = field(default_factory=list)

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps]


def parse_day(text: str, today: date) -> date:
    """Understands 'today', 'tomorrow', 'next week', weekday names and ISO or UK dates.
    Defaults to the next working day."""
    t = text.lower()
    if m := re.search(r"\b(\d{4}-\d{2}-\d{2})\b", t):
        return date.fromisoformat(m.group(1))
    if m := re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", t):
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    if "tomorrow" in t:
        return today + timedelta(days=1)
    if "next week" in t:
        return today + timedelta(days=7 - today.weekday())
    for i, name in enumerate(_WEEKDAYS):
        if re.search(rf"\b{name}\b", t):
            ahead = (i - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)
    if re.search(r"\btoday\b", t):
        return today
    nxt = today + timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += timedelta(days=1)
    return nxt


def _job_from_text(t: str) -> str:
    if any(w in t for w in ("head unit", "stereo", "audio", "speaker", "camera", "fit ")):
        return "audio_fit"
    if "service" in t:
        return "service"
    if any(w in t for w in ("repair", "fix", "brake", "clutch")):
        return "repair"
    return "mot"


class RuleBasedAgent:
    def __init__(self, tools: ToolRegistry, today: Callable[[], date] = date.today):
        self.tools = tools
        self.today = today

    def run(self, message: str) -> AgentResponse:
        t = message.lower()
        reg = find_registration(message)
        wants_booking = bool(re.search(r"\b(book|booking|bring .* in|slot)\b", t))
        wants_courtesy = any(
            w in t for w in ("courtesy", "loan car", "hire car", "replacement car")
        )
        wants_risk = any(w in t for w in ("risk", "fail", "pass", "likely", "pre-mot", "mot"))
        is_question = "?" in message or t.startswith(
            ("what", "how", "can", "do", "will", "is", "when")
        )

        steps: list[ToolResult] = []
        lines: list[str] = []

        if reg:
            vehicle = self.tools.call("lookup_vehicle", registration=reg)
            steps.append(vehicle)
            if not vehicle.ok:
                return AgentResponse(vehicle.output["error"], steps)
            v = vehicle.output
            lines.append(
                f"{v['registration']} is a {v['vehicle']} on {v['mileage']:,} miles, "
                f"MOT due {v['mot_due']:%d %b %Y}."
            )

            if wants_risk and "mot_risk" in self.tools:
                risk = self.tools.call("mot_risk", registration=reg)
                steps.append(risk)
                r = risk.output
                why = f" ({', '.join(r['reasons'])})" if r["reasons"] else ""
                lines.append(f"MOT failure risk is {r['band']} at {r['probability']:.0%}{why}.")
                if r["recommend_pre_mot_check"]:
                    lines.append("I'd recommend our 20-minute pre-MOT check.")

        if wants_booking and "book_job" in self.tools:
            if not reg:
                return AgentResponse("Which registration should I book in?", steps)
            day = parse_day(message, self.today())
            booking = self.tools.call(
                "book_job",
                registration=reg,
                job=_job_from_text(t),
                day=day.isoformat(),
                courtesy_car=wants_courtesy,
            )
            steps.append(booking)
            if booking.ok:
                b = booking.output
                moved = f" (moved from {b['moved_from']:%a %d %b})" if b["moved_from"] else ""
                lines.append(
                    f"Booked {b['job'].replace('_', ' ')} #{b['booking_id']} for "
                    f"{b['day']:%A %d %B}{moved}."
                )
                if b["courtesy_car"]:
                    lines.append(f"Courtesy car reserved: {b['courtesy_car']}.")
                lines.extend(b["notes"])
            else:
                lines.append(f"Couldn't book: {booking.output['error']}")
        elif wants_courtesy and "check_courtesy_cars" in self.tools:
            day = parse_day(message, self.today())
            if day.weekday() >= 5:
                lines.append(
                    f"The workshop is closed on {day:%A}s, so I've checked the next working day."
                )
                day += timedelta(days=7 - day.weekday())
            cars = self.tools.call("check_courtesy_cars", day=day.isoformat())
            steps.append(cars)
            free = cars.output["available"]
            lines.append(f"Free courtesy cars on {day:%a %d %b}: {', '.join(free) or 'none'}.")

        if is_question and not wants_booking and "search_knowledge" in self.tools:
            knowledge = self.tools.call("search_knowledge", question=message)
            steps.append(knowledge)
            k = knowledge.output
            lines.append(k["answer"])
            if k["citations"]:
                lines.append("Sources: " + "; ".join(dict.fromkeys(k["citations"])))

        if not lines:
            lines.append(
                "I can look up a car, predict MOT risk, book jobs, check courtesy cars "
                "or answer questions about our policies. Try including a registration."
            )
        return AgentResponse("\n".join(lines), steps)


class OpenAIToolAgent:
    """LLM-driven tool use via the OpenAI chat completions API."""

    URL = "https://api.openai.com/v1/chat/completions"
    SYSTEM = (
        "You are Garage Copilot, an assistant for a UK garage. Use the tools to look up "
        "vehicles, predict MOT risk, check policy and book jobs. Today is {today}. Never "
        "invent vehicle data or policy: call a tool. Be brief and cite policy sources."
    )

    def __init__(
        self,
        tools: ToolRegistry,
        api_key: str,
        model: str = "gpt-4o-mini",
        client: httpx.Client | None = None,
        max_steps: int = 6,
        today: Callable[[], date] = date.today,
    ):
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self.today = today
        self._client = client or httpx.Client(timeout=60.0)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def run(self, message: str) -> AgentResponse:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.SYSTEM.format(today=self.today().isoformat())},
            {"role": "user", "content": message},
        ]
        steps: list[ToolResult] = []
        for _ in range(self.max_steps):
            resp = self._client.post(
                self.URL,
                headers=self._headers,
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": self.tools.schemas(),
                    "temperature": 0,
                },
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                return AgentResponse(msg.get("content") or "", steps)
            messages.append(msg)
            for call in calls:
                try:
                    args = json.loads(call["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = self.tools.call(call["function"]["name"], **args)
                steps.append(result)
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result.to_json()}
                )
        return AgentResponse("Stopped after too many steps; please rephrase.", steps)
