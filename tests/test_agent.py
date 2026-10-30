import json
from datetime import date

import httpx
import pytest

from garage_copilot.agent import OpenAIToolAgent, parse_day
from garage_copilot.fleet import FleetService

WED = date(2026, 10, 28)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("tomorrow", date(2026, 10, 29)),
        ("next week", date(2026, 11, 2)),
        ("on Friday", date(2026, 10, 30)),
        ("on wednesday", date(2026, 11, 4)),  # same weekday means next week
        ("2026-12-01", date(2026, 12, 1)),
        ("03/11/2026", date(2026, 11, 3)),
        ("whenever", date(2026, 10, 29)),
    ],
)
def test_parse_day(text, expected):
    assert parse_day(text, WED) == expected


def test_parse_day_skips_weekend_by_default():
    assert parse_day("asap", date(2026, 10, 30)) == date(2026, 11, 2)


def test_fleet_allocates_and_respects_policy():
    fleet = FleetService.with_sample_fleet()
    day = date(2026, 11, 2)
    mot = fleet.book("AB19 CDE", "mot", day, want_courtesy_car=True)
    assert mot.courtesy_car is None and "four hours" in mot.notes[0]
    for _ in range(3):
        fleet.book("CD17 FGH", "service", day, want_courtesy_car=True)
    last = fleet.book("CD17 FGH", "service", day, want_courtesy_car=True)
    assert last.courtesy_car is None and "No courtesy car" in last.notes[0]
    with pytest.raises(ValueError):
        fleet.book("AB19 CDE", "mot", date(2026, 10, 31))  # Saturday


def test_agent_books_service_with_courtesy_car(copilot):
    result = copilot.agent.run("Book GH15 MNP in for a service on Friday and I need a courtesy car")
    assert result.tools_used == ["lookup_vehicle", "book_job"]
    assert "Friday 30 October" in result.reply
    assert "Courtesy car reserved" in result.reply


def test_agent_risk_question_uses_model_and_knowledge(copilot):
    result = copilot.agent.run("Is RS14 ABC likely to fail its MOT?")
    assert result.tools_used == ["lookup_vehicle", "mot_risk", "search_knowledge"]
    assert "MOT failure risk is" in result.reply


def test_agent_handles_unknown_vehicle(copilot):
    result = copilot.agent.run("Book ZZ99 ZZZ for an MOT")
    assert "No vehicle found" in result.reply


def test_agent_asks_for_registration(copilot):
    assert "Which registration" in copilot.agent.run("Can I book an MOT tomorrow?").reply


def test_agent_moves_weekend_courtesy_check(copilot):
    result = copilot.agent.run("Any courtesy cars free on Saturday?")
    assert "closed on Saturdays" in result.reply
    assert "Mon 02 Nov" in result.reply


def test_openai_tool_agent_runs_tool_loop(copilot):
    responses = iter(
        [
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "mot_risk",
                                        "arguments": json.dumps({"registration": "RS14 ABC"}),
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "High risk: book a pre-MOT check.",
                        }
                    }
                ]
            },
        ]
    )
    sent = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=next(responses))

    agent = OpenAIToolAgent(
        copilot.tools,
        "key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        today=lambda: WED,
    )
    result = agent.run("What's the MOT risk on RS14 ABC?")
    assert result.reply == "High risk: book a pre-MOT check."
    assert result.tools_used == ["mot_risk"] and result.steps[0].ok
    assert sent[0]["tools"][0]["type"] == "function"
    assert sent[1]["messages"][-1]["role"] == "tool"


def test_tool_registry_survives_bad_llm_arguments(copilot):
    result = copilot.tools.call("mot_risk", reg="AB19 CDE")  # wrong argument name
    assert not result.ok and "error" in result.output
