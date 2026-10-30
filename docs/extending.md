# Extending Garage Copilot

A short playbook for the four changes people make most often. Each one is a small, local change; none of them touch the agent or the API.

## 1. Add a knowledge document

Drop a markdown file into `src/garage_copilot/data/knowledge/` with front matter:

```markdown
---
title: Tyre policy
source: Sample workshop policy (illustrative)
---

# Tyre policy

## Puncture repairs

We repair punctures in the central three quarters of the tread...
```

Each `##` section becomes a chunk with its own citation. Then add two or three questions to `data/eval/rag_questions.jsonl` and run `garage-copilot eval` to check retrieval still scores well.

## 2. Add a tool

Tools live in `agent/tools.py`. A tool is a function plus a JSON schema, so the same tool works for the offline planner and for LLM function calling.

```python
def tyre_quote(registration: str, tyres: int = 1) -> dict[str, Any]:
    v = lookup.get(registration)
    return {"vehicle": f"{v.make} {v.model}", "price": 85 * tyres}

Tool(
    "tyre_quote",
    "Quote for new tyres on a vehicle.",
    _obj({**reg_param, "tyres": {"type": "integer", "minimum": 1, "maximum": 4}}, ["registration"]),
    tyre_quote,
)
```

Add it to the list in `build_tools`, add its name to the presets that should have it, and write a test in `tests/test_agent.py`.

## 3. Add a preset

A preset configures the copilot for a type of business without new code. Add an entry to `presets.py`:

```python
Preset(
    "tyre_centre",
    "Tyre and wheel alignment centre",
    knowledge=("tyre-policy", "booking-policy"),
    tools=("lookup_vehicle", "search_knowledge", "book_job", "tyre_quote"),
)
```

Run it with `GARAGE_PRESET=tyre_centre garage-copilot serve`. `tests/test_presets.py` checks every preset builds.

## 4. Swap an adapter

The rest of the app depends on small interfaces, not concrete services:

| Interface | Ships with | Swap in |
| --- | --- | --- |
| `VehicleLookup` | `MockVehicleLookup`, `DvsaMotHistoryClient` | A DMS or CRM API |
| `Embedder` | `TfidfEmbedder`, `OpenAIEmbedder` | Any embedding model |
| `LLMClient` | `OpenAIClient`, `AnthropicClient` | A local model server |

Write a class with the same method, register it in `core.py` behind a setting, and test it with `httpx.MockTransport` so CI needs no network.
