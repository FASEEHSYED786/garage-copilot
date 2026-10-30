"""Presets: one codebase, configured per type of business.

A preset picks which knowledge documents are indexed and which tools the agent can use.
Adding a new business type is a new entry here, not new code.
"""

from __future__ import annotations

from dataclasses import dataclass

ALL_TOOLS = ("lookup_vehicle", "mot_risk", "search_knowledge", "check_courtesy_cars", "book_job")


@dataclass(frozen=True)
class Preset:
    name: str
    description: str
    knowledge: tuple[str, ...] | None  # document file stems, None = all
    tools: tuple[str, ...] = ALL_TOOLS


PRESETS: dict[str, Preset] = {
    p.name: p
    for p in [
        Preset("garage", "Full workshop: MOTs, servicing, repairs and courtesy cars", None),
        Preset(
            "audio_fitter",
            "Car audio and accessory fitter",
            ("audio-fitting", "booking-policy"),
            ("lookup_vehicle", "search_knowledge", "book_job"),
        ),
        Preset(
            "fleet_hire",
            "Courtesy car or pool fleet desk",
            ("courtesy-car-terms", "booking-policy"),
            ("lookup_vehicle", "search_knowledge", "check_courtesy_cars"),
        ),
    ]
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"Unknown preset {name!r}. Choose from: {', '.join(PRESETS)}") from None
