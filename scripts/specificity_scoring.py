# state_of_neuro/scripts/specificity_scoring.py
"""Score keyword specificity with the historical classifier and cached decisions.

Recovered from bozelosp/state-of-neuroscience at 61358677e2ce862dd465c99c61541782d8549a70.
Source: all_categories/tag_specificity_flagging.py. API failures propagate;
missing decisions are never silently replaced by a generic classification.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

EXAMPLE_TOO_GENERIC = [
    "rats", "mice", "in vitro", "in vivo", "animals",
    "cells", "age", "training", "exercise", "behavior"
]

EXAMPLE_GENERIC = [
    "transgenic mice", "knockout mice", "rodent models",
    "mouse models", "adult rats", "cell culture",
    "behavioral tests", "experimental design", "rodents", "animal model"
]

EXAMPLE_SPECIFIC_ENOUGH = [
    "C57BL/6 mice", "Sprague-Dawley rats", "Wistar rats",
    "zebrafish", "Drosophila melanogaster", "SH-SY5Y cells",
    "Morris water maze", "6-hydroxydopamine",
    "Long-Evans rats", "Caenorhabditis elegans"
]

EXAMPLE_VERY_SPECIFIC = [
    "APP/PS1 mice", "3xTg-AD mice", "LRRK2 G2019S knock-in",
    "Thy1-YFP-H line", "5XFAD mice", "hSOD1-G93A transgenic",
    "TDP-43 A315T mice", "db/db mice", "shiverer mice", "R6/2 huntingtin mice"
]

def build_system_prompt() -> str:
    """Return the enhanced system prompt with training examples."""
    return f"""You are the Tag-Specificity Inspector for neuroscience literature.

Your task is to classify how specific a given tag is, using exactly one of these categories:
  "too_generic"     – broader than useful for research or too vague
  "generic"         – broad but still within scientific scope
  "specific_enough" – concrete, informative, fairly common in literature
  "very_specific"   – ultra-niche, specific strains/mutations/identifiers

Training examples to guide your classification:

Too Generic (avoid these):
{EXAMPLE_TOO_GENERIC}

Generic (acceptable but broad):
{EXAMPLE_GENERIC}

Specific Enough (good specificity):
{EXAMPLE_SPECIFIC_ENOUGH}

Very Specific (highly specialized):
{EXAMPLE_VERY_SPECIFIC}

Output ONLY JSON with exactly one boolean set to true:
{{"too_generic": <bool>, "generic": <bool>, "specific_enough": <bool>, "very_specific": <bool>}}

** Exactly one must be true **"""

def build_user_prompt(tag: str) -> str:
    """Return the user prompt containing just the tag."""
    return f'Tag: "{tag}"'

SCHEMA = {
    "type": "object",
    "properties": {
        "too_generic":     {"type": "boolean"},
        "generic":         {"type": "boolean"},
        "specific_enough": {"type": "boolean"},
        "very_specific":   {"type": "boolean"},
    },
    "required": ["too_generic", "generic", "specific_enough", "very_specific"],
    "additionalProperties": False,
}


@dataclass
class SpecificityConfig:
    records_path: Path
    output_path: Path
    model: str | None = None
    batch_size: int = 32
    temperature: float | None = 0.0


def validate_flags(flags: dict, tag: str) -> None:
    """Require an actual classifier decision, rather than a fabricated default."""
    keys = SCHEMA["required"]
    if any(type(flags.get(key)) is not bool for key in keys) or sum(flags[k] for k in keys) != 1:
        raise ValueError(f"Invalid specificity decision for {tag!r}: expected one true flag")


def run(config: SpecificityConfig, client=None) -> dict:
    """Reuse saved scores and classify missing terms with the recovered prompt."""
    records = json.loads(config.records_path.read_text(encoding="utf-8"))
    frequencies = {record["label"]: record["frequency"] for record in records}
    flags = json.loads(config.output_path.read_text()) if config.output_path.exists() else {}
    for tag in frequencies.keys() & flags.keys():
        validate_flags(flags[tag], tag)
    todo = [tag for tag in frequencies if tag not in flags]
    if todo and not config.model:
        raise ValueError("Missing specificity decisions: provide the recorded cache or set model for a new run")
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        temporary = config.output_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(flags, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(config.output_path)

    async def classify(api) -> None:
        for offset in range(0, len(todo), config.batch_size):
            batch = todo[offset:offset + config.batch_size]
            responses = await asyncio.gather(*[
                api.responses.create(
                    model=config.model,
                    instructions=build_system_prompt(),
                    input=build_user_prompt(tag),
                    **({"temperature": config.temperature} if config.temperature is not None else {}),
                    text={"format": {"type": "json_schema", "strict": True,
                                     "name": "tag_specificity", "schema": SCHEMA}},
                ) for tag in batch
            ])
            for tag, response in zip(batch, responses):
                decision = json.loads(response.output_text)
                validate_flags(decision, tag)
                flags[tag] = {**decision, "model": config.model,
                              "frequency": frequencies[tag]}
            save()

    async def execute() -> None:
        if client is not None:
            await classify(client)
        else:
            from openai import AsyncOpenAI
            async with AsyncOpenAI() as api:
                await classify(api)

    if todo:
        asyncio.run(execute())
    for tag, frequency in frequencies.items():
        flags[tag]["frequency"] = frequency
    save()
    return {"tags_scored": len(frequencies), "new_decisions": len(todo),
            "output_path": str(config.output_path)}
