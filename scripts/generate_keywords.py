# state_of_neuro/scripts/generate_keywords.py
"""Step 4 of the pipeline - replay asynchronous keyword generation.

Instead of calling the OpenAI API, this module loads precomputed keyword
responses and replays the original concurrency profile with ``asyncio``. Each
incoming abstract must already include the ``hash_id`` produced by
``scripts.prepare_abstracts`` so the generated keywords align with the cached
responses on disk.

Common entry points include ``scripts/run_step.py keyword_generation`` or a
direct call using :class:`KeywordGenerationConfig`.

Prerequisites
-------------
- Abstract batches named ``abstracts_batch_*.json`` inside ``input_dir``.
- Keyword response fixture (JSON object keyed by ``hash_id``) supplied via
  ``responses_path``.
- Optional prompt bundle containing ``system_prompt`` and ``few_shot_messages``.

Outputs
-------
- JSON files ``keywords_batch_*.json`` with lists of inferred keywords keyed by
  ``hash_id``.
- Aggregate counts returned via :class:`KeywordGenerationResult` for logging and
  audit trails.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass
class KeywordGenerationConfig:
    """Configuration for keyword generation using precomputed responses."""

    input_dir: Path
    output_dir: Path
    responses_path: Path
    max_concurrency: int = 50
    prompt_path: Path | None = None


@dataclass
class KeywordGenerationResult:
    """Summary of keyword generation."""

    total_abstracts: int
    keywords_generated: int
    batches_processed: int
    output_files: Sequence[Path]
    system_prompt: str | None
    few_shot_messages: Sequence[Mapping[str, str]] | None


def run(config: KeywordGenerationConfig) -> KeywordGenerationResult:
    """Emit keywords for each batch using precomputed responses."""

    if not config.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if not config.responses_path.exists():
        raise FileNotFoundError(f"Keyword responses file not found: {config.responses_path}")
    if config.max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")

    abstract_files = sorted(config.input_dir.glob("abstracts_batch_*.json"))
    if not abstract_files:
        raise FileNotFoundError(f"No abstract batches found in {config.input_dir}")

    responses = _load_responses(config.responses_path)
    system_prompt: str | None = None
    few_shot_messages: Sequence[Mapping[str, str]] | None = None
    if config.prompt_path:
        prompt_payload = _load_prompt(config.prompt_path)
        system_prompt = prompt_payload.get("system_prompt")
        few_shot_messages = prompt_payload.get("few_shot_messages") or []
    config.output_dir.mkdir(parents=True, exist_ok=True)

    total_abstracts = 0
    keywords_generated = 0
    output_files: List[Path] = []

    for index, batch_path in enumerate(abstract_files, start=1):
        abstracts = _load_abstracts(batch_path)
        total_abstracts += len(abstracts)
        keyword_map = asyncio.run(
            _generate_keywords_for_batch(abstracts, responses, config.max_concurrency)
        )
        keywords_generated += sum(len(keywords) for keywords in keyword_map.values())
        output_path = config.output_dir / f"keywords_batch_{index:04d}.json"
        output_path.write_text(
            json.dumps(keyword_map, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_files.append(output_path)

    return KeywordGenerationResult(
        total_abstracts=total_abstracts,
        keywords_generated=keywords_generated,
        batches_processed=len(abstract_files),
        output_files=tuple(output_files),
        system_prompt=system_prompt,
        few_shot_messages=tuple(few_shot_messages) if few_shot_messages else None,
    )


def request_keywords_from_openai(
    abstract: str,
    *,
    system_prompt: str | None,
    few_shot_messages: Sequence[Mapping[str, str]] | None = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    client: Any | None = None,
) -> str:
    """Call the OpenAI Responses API to generate keywords for ``abstract``.

    Parameters
    ----------
    abstract:
        The abstract text to tag.
    system_prompt:
        Optional system instructions (``instructions`` field in the Responses API).
    few_shot_messages:
        Optional sequence of prior user/assistant messages to include in the request.
    model:
        Model ID used for keyword extraction (defaults to ``gpt-4o-mini``).
    temperature:
        Sampling temperature for the Responses API call (defaults to ``0.0`` for deterministic behaviour).
    max_output_tokens:
        Optional upper bound for generated tokens.
    client:
        An ``openai.OpenAI`` client instance. When ``None`` a client is created on demand.

    Returns
    -------
    str
        The raw keyword/topic string returned by the model.
    """

    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - only triggered when package is absent.
        raise RuntimeError(
            "Install the 'openai' package (>=1.0.0) to issue live keyword extraction requests."
        ) from exc

    client = client or OpenAI()
    input_items = _build_responses_input(abstract, few_shot_messages)

    payload: Dict[str, Any] = {
        "model": model,
        "input": input_items,
        "temperature": float(temperature),
    }
    if system_prompt:
        payload["instructions"] = system_prompt
    if max_output_tokens is not None:
        payload["max_output_tokens"] = int(max_output_tokens)

    response = client.responses.create(**payload)
    return _extract_text_from_response(response)


def _build_responses_input(
    abstract: str, few_shot_messages: Sequence[Mapping[str, str]] | None
) -> List[Mapping[str, Any]]:
    items: List[Mapping[str, Any]] = []
    if few_shot_messages:
        for message in few_shot_messages:
            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                continue
            items.append(
                {
                    "role": str(role),
                    "content": [{"type": "text", "text": str(content)}],
                }
            )
    items.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": abstract}],
        }
    )
    return items


def _extract_text_from_response(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    if isinstance(output_text, list):
        combined = "".join(str(part) for part in output_text if part)
        if combined.strip():
            return combined.strip()

    output = getattr(response, "output", None)
    if output is None and hasattr(response, "model_dump"):
        try:
            output = response.model_dump().get("output")
        except Exception:  # pragma: no cover - defensive fallback.
            output = None

    texts: List[str] = []
    for item in output or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        if isinstance(content, list):
            for entry in content:
                text_value = _extract_text_field(entry)
                if text_value:
                    texts.append(text_value)
        else:
            text_value = _extract_text_field(content)
            if text_value:
                texts.append(text_value)

    combined = "".join(texts).strip()
    if combined:
        return combined

    raise RuntimeError("OpenAI response did not contain text output.")


def _extract_text_field(entry: Any) -> str | None:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        text_value = entry.get("text")
        if text_value:
            return str(text_value)
    text_attr = getattr(entry, "text", None)
    if text_attr:
        return str(text_attr)
    return None


def _load_responses(path: Path) -> Mapping[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Keyword responses must be a JSON object.")
    normalised: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if not isinstance(value, list):
            raise ValueError(f"Keyword response for {key!r} must be a list.")
        normalised[str(key)] = [str(item) for item in value]
    return normalised


def _load_abstracts(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Abstract batch must be a list: {path}")
    return payload


def _load_prompt(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Keyword prompt file must be a JSON object.")
    return payload


async def _generate_keywords_for_batch(
    abstracts: Iterable[dict],
    responses: Mapping[str, Sequence[str]],
    max_concurrency: int,
) -> Dict[str, Sequence[str]]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _produce(record: dict) -> tuple[str, Sequence[str]]:
        async with semaphore:
            hash_id = str(record.get("hash_id") or "")
            if not hash_id:
                raise ValueError("Abstract record missing 'hash_id'.")
            if hash_id not in responses:
                raise KeyError(f"No keyword response recorded for hash_id {hash_id}")
            # Yield control to preserve concurrency characteristics.
            await asyncio.sleep(0)
            return hash_id, responses[hash_id]

    tasks = [_produce(record) for record in abstracts]
    results = await asyncio.gather(*tasks)
    return {hash_id: keywords for hash_id, keywords in results}


__all__ = [
    "KeywordGenerationConfig",
    "KeywordGenerationResult",
    "run",
    "request_keywords_from_openai",
]
