# state_of_neuro/tests/unit/test_generate_keywords.py
"""Unit tests for asynchronous keyword generation using precomputed responses."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import generate_keywords
import pytest


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden" / "keywords"
PROMPT_PATH = ROOT / "configs" / "keyword_prompt.json"


def test_generate_keywords_produces_expected_batches(tmp_path: Path) -> None:
    input_dir = tmp_path / "abstracts"
    input_dir.mkdir()
    shutil.copy(FIXTURES / "abstracts" / "abstracts_batch_0001.json", input_dir)

    output_dir = tmp_path / "keywords"
    config = generate_keywords.KeywordGenerationConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        responses_path=FIXTURES / "keywords" / "responses.json",
        prompt_path=PROMPT_PATH,
        max_concurrency=4,
    )

    result = generate_keywords.run(config)

    assert result.total_abstracts == 2
    assert result.keywords_generated == 6
    assert result.batches_processed == 1
    assert len(result.output_files) == 1

    output_path = result.output_files[0]
    generated = json.loads(output_path.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN.joinpath("keywords_batch_0001.json").read_text(encoding="utf-8"))
    assert generated == golden

    prompt_payload = json.loads(PROMPT_PATH.read_text(encoding="utf-8"))
    assert result.system_prompt == prompt_payload["system_prompt"]
    assert list(result.few_shot_messages or []) == prompt_payload["few_shot_messages"]


class _DummyResponse:
    def __init__(self, text: str):
        self.output_text = text


class _RecordingResponses:
    def __init__(self, response: _DummyResponse):
        self._response = response
        self.captured_payload: dict | None = None

    def create(self, **payload):
        self.captured_payload = payload
        return self._response


class _RecordingClient:
    def __init__(self, response_text: str):
        self.responses = _RecordingResponses(_DummyResponse(response_text))


def test_request_keywords_from_openai_builds_responses_payload() -> None:
    response_text = "keyword one; keyword two =//= topic"
    client = _RecordingClient(response_text)

    abstract = "Sample abstract about neural circuits."
    system_prompt = "System prompt"
    few_shot = [
        {"role": "user", "content": "Example input"},
        {"role": "assistant", "content": "Example output"},
    ]

    result = generate_keywords.request_keywords_from_openai(
        abstract,
        system_prompt=system_prompt,
        few_shot_messages=few_shot,
        model="gpt-4o-mini",
        temperature=0.3,
        max_output_tokens=256,
        client=client,
    )

    assert result == response_text

    payload = client.responses.captured_payload
    assert payload is not None
    assert payload["model"] == "gpt-4o-mini"
    assert payload["temperature"] == pytest.approx(0.3)
    assert payload["max_output_tokens"] == 256
    assert payload["instructions"] == system_prompt

    input_items = payload["input"]
    assert len(input_items) == 3
    assert input_items[0]["role"] == "user"
    assert input_items[0]["content"][0]["text"] == "Example input"
    assert input_items[1]["role"] == "assistant"
    assert input_items[1]["content"][0]["text"] == "Example output"
    assert input_items[2]["role"] == "user"
    assert input_items[2]["content"][0]["text"] == abstract
