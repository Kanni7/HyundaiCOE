"""Semantic-parser plumbing with a mocked LLM (fast), plus real-model integration tests.

Run integration tests with:  pytest -m integration
"""
import json

import pytest

from bhashadrive.config import LLMConfig
from bhashadrive.nlu.base import LLMTimeoutError, LLMUnavailableError
from bhashadrive.nlu.local_llm import MockLLM, OllamaLLM
from bhashadrive.nlu.semantic_parser import SYSTEM_PROMPT, LLMSemanticParser
from bhashadrive.evaluation.evaluate import load_dataset

from .test_schemas import good_raw


def test_context_summary_is_sent_to_model():
    llm = MockLLM(good_raw())
    LLMSemanticParser(llm).parse("A bit more", "Last action executed: increase the volume")
    assert "Last action executed: increase the volume" in llm.calls[0][1]
    assert "A bit more" in llm.calls[0][1]


def test_remote_endpoint_refused_by_default():
    with pytest.raises(LLMUnavailableError, match="non-local"):
        OllamaLLM(LLMConfig(endpoint="https://api.example.com"))


def test_unreachable_ollama_reports_actionable_error():
    llm = OllamaLLM(LLMConfig(endpoint="http://127.0.0.1:9", timeout_s=1))
    with pytest.raises(LLMUnavailableError, match="ollama serve"):
        llm.health_check()
    with pytest.raises((LLMUnavailableError, LLMTimeoutError)):  # Windows may time out instead of refusing
        llm.generate_json("s", "u")


def test_dataset_texts_not_copied_into_prompt():
    """Held-out check: evaluation utterances must not appear verbatim as prompt examples."""
    leaked = [c["text"] for c in load_dataset() if len(c["text"]) > 12 and c["text"].lower() in SYSTEM_PROMPT.lower()]
    assert leaked == []


def test_dataset_is_well_formed():
    cases = load_dataset()
    assert len(cases) >= 80
    cats = {c["category"] for c in cases}
    assert {"explicit", "indirect", "paraphrase", "code_switch", "hindi", "negation", "ambiguous", "followup",
            "followup_no_context", "unsupported", "out_of_domain", "asr_error", "unauthorized"} <= cats


# ------------------------------------------------------------------ integration (real local model)
def _real_parser():
    llm = OllamaLLM()
    try:
        llm.health_check()
    except LLMUnavailableError as exc:
        pytest.skip(str(exc))
    llm.warmup()
    return LLMSemanticParser(llm)


@pytest.mark.integration
@pytest.mark.parametrize("text,intent,direction", [
    ("Bhai, volume thoda badha do.", "volume_adjust", "increase"),
    ("I'm freezing.", "climate_adjust_temperature", "increase"),
    ("Fan ki speed kam karo.", "fan_adjust_speed", "decrease"),
])
def test_real_model_core_cases(text, intent, direction):
    f = _real_parser().parse(text, "Pending proposal awaiting the driver's yes/no: none\nLast action executed: none")
    assert f.intent.value == intent and f.parameters.direction.value == direction


@pytest.mark.integration
def test_real_model_negation():
    f = _real_parser().parse("Please don't turn the AC off.", "Pending proposal awaiting the driver's yes/no: none\nLast action executed: none")
    assert f.negated is True
