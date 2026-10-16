import pytest
from pydantic import ValidationError

from bhashadrive.nlu.base import LLMOutputError
from bhashadrive.nlu.local_llm import MockLLM
from bhashadrive.nlu.semantic_parser import FRAME_JSON_SCHEMA, LLMSemanticParser
from bhashadrive.schemas import Command, Intent, Parameters, SemanticFrame


def good_raw(**over):
    raw = {"meaning": "driver wants it louder", "negated": False, "language_pattern": "hinglish",
           "user_goal": "Louder", "speech_act": "direct_command", "intent": "volume_adjust",
           "parameters": {"direction": "increase", "amount": "small", "target_temperature_celsius": None},
           "context_relation": "none", "ambiguous": False, "clarifying_question": None, "reason": "x"}
    raw.update(over)
    return raw


def test_valid_frame_parses():
    f = LLMSemanticParser(MockLLM(good_raw())).parse("Volume thoda badha do", "")
    assert f.intent == Intent.VOLUME_ADJUST and f.parameters.amount.value == "small"


@pytest.mark.parametrize("bad", [
    {"intent": "open_trunk"},                       # not whitelisted
    {"intent": "brake"},
    {"speech_act": "shout"},
    {"parameters": {"direction": "sideways"}},
    {"parameters": {"amount": "huge"}},
    {"parameters": {"shell": "rm -rf /"}},          # unknown parameter is forbidden
    {"parameters": "volume up"},
    {"context_relation": "whatever"},
])
def test_invalid_model_output_rejected(bad):
    with pytest.raises(LLMOutputError):
        LLMSemanticParser(MockLLM(good_raw(**bad))).parse("x", "")


def test_missing_required_field_rejected():
    raw = good_raw()
    del raw["intent"]
    with pytest.raises(LLMOutputError):
        LLMSemanticParser(MockLLM(raw)).parse("x", "")


def test_command_forbids_extra_fields():
    with pytest.raises(ValidationError):
        Command(intent="volume_adjust", parameters={"direction": "increase"}, execute_shell="ls")
    with pytest.raises(ValidationError):
        Parameters(direction="increase", raw_can_frame="0x123")


def test_schema_enum_matches_intent_enum():
    assert set(FRAME_JSON_SCHEMA["properties"]["intent"]["enum"]) == {i.value for i in Intent}


def test_destination_whitespace_normalised():
    assert Parameters(destination="  India   Gate ").destination == "India Gate"


def test_frame_defaults():
    f = SemanticFrame(speech_act="statement", intent="none")
    assert not f.negated and not f.ambiguous and f.context_relation.value == "none"
