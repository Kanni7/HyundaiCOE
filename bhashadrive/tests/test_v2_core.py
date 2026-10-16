"""V2 schema, slot extraction, policy and pipeline tests (fake model, no real weights needed)."""
import numpy as np
import pytest
from pydantic import ValidationError

from bhashadrive.audio.asr import MockASR
from bhashadrive.config import VehicleLimits
from bhashadrive.nlu.slots import (extract_amount, extract_destination, extract_temperature, negation_cue,
                                   quoted_reference)
from bhashadrive.nlu.taxonomy import IntentV2
from bhashadrive.policy.policy_v2 import PolicyV2
from bhashadrive.schemas_v2 import ActionParameters, DecisionV2, IntentPrediction, ResultV2, VehicleAction
from bhashadrive.vehicle.base import dispatch_v2
from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

from .fakes_v2 import THRESHOLDS, FakeIntentModel, make_pipeline


# ---- schema -----------------------------------------------------------------------
def test_vehicle_action_requires_parameters():
    with pytest.raises(ValidationError):
        VehicleAction(intent="climate_temperature_set")
    with pytest.raises(ValidationError):
        VehicleAction(intent="navigation_destination_request")
    a = VehicleAction(intent="climate_temperature_set", parameters={"target_temperature_celsius": 22})
    assert a.parameters.target_temperature_celsius == 22


def test_vehicle_action_rejects_extra_and_non_executable():
    with pytest.raises(ValidationError):
        VehicleAction(intent="media_pause", parameters={"target_temperature_celsius": 22})
    for bad in ("unsupported", "ambiguous", "dialog_confirm"):
        with pytest.raises(ValidationError):
            VehicleAction(intent=bad)
    with pytest.raises(ValidationError):
        VehicleAction(intent="not_an_intent")
    with pytest.raises(ValidationError):
        ActionParameters(unknown=1)


def test_temperature_direction_fixed_by_intent():
    assert VehicleAction(intent="climate_temperature_up").parameters.direction == "warmer"
    with pytest.raises(ValidationError):
        VehicleAction(intent="climate_temperature_down", parameters={"direction": "warmer"})


def test_result_schema_serialises():
    r = ResultV2(transcript="x", decision="reject", reason_code="unsupported_request")
    assert r.model_dump(mode="json")["decision"] == "reject"


# ---- slots --------------------------------------------------------------------------
@pytest.mark.parametrize("text,value", [
    ("Set the temperature to 22", 22), ("AC baais pe kar do", 22), ("तापमान बाईस डिग्री करो", 22),
    ("२४ डिग्री कर दो", 24), ("twenty five please", 25), ("hold at 19.5", 19.5), ("make it warmer", None),
    ("set 2 zones to 22 degrees", 22),
])
def test_extract_temperature(text, value):
    assert extract_temperature(text) == value


def test_extract_temperature_ambiguous_numbers_returns_none():
    assert extract_temperature("22 or 24") is None


@pytest.mark.parametrize("text,dest", [
    ("Take me to Connaught Place.", "Connaught Place"), ("India Gate ka raasta dikhao", "India Gate"),
    ("मुझे एयरपोर्ट ले चलो", "एयरपोर्ट"), ("Find the nearest charging station", "charging station"),
    ("Mujhe Saket jaana hai", "Saket"),
])
def test_extract_destination(text, dest):
    assert extract_destination(text) == dest


def test_extract_amount():
    assert extract_amount("thoda warm kar do") == "small"
    assert extract_amount("blast it, full speed") == "large"
    assert extract_amount("turn it up") is None


@pytest.mark.parametrize("text,neg", [
    ("Don't turn the AC off", True), ("AC band mat karna", True), ("एसी बंद मत करो", True),
    ("never switch off the climate", True), ("stop the music", False), ("Not the temperature, reduce the fan", False),
    ("I'm freezing", False), ("AC band karne ki zaroorat nahi", True),
])
def test_negation_cues(text, neg):
    assert bool(negation_cue(text)) is neg


def test_quoted_reference():
    assert quoted_reference("what does 'chill mode' do?")
    assert quoted_reference("delete chill mode")
    assert not quoted_reference("Chill mode, yaar")
    assert not quoted_reference("I'm cold, it's freezing")


# ---- policy --------------------------------------------------------------------------
def pred(intent, score=0.95, margin=0.9, neg=0.0, cue=None, imp=0.0):
    return IntentPrediction(intent=intent, score=score, margin=margin, top_k=[(intent, score), ("volume_down", 0.03)],
                            negation_score=neg, negation_cue=cue, implicit_score=imp)


@pytest.fixture
def policy():
    return PolicyV2(VehicleLimits(), THRESHOLDS)


def test_policy_explicit_executes(policy):
    out = policy.decide("volume badha do", pred("volume_up"))
    assert out.decision == DecisionV2.EXECUTE and out.action.intent == IntentV2.VOLUME_UP


def test_policy_implicit_asks(policy):
    out = policy.decide("I'm freezing", pred("climate_temperature_up", imp=0.9))
    assert out.decision == DecisionV2.ASK_CONFIRMATION and out.action.parameters.direction == "warmer"


def test_policy_negation_never_executes(policy):
    assert policy.decide("Don't turn the AC off", pred("climate_off", cue="Don't")).reason_code == "negated_no_action"
    assert policy.decide("x", pred("climate_off", neg=0.8)).decision == DecisionV2.REJECT


def test_policy_unsupported_and_ambiguous(policy):
    assert policy.decide("open the sunroof", pred("unsupported")).decision == DecisionV2.REJECT
    assert policy.decide("change it", pred("ambiguous")).decision == DecisionV2.CLARIFY
    assert policy.decide("yes", pred("dialog_confirm")).reason_code == "missing_context"


def test_policy_low_confidence_clarifies(policy):
    out = policy.decide("fan", pred("climate_fan_speed_up", score=0.5, margin=0.1))
    assert out.decision == DecisionV2.CLARIFY and out.reason_code == "low_confidence"


@pytest.mark.parametrize("text,decision,code", [
    ("set the temperature to 35", DecisionV2.REJECT, "invalid_parameter"),
    ("set the temperature to 12 degrees", DecisionV2.REJECT, "invalid_parameter"),
    ("set the temperature", DecisionV2.CLARIFY, "missing_parameter"),
    ("set the temperature to 30", DecisionV2.EXECUTE, "explicit_command"),
    ("set the temperature to 16", DecisionV2.EXECUTE, "explicit_command"),
])
def test_policy_numeric_bounds(policy, text, decision, code):
    out = policy.decide(text, pred("climate_temperature_set"))
    assert (out.decision, out.reason_code) == (decision, code)


def test_policy_rejects_unsafe_destination(policy):
    out = policy.decide("navigate to $(rm -rf /)", pred("navigation_destination_request"))
    assert out.decision == DecisionV2.REJECT


# ---- vehicle -------------------------------------------------------------------------
def test_dispatch_v2_covers_every_executable_intent():
    from bhashadrive.nlu.taxonomy import EXECUTABLE_INTENTS

    v, lim = MockVehicleAdapter(), VehicleLimits()
    for it in EXECUTABLE_INTENTS:
        params = {"target_temperature_celsius": 21} if it == IntentV2.CLIMATE_TEMPERATURE_SET else (
            {"destination": "India Gate"} if it == IntentV2.NAVIGATION_DESTINATION_REQUEST else {})
        r = dispatch_v2(v, VehicleAction(intent=it, parameters=params), lim)
        assert r.message != "Action not in allowlist"
    assert v.get_state()["navigation_request"]["query"] == "India Gate"


def test_dispatch_v2_respects_limits():
    v = MockVehicleAdapter()
    v.state["volume"] = 30
    r = dispatch_v2(v, VehicleAction(intent="volume_up"), VehicleLimits())
    assert not r.success and v.state["volume"] == 30


# ---- pipeline (built-in commands, context) ---------------------------------------------
def test_pipeline_explicit_execution_changes_state():
    p = make_pipeline()
    r = p.process_text("volume badha do")
    assert r.decision == DecisionV2.EXECUTE and r.action_result.success and p.vehicle.state["volume"] == 12


def test_pipeline_implicit_then_yes_executes():
    p = make_pipeline()
    r = p.process_text("I'm freezing")
    assert r.decision == DecisionV2.ASK_CONFIRMATION and p.vehicle.state["cabin_temperature_c"] == 22
    r = p.process_text("yes")
    assert r.decision == DecisionV2.EXECUTE and r.reason_code == "confirmed_by_user"
    assert p.vehicle.state["cabin_temperature_c"] == 23


def test_pipeline_implicit_then_no_cancels():
    p = make_pipeline()
    p.process_text("I'm freezing")
    assert p.process_text("no").reason_code == "cancelled_by_user"
    assert p.process_text("yes").reason_code == "missing_context"
    assert p.vehicle.state["cabin_temperature_c"] == 22


def test_pipeline_pending_expires():
    p = make_pipeline()
    t = [0.0]
    p.state._clock = lambda: t[0]
    p.process_text("I'm freezing")
    t[0] = 1000.0
    assert p.process_text("yes").reason_code == "missing_context"


def test_pipeline_negation_and_empty():
    p = make_pipeline()
    assert p.process_text("Don't turn the AC off").decision == DecisionV2.REJECT
    assert p.vehicle.state["ac_on"] is True
    assert p.process_text("   ").reason_code == "empty_transcript"


def test_pipeline_asr_failures_do_not_act():
    p = make_pipeline()
    for status in ("silence", "no_audio", "error"):
        p.asr = MockASR("", status=status)
        r = p.process_audio_array(np.zeros(16000, np.float32))
        assert r.decision == DecisionV2.CLARIFY and r.action is None
    assert p.vehicle.log == []


def test_pipeline_audio_path_runs_nlu():
    p = make_pipeline()
    p.asr = MockASR("pause the music")
    r = p.process_audio_file("whatever.wav")
    assert r.decision == DecisionV2.EXECUTE and r.asr.text == "pause the music" and "end_to_end_s" in r.timings_s
