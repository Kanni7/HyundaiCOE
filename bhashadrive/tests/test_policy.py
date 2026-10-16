from bhashadrive.policy.command_policy import CommandPolicy
from bhashadrive.schemas import Command, Decision, ResolvedRequest

from .conftest import V, frame

P = CommandPolicy()


def decide(f, cmd=None, verification=None, **res):
    if cmd is None and f.intent.value not in ("none", "unsupported"):
        cmd = Command(intent=f.intent, parameters=f.parameters)
    return P.decide(f, ResolvedRequest(command=cmd, **res), None, verification)


def test_explicit_command_executes():
    f = frame("volume_adjust", direction="increase")
    assert decide(f, verification=V("volume")).decision == Decision.EXECUTE


def test_indirect_request_asks_confirmation():
    f = frame("climate_adjust_temperature", act="indirect_request", direction="increase")
    r = decide(f, verification=V("cabin_temperature", "describes_situation"))
    assert r.decision == Decision.ASK_CONFIRMATION and "warmer" in r.clarifying_question


def test_explicitness_disagreement_downgrades_to_confirmation():
    f = frame("volume_adjust", direction="decrease")
    r = decide(f, verification=V("volume", "describes_situation"))
    assert r.decision == Decision.ASK_CONFIRMATION


def test_function_disagreement_rejects():
    f = frame("climate_adjust_temperature", direction="increase")
    r = decide(f, verification=V("none"))
    assert r.decision == Decision.CLARIFY_OR_REJECT and r.reason_code == "interpretation_disagreement"


def test_negation_never_executes():
    f = frame("climate_ac_power", negated=True, power="off")
    r = decide(f, verification=V("ac_power"))
    assert r.decision == Decision.CLARIFY_OR_REJECT and r.reason_code == "negated_no_action"


def test_ambiguous_clarifies():
    f = frame("volume_adjust", ambiguous=True, direction="increase")
    assert decide(f).decision == Decision.CLARIFY_OR_REJECT


def test_out_of_range_temperature_rejected():
    f = frame("climate_set_temperature", target_temperature_celsius=85)
    r = decide(f, verification=V("cabin_temperature"))
    assert r.decision == Decision.CLARIFY_OR_REJECT and r.reason_code == "invalid_parameter"


def test_missing_parameters_clarify():
    for f in (frame("climate_set_temperature"), frame("fan_adjust_speed"), frame("navigation_search"),
              frame("climate_ac_power")):
        r = decide(f)
        assert r.decision == Decision.CLARIFY_OR_REJECT and r.reason_code == "missing_parameter"


def test_unsafe_destination_rejected():
    f = frame("navigation_search", destination="home; rm -rf /")
    assert decide(f).reason_code == "invalid_parameter"


def test_unsupported_rejected():
    f = frame("unsupported", act="direct_command")
    assert decide(f).reason_code == "unsupported_request"


def test_request_without_whitelisted_intent_is_unsupported():
    # e.g. "Brake laga do" interpreted as intent=none but a direct command
    assert decide(frame("none", act="direct_command")).reason_code == "unsupported_request"


def test_model_error_rejects():
    r = P.decide(None, None, "LLMTimeoutError: timed out")
    assert r.decision == Decision.CLARIFY_OR_REJECT and r.reason_code == "model_error"


def test_missing_context_clarifies():
    r = P.decide(frame("none", act="follow_up", rel="repeat_last"), ResolvedRequest(context_action="missing_context"))
    assert r.reason_code == "missing_context"


def test_confirmed_by_user_executes_even_if_original_was_implicit():
    f = frame("none", act="follow_up", rel="confirm_pending")
    cmd = Command(intent="climate_adjust_temperature", parameters={"direction": "increase"})
    r = P.decide(f, ResolvedRequest(command=cmd, confirmed_by_user=True, context_action="used_pending"))
    assert r.decision == Decision.EXECUTE
