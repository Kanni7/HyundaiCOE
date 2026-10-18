"""Deterministic decision policy for the V2 runtime.

The trained model proposes an intent with a calibrated score; this layer decides whether
anything happens. Order: out-of-scope -> ambiguity -> negation -> confidence -> parameter
extraction/validation -> explicit vs implicit. When in doubt it asks or declines.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from ..config import VehicleLimits
from ..nlu.slots import extract_amount, extract_destination, extract_temperature
from ..nlu.taxonomy import DESCRIPTIONS, DIALOG_INTENTS, EXECUTABLE_INTENTS, PARAM_SCHEMA, IntentV2
from ..schemas_v2 import ActionParameters, DecisionV2, IntentPrediction, SpeechActV2, VehicleAction

_UNSAFE_DEST = re.compile(r"[<>{}`$;|\\\x00-\x1f]")


class ActionValidationError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def validate_action(action: VehicleAction, limits: VehicleLimits) -> VehicleAction:
    """Numeric bounds and text safety on top of the schema check done by VehicleAction itself."""
    p = action.parameters
    if action.intent == IntentV2.CLIMATE_TEMPERATURE_SET:
        t = p.target_temperature_celsius
        if not (limits.temp_min_c <= t <= limits.temp_max_c):
            raise ActionValidationError(
                "invalid_parameter", f"{t:g}°C is outside the allowed range {limits.temp_min_c:g}-{limits.temp_max_c:g}°C")
    if action.intent == IntentV2.NAVIGATION_DESTINATION_REQUEST:
        if len(p.destination) > limits.max_destination_chars or _UNSAFE_DEST.search(p.destination):
            raise ActionValidationError("invalid_parameter", "destination text is not acceptable")
    return action


def build_action(intent: IntentV2, text: str) -> VehicleAction:
    """Fill the intent's parameter slots from the transcript; raise ActionValidationError if incomplete."""
    schema = PARAM_SCHEMA[intent]
    params = {}
    if "target_temperature_celsius" in schema:
        params["target_temperature_celsius"] = extract_temperature(text)
    if "destination" in schema:
        params["destination"] = extract_destination(text)
    if "amount" in schema:
        params["amount"] = extract_amount(text)
    try:
        return VehicleAction(intent=intent, parameters=ActionParameters(**params))
    except ValidationError as exc:
        missing = [k for k, req in schema.items() if req and params.get(k) is None and k != "direction"]
        what = {"target_temperature_celsius": "which temperature", "destination": "where to go"}
        if missing:
            raise ActionValidationError("missing_parameter", " and ".join(what.get(m, m) for m in missing)) from exc
        raise ActionValidationError("invalid_parameter", str(exc.errors()[0]["msg"])) from exc


@dataclass
class PolicyOutcomeV2:
    decision: DecisionV2
    reason_code: str
    speech_act: SpeechActV2 = SpeechActV2.OTHER
    action: VehicleAction | None = None
    question: str | None = None
    negated: bool = False
    params: dict = field(default_factory=dict)


class PolicyV2:
    def __init__(self, limits: VehicleLimits, thresholds: dict, negation_threshold: float = 0.5,
                 implicit_threshold: float = 0.5):
        self.limits, self.th = limits, thresholds
        self.neg_th, self.imp_th = negation_threshold, implicit_threshold

    def is_negated(self, pred: IntentPrediction) -> bool:
        return bool(pred.negation_cue) or (pred.negation_score or 0.0) >= self.neg_th

    def decide(self, text: str, pred: IntentPrediction) -> PolicyOutcomeV2:
        it = pred.intent
        if it == IntentV2.UNSUPPORTED:
            return PolicyOutcomeV2(DecisionV2.REJECT, "unsupported_request")
        if it == IntentV2.AMBIGUOUS:
            if re.search(r"\b(temperature|taapmaan|climate|cabin|garmi|thand|cold|hot|warm)\b", text, re.I):
                q = "Did you want to set the temperature, or make the cabin warmer or cooler?"
            elif re.search(r"\b(fan|blower|airflow|pankh\w*|hawa)\b", text, re.I):
                q = "Did you want to increase or decrease the fan speed?"
            elif re.search(r"\b(volume|awaaz|aawaz|awaz|sound|audio)\b", text, re.I):
                q = "Did you want to turn the volume up or down?"
            elif re.search(r"\b(music|song|track|gaana|gana)\b", text, re.I):
                q = "Did you want to play, pause, or change the song?"
            elif re.search(r"\b(ac|air\s*con\w*)\b", text, re.I):
                q = "Did you want to turn the AC on or off, or change the temperature?"
            else:
                q = "Which setting should I change: temperature, fan, volume or media?"
            return PolicyOutcomeV2(DecisionV2.CLARIFY, "ambiguous", question=q)
        if it in DIALOG_INTENTS:
            # Without a pending proposal a bare yes/no has nothing to act on.
            return PolicyOutcomeV2(DecisionV2.CLARIFY, "missing_context", SpeechActV2.DIALOG,
                                   question="What would you like me to do?")
        assert it in EXECUTABLE_INTENTS
        if self.is_negated(pred):
            return PolicyOutcomeV2(DecisionV2.REJECT, "negated_no_action", SpeechActV2.DIRECT_COMMAND, negated=True)
        if pred.score < self.th["min_score"] or pred.margin < self.th["min_margin"]:
            alts = [IntentV2(i) for i, _ in pred.top_k[:2] if IntentV2(i) in EXECUTABLE_INTENTS]
            q = (f"Did you want me to {DESCRIPTIONS[alts[0]]} or {DESCRIPTIONS[alts[1]]}?" if len(alts) == 2
                 else f"Did you want me to {DESCRIPTIONS[it]}?")
            return PolicyOutcomeV2(DecisionV2.CLARIFY, "low_confidence", question=q)
        try:
            action = validate_action(build_action(it, text), self.limits)
        except ActionValidationError as e:
            if e.code == "missing_parameter":
                if "temperature" in e.message:
                    msg = "What temperature would you like me to set (e.g. 22 degrees), or make it warmer or cooler?"
                else:
                    msg = f"I need a bit more detail: {e.message}?"
                return PolicyOutcomeV2(DecisionV2.CLARIFY, e.code, SpeechActV2.DIRECT_COMMAND, question=msg)
            return PolicyOutcomeV2(DecisionV2.REJECT, e.code, SpeechActV2.DIRECT_COMMAND, question=f"I can't do that: {e.message}.")
        params = action.parameters.model_dump(exclude_none=True)
        if (pred.implicit_score or 0.0) >= self.imp_th:
            return PolicyOutcomeV2(DecisionV2.ASK_CONFIRMATION, "implicit_request", SpeechActV2.IMPLICIT_REQUEST, action,

                                   question=f"Would you like me to {action.describe()}?", params=params)
        return PolicyOutcomeV2(DecisionV2.EXECUTE, "explicit_command", SpeechActV2.DIRECT_COMMAND, action, params=params)
