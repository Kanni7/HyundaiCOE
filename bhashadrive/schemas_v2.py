"""Validated data contracts for the V2 (compact semantic model) runtime.

`VehicleAction` is the only object that can reach the vehicle adapter. Its parameters are
checked against the per-intent schema in `nlu.taxonomy`; numeric bounds are checked by
`policy.validation_v2` against the configured vehicle limits.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .nlu.taxonomy import DESCRIPTIONS, EXECUTABLE_INTENTS, FIXED_DIRECTION, PARAM_SCHEMA, IntentV2
from .schemas import ASRResult, ActionResult


class SpeechActV2(str, Enum):
    DIRECT_COMMAND = "direct_command"
    IMPLICIT_REQUEST = "implicit_request"  # "I'm freezing" -> confirm before acting
    DIALOG = "dialog"  # yes / no / cancel
    TEACH = "teach"  # "when I say X, do Y"
    MANAGE = "manage"  # delete / rename / correct a personal command
    OTHER = "other"


class DecisionV2(str, Enum):
    EXECUTE = "execute"
    ASK_CONFIRMATION = "ask_confirmation"
    CLARIFY = "clarify"
    REJECT = "reject"
    SAVED = "saved"  # a personalization change was committed (no vehicle action)


class ActionParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Literal["warmer", "cooler"] | None = None
    amount: Literal["small", "medium", "large"] | None = None
    target_temperature_celsius: float | None = None
    destination: str | None = Field(default=None, max_length=200)

    @field_validator("destination")
    @classmethod
    def _clean_destination(cls, v: str | None) -> str | None:
        v = " ".join((v or "").split())
        return v or None


class VehicleAction(BaseModel):
    """A fully specified, schema-checked request for one whitelisted vehicle function."""

    model_config = ConfigDict(extra="forbid")

    intent: IntentV2
    parameters: ActionParameters = Field(default_factory=ActionParameters)

    @model_validator(mode="after")
    def _check_schema(self) -> "VehicleAction":
        if self.intent not in EXECUTABLE_INTENTS:
            raise ValueError(f"{self.intent.value} is not an executable vehicle action")
        schema = PARAM_SCHEMA[self.intent]
        if self.intent in FIXED_DIRECTION:
            if self.parameters.direction not in (None, FIXED_DIRECTION[self.intent]):
                raise ValueError("direction contradicts the intent")
            self.parameters.direction = FIXED_DIRECTION[self.intent]
        given = {k for k, v in self.parameters.model_dump().items() if v is not None}
        extra = given - set(schema)
        if extra:
            raise ValueError(f"parameters {sorted(extra)} are not allowed for {self.intent.value}")
        missing = [k for k, req in schema.items() if req and k not in given]
        if missing:
            raise ValueError(f"missing required parameters {missing} for {self.intent.value}")
        return self

    def describe(self) -> str:
        p = self.parameters
        if self.intent == IntentV2.CLIMATE_TEMPERATURE_SET:
            return f"set the cabin temperature to {p.target_temperature_celsius:g}°C"
        if self.intent == IntentV2.NAVIGATION_DESTINATION_REQUEST:
            return f"search navigation for '{p.destination}'"
        amt = f" ({p.amount} step)" if p.amount else ""
        return DESCRIPTIONS[self.intent] + amt


class IntentPrediction(BaseModel):
    """Raw output of the trained classifier heads for one utterance."""

    intent: IntentV2
    score: float  # temperature-scaled softmax probability of `intent` (see reports/nlu_eval.json)
    margin: float  # score(top1) - score(top2)
    top_k: list[tuple[str, float]] = Field(default_factory=list)
    negation_score: float | None = None  # P(negated) from the learned negation head
    negation_cue: str | None = None  # matched lexical negation cue, if any
    implicit_score: float | None = None  # P(implicit request) from the speech-act head
    model_version: str = ""


class PersonalMatch(BaseModel):
    command_id: int
    trigger: str
    matched_phrase: str
    similarity: float
    margin: float  # best - second-best (different command); 1.0 if only one command
    action: VehicleAction
    accepted: bool
    reason: str = ""


class ResultV2(BaseModel):
    """Structured end-to-end result of one turn."""

    transcript: str
    normalized_transcript: str = ""
    intent: IntentV2 | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    speech_act: SpeechActV2 = SpeechActV2.OTHER
    decision: DecisionV2
    confidence: float | None = None
    confidence_meaning: str | None = None
    reason_code: str
    negated: bool = False
    response_text: str = ""
    source: Literal["builtin", "personal", "context", "personalization", "none"] = "none"
    prediction: IntentPrediction | None = None
    personal_match: PersonalMatch | None = None
    action: VehicleAction | None = None
    action_result: ActionResult | None = None
    asr: ASRResult | None = None
    timings_s: dict[str, float] = Field(default_factory=dict)
