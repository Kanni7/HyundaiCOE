"""Typed, validated data contracts shared by every pipeline stage.

The language model's output is parsed into `SemanticFrame`. Anything that does
not validate is rejected; the model can never produce an arbitrary command.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Intent(str, Enum):
    """Whitelisted vehicle intents. Nothing outside this list can be executed."""

    CLIMATE_ADJUST_TEMPERATURE = "climate_adjust_temperature"
    CLIMATE_SET_TEMPERATURE = "climate_set_temperature"
    CLIMATE_AC_POWER = "climate_ac_power"
    FAN_ADJUST_SPEED = "fan_adjust_speed"
    VOLUME_ADJUST = "volume_adjust"
    MEDIA_PLAY = "media_play"
    MEDIA_PAUSE = "media_pause"
    MEDIA_NEXT_TRACK = "media_next_track"
    MEDIA_PREVIOUS_TRACK = "media_previous_track"
    NAVIGATION_SEARCH = "navigation_search"
    NONE = "none"  # pure conversational follow-up (yes/no/cancel) or nothing to do
    UNSUPPORTED = "unsupported"  # out of domain or not permitted


ACTIONABLE_INTENTS = {i for i in Intent if i not in (Intent.NONE, Intent.UNSUPPORTED)}
ADJUSTABLE_INTENTS = {Intent.CLIMATE_ADJUST_TEMPERATURE, Intent.FAN_ADJUST_SPEED, Intent.VOLUME_ADJUST}


class SpeechAct(str, Enum):
    DIRECT_COMMAND = "direct_command"
    INDIRECT_REQUEST = "indirect_request"
    STATEMENT = "statement"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"
    UNSUPPORTED_REQUEST = "unsupported_request"


class ContextRelation(str, Enum):
    """How an utterance relates to the conversation state."""

    NONE = "none"  # stands on its own
    CONFIRM_PENDING = "confirm_pending"  # "yes", "haan kar do", "yes, just a little"
    REJECT_PENDING = "reject_pending"  # "no", "nahi rehne do"
    CANCEL = "cancel"  # "cancel that"
    REPEAT_LAST = "repeat_last"  # "a bit more", "aur thoda"
    REVERSE_LAST = "reverse_last"  # "that's too much", "zyada ho gaya"
    CORRECT_PREVIOUS = "correct_previous"  # "not the temperature, reduce the fan"


class Decision(str, Enum):
    EXECUTE = "execute"
    ASK_CONFIRMATION = "ask_confirmation"
    CLARIFY_OR_REJECT = "clarify_or_reject"


class Direction(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


class Amount(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


LanguagePattern = Literal["english", "hindi", "hinglish", "romanized_hindi", "other", "unknown"]


class Parameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: Direction | None = None  # for temperature: increase == warmer
    amount: Amount | None = None
    target_temperature_celsius: float | None = None
    power: Literal["on", "off"] | None = None
    destination: str | None = Field(default=None, max_length=200)

    @field_validator("destination")
    @classmethod
    def _strip_destination(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = " ".join(v.split())
        return v or None


class SemanticFrame(BaseModel):
    """Structured interpretation produced by the semantic parser (LLM output)."""

    model_config = ConfigDict(extra="ignore")

    meaning: str = Field(default="", max_length=400)  # model's English paraphrase (reasoning aid, never executed)
    language_pattern: LanguagePattern = "unknown"
    user_goal: str = Field(default="", max_length=300)
    speech_act: SpeechAct
    intent: Intent
    parameters: Parameters = Field(default_factory=Parameters)
    negated: bool = False  # user asked NOT to do the action
    context_relation: ContextRelation = ContextRelation.NONE
    ambiguous: bool = False
    clarifying_question: str | None = Field(default=None, max_length=300)
    reason: str = Field(default="", max_length=300)


class Verification(BaseModel):
    """Independent second-opinion classification used by the policy as an agreement check."""

    function: Literal["cabin_temperature", "ac_power", "fan_speed", "volume", "media_playback", "navigation",
                      "none", "unsupported"]
    request_type: Literal["explicit_request", "describes_situation", "other"] = "other"


class Command(BaseModel):
    """A concrete, fully-parameterised vehicle request (not yet authorised)."""

    model_config = ConfigDict(extra="forbid")

    intent: Intent
    parameters: Parameters = Field(default_factory=Parameters)

    def describe(self) -> str:
        p = self.parameters
        amt = f" a {p.amount.value} amount" if p.amount else ""
        match self.intent:
            case Intent.CLIMATE_ADJUST_TEMPERATURE:
                return f"make the cabin {'warmer' if p.direction == Direction.INCREASE else 'cooler'}{' by' + amt if amt else ''}"
            case Intent.CLIMATE_SET_TEMPERATURE:
                return f"set the cabin temperature to {p.target_temperature_celsius:g}°C"
            case Intent.CLIMATE_AC_POWER:
                return f"turn the AC {p.power}"
            case Intent.FAN_ADJUST_SPEED:
                return f"{p.direction.value if p.direction else 'change'} the fan speed{' by' + amt if amt else ''}"
            case Intent.VOLUME_ADJUST:
                return f"{p.direction.value if p.direction else 'change'} the volume{' by' + amt if amt else ''}"
            case Intent.MEDIA_PLAY:
                return "play media"
            case Intent.MEDIA_PAUSE:
                return "pause media"
            case Intent.MEDIA_NEXT_TRACK:
                return "skip to the next track"
            case Intent.MEDIA_PREVIOUS_TRACK:
                return "go back to the previous track"
            case Intent.NAVIGATION_SEARCH:
                return f"search navigation for '{p.destination}'"
        return self.intent.value


class ResolvedRequest(BaseModel):
    """Output of the context resolver, input to the policy layer."""

    command: Command | None = None
    confirmed_by_user: bool = False  # user explicitly confirmed a pending proposal
    context_action: Literal["none", "used_pending", "used_last", "cleared_pending", "missing_context"] = "none"
    note: str = ""


class PolicyResult(BaseModel):
    decision: Decision
    reason_code: str
    command: Command | None = None
    clarifying_question: str | None = None


class ActionResult(BaseModel):
    success: bool
    intent: Intent
    message: str
    state: dict[str, Any] = Field(default_factory=dict)


class ASRResult(BaseModel):
    text: str
    raw_text: str | None = None  # model output before transliteration/normalisation
    language: str | None = None
    language_probability: float | None = None
    raw_detected_language: str | None = None  # before restricting to allowed languages
    duration_s: float | None = None
    segments: list[dict[str, Any]] = Field(default_factory=list)
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    status: Literal["ok", "silence", "no_audio", "error"] = "ok"
    error: str | None = None
    latency_s: float | None = None


class PipelineResult(BaseModel):
    transcript: str
    normalized_transcript: str
    asr: ASRResult | None = None
    frame: SemanticFrame | None = None
    nlu_error: str | None = None
    verification: Verification | None = None
    resolved: ResolvedRequest | None = None
    policy: PolicyResult
    action: ActionResult | None = None
    response_text: str
    timings_s: dict[str, float] = Field(default_factory=dict)
