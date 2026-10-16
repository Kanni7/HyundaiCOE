"""Semantic interpretation: transcript + context summary -> validated SemanticFrame.

The local LLM does the understanding. Its output is constrained to a JSON schema
and then validated by Pydantic. Invalid output raises; there is no keyword
fallback.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from pydantic import ValidationError

from ..schemas import SemanticFrame, Verification
from .base import LanguageModel, LLMOutputError

SYSTEM_PROMPT = """You are the language-understanding module of an in-car voice assistant used in India.
Drivers speak English, Hindi (Devanagari), Hindi in Roman script, or a mix (Hinglish).
Infer what the driver MEANS, not which words they used. Output ONE JSON object only.

SUPPORTED intents (anything else is "unsupported"):
- climate_adjust_temperature: make cabin warmer (direction=increase) or cooler (direction=decrease).
  Feeling cold / freezing / thand => warmer. Feeling hot / sweating / garmi => cooler.
- climate_set_temperature: set cabin to a specific number of degrees Celsius (target_temperature_celsius).
- climate_ac_power: switch the AC/climate system on or off (power="on"/"off").
- fan_adjust_speed: blower/fan speed up (increase) or down (decrease).
- volume_adjust: media/audio volume louder (increase) or quieter (decrease).
- media_play, media_pause, media_next_track, media_previous_track.
- navigation_search: find/navigate to a place (destination = the place, copied as spoken).
- none: no vehicle action of its own, e.g. a bare "yes", "no", "cancel" reply.
- unsupported: anything outside this list (driving controls like steering/brakes/acceleration/speed, windows, doors,
  calls, general chit-chat, questions about the world, shell/system commands).

Fields (fill them in this order):
- meaning: FIRST, one English sentence saying what the driver means, keeping any negation explicit
  (e.g. "The driver feels cold, which implies the cabin should be warmer" or
  "The driver does NOT want the AC switched off"). Think about Hindi words carefully:
  thand/ठंड/sardi = cold, garmi/गर्मी = hot, mat/मत/nahi/नहीं/na = not, band/बंद = off, chalu/on = on,
  badha/badhao/तेज़ = more, kam/कम/dheema = less, thoda/थोड़ा = a little, agla = next, pichla = previous.
- negated: true when the driver says NOT to do the action ("don't", "mat", "nahi karna").
  If one clause is negated and another clause asks for a different action ("don't X, just do Y"),
  choose Y with negated=false.
  Then intent is the action they do NOT want, e.g. "don't turn off the AC" -> climate_ac_power, power=off, negated=true.
- speech_act: direct_command (an explicit or polite request to do X, incl. "can you ..."),
  indirect_request (describes a feeling/problem implying an action, e.g. discomfort),
  statement (information with no implied request), question (asks for info), follow_up (reply that
  only makes sense with the conversation), unsupported_request.
- amount: small ("thoda", "a bit", "slightly"), medium, large ("bahut", "a lot"), or null.
- context_relation (use the CONVERSATION STATE given in the user message):
  none = stands alone (the DEFAULT); confirm_pending = agrees to the pending proposal (may add an amount, e.g. "yes, slightly");
  reject_pending = declines the pending proposal; cancel = asks to cancel/stop what was proposed;
  repeat_last = wants the same adjustment again ("more", "aur", "again");
  reverse_last = the last change overshot ("too much", "zyada ho gaya") ;
  correct_previous = rejects the previous target and names a different action ("not X, do Y").
  If the utterance is a complete request on its own (names a function and what to do), use "none"
  even when there is earlier state. Use other relations only for replies that depend on the state.
  Short replies like "more" or "yes" with NO matching state still get their relation; never invent state.
  For confirm_pending/reject_pending/cancel set intent "none" unless a new action is named.
  For repeat_last/reverse_last copy the intent of the last action if there is one, else "none".
- If a new request does not say or clearly imply WHICH function (e.g. "make it higher"), set intent "none"
  and ambiguous=true. Never guess a function.
- ambiguous: true if you cannot tell which function or direction is meant (e.g. "change it", "adjust that"
  with no state, or a request that could mean several functions).
- clarifying_question: a short question to the driver if ambiguous, else null.
- language_pattern: english | hindi | hinglish | romanized_hindi | other.
- user_goal: one short English sentence describing what the driver wants.
- reason: a few words justifying the interpretation.

Examples (illustrative; generalise, do not memorise):
"crank up the music a lot" -> {"meaning":"The driver wants the music much louder","negated":false,"speech_act":"direct_command","intent":"volume_adjust","parameters":{"direction":"increase","amount":"large"},"context_relation":"none","ambiguous":false}
"mujhe pasina aa raha hai" -> {"meaning":"The driver is sweating, so the cabin should be cooler","negated":false,"speech_act":"indirect_request","intent":"climate_adjust_temperature","parameters":{"direction":"decrease"},"context_relation":"none","ambiguous":false}
"gaana mat rokna" -> {"meaning":"The driver does NOT want the music paused","negated":true,"speech_act":"direct_command","intent":"media_pause","parameters":{},"context_relation":"none","ambiguous":false}
"what is the capital of France" -> {"meaning":"A general-knowledge question, not a car function","negated":false,"speech_act":"question","intent":"unsupported","parameters":{},"context_relation":"none","ambiguous":false}
"""

# JSON schema sent to Ollama for constrained decoding (mirrors schemas.SemanticFrame).
FRAME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "meaning": {"type": "string"},
        "negated": {"type": "boolean"},
        "language_pattern": {"type": "string", "enum": ["english", "hindi", "hinglish", "romanized_hindi", "other"]},
        "user_goal": {"type": "string"},
        "speech_act": {"type": "string", "enum": ["direct_command", "indirect_request", "statement", "question", "follow_up", "unsupported_request"]},
        "intent": {"type": "string", "enum": [
            "climate_adjust_temperature", "climate_set_temperature", "climate_ac_power", "fan_adjust_speed",
            "volume_adjust", "media_play", "media_pause", "media_next_track", "media_previous_track",
            "navigation_search", "none", "unsupported"]},
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {"type": ["string", "null"], "enum": ["increase", "decrease", None]},
                "amount": {"type": ["string", "null"], "enum": ["small", "medium", "large", None]},
                "target_temperature_celsius": {"type": ["number", "null"]},
                "power": {"type": ["string", "null"], "enum": ["on", "off", None]},
                "destination": {"type": ["string", "null"]},
            },
        },
        "context_relation": {"type": "string", "enum": [
            "none", "confirm_pending", "reject_pending", "cancel", "repeat_last", "reverse_last", "correct_previous"]},
        "ambiguous": {"type": "boolean"},
        "clarifying_question": {"type": ["string", "null"]},
        "reason": {"type": "string"},
    },
    "required": ["meaning", "negated", "language_pattern", "user_goal", "speech_act", "intent", "parameters",
                 "context_relation", "ambiguous", "clarifying_question", "reason"],
}


def build_user_message(transcript: str, context_summary: str) -> str:
    return f"CONVERSATION STATE:\n{context_summary}\n\nDRIVER SAID: {transcript}\n\nReturn the JSON object."


def clean_parameters(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop keys the model left null/empty; keep everything else for strict validation."""
    params = raw.get("parameters") or {}
    if not isinstance(params, dict):
        raise LLMOutputError("parameters must be an object")
    return {k: v for k, v in params.items() if v not in (None, "", "null")}


VERIFIER_PROMPT = """You double-check an in-car voice assistant. Drivers speak English, Hindi or Hinglish.
Read the driver's sentence BY ITSELF (no earlier conversation) and fill two fields.

function - the car function the sentence is about:
  cabin_temperature = heating/cooling the cabin, or the driver feeling hot, cold, sweaty or shivering
  ac_power = switching the AC on or off
  fan_speed = blower or air-flow strength
  volume = how loud the audio is, or not being able to hear it
  media_playback = play, pause, stop, skip, next or previous song/track
  navigation = finding or going to a place
  unsupported = another car or phone feature (windows, sunroof, brakes, steering, speed, airbags, calls, weather)
  none = ONLY if the sentence names no function and implies none (e.g. "change it", "a bit more", "yes"),
         or is unrelated to the car (hunger, small talk)
request_type - explicit_request if the driver tells/asks the assistant to do something (an imperative in any
  language, "please", "can you", "kar do", "karo", "dikhao", "lagao"); describes_situation if the driver only
  describes a feeling or situation; other for anything else.
Return JSON {"function": "...", "request_type": "..."}."""

VERIFIER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"function": {"type": "string", "enum": [
        "cabin_temperature", "ac_power", "fan_speed", "volume", "media_playback", "navigation", "none", "unsupported"]},
        "request_type": {"type": "string", "enum": ["explicit_request", "describes_situation", "other"]}},
    "required": ["function", "request_type"],
}

# Maps the verifier's coarse function to the parser intents it is consistent with.
VERIFIER_FUNCTIONS: dict[str, set[str]] = {
    "cabin_temperature": {"climate_adjust_temperature", "climate_set_temperature"},
    "ac_power": {"climate_ac_power", "climate_adjust_temperature"},
    "fan_speed": {"fan_adjust_speed"},
    "volume": {"volume_adjust"},
    "media_playback": {"media_play", "media_pause", "media_next_track", "media_previous_track"},
    "navigation": {"navigation_search"},
    "none": set(),
    "unsupported": set(),
}


class SemanticParser(ABC):
    name = "abstract"

    @abstractmethod
    def parse(self, transcript: str, context_summary: str) -> SemanticFrame: ...

    def verify(self, transcript: str) -> Verification | None:
        """Independent second opinion on a standalone utterance (None = this parser has no verifier)."""
        return None


class LLMSemanticParser(SemanticParser):
    def __init__(self, llm: LanguageModel):
        self.llm = llm
        self.name = f"llm:{llm.name}"
        self.last_latency_s: float | None = None

    def parse(self, transcript: str, context_summary: str) -> SemanticFrame:
        t0 = time.perf_counter()
        raw = self.llm.generate_json(SYSTEM_PROMPT, build_user_message(transcript, context_summary), FRAME_JSON_SCHEMA)
        self.last_latency_s = time.perf_counter() - t0
        raw = dict(raw)
        raw["parameters"] = clean_parameters(raw)
        try:
            return SemanticFrame.model_validate(raw)
        except ValidationError as exc:
            raise LLMOutputError(f"Model output failed schema validation: {exc.errors()[:3]}") from exc

    def verify(self, transcript: str) -> Verification:
        raw = self.llm.generate_json(VERIFIER_PROMPT, f"Driver said: {transcript}", VERIFIER_SCHEMA)
        try:
            return Verification.model_validate(raw)
        except ValidationError as exc:
            raise LLMOutputError(f"Verifier output failed validation: {exc.errors()[:2]}") from exc


class StaticSemanticParser(SemanticParser):
    """Returns pre-built frames; used by unit tests to isolate context/policy logic."""

    name = "static"

    def __init__(self, frames: dict[str, SemanticFrame | Exception]):
        self.frames = frames

    def parse(self, transcript: str, context_summary: str) -> SemanticFrame:
        item = self.frames[transcript]
        if isinstance(item, Exception):
            raise item
        return item
