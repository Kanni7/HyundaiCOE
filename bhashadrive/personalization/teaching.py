"""Recognise teach / manage requests and turn them into *proposals* that need confirmation.

Deterministic parsing is limited to finding clearly formatted boundaries (the trigger phrase
and the action clause). The action clause itself is interpreted by the trained intent model
and goes through the same parameter extraction and validation as a normal command.
Nothing is saved here; the pipeline saves only after the user confirms.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..nlu.taxonomy import DESCRIPTIONS, EXECUTABLE_INTENTS, IntentV2
from ..policy.policy_v2 import ActionValidationError, build_action, validate_action
from ..schemas_v2 import VehicleAction

_Q = "\"'“”‘’"
_TRIG = rf"[{_Q}]?(?P<trig>[^{_Q}]+?)[{_Q}]?"
_TEACH = [
    # English: When I say 'X', <action>   |  If I say X then <action>
    re.compile(rf"^\s*(?:when(?:ever)?|if)\s+i\s+say\s+{_TRIG}\s*(?:[,;:]\s*|\s+then\s+|\s*[{_Q}]\s+)(?:then\s+|please\s+|you\s+should\s+|just\s+)?(?P<act>.+)$", re.I),
    # Hinglish: Jab (bhi) main 'X' bolun, (to) <action>
    re.compile(rf"^\s*jab(?:\s+bhi)?\s+(?:main|mai|me|mein|hum)\s+{_TRIG}\s+(?:bolun|bolu|boloon|bolunga|bolungi|bolta hoon|"
               rf"bolti hoon|kahun|kahu|kahoon)\s*[,;:]?\s*(?:to|toh|tab)?\s*(?P<act>.+)$", re.I),
    # Devanagari: जब (भी) मैं 'X' बोलूँ, (तो) <action>
    re.compile(rf"^\s*जब(?:\s+भी)?\s+(?:मैं|हम)\s+{_TRIG}\s+(?:बोलूँ|बोलूं|बोलू|कहूँ|कहूं|बोलूँगा|बोलूँगी)\s*[,;:]?\s*(?:तो|तब)?\s*(?P<act>.+)$"),
]
_DELETE = [
    re.compile(rf"^\s*(?:please\s+)?(?:delete|remove|forget)\s+(?:the\s+|my\s+)?(?:shortcut|command|phrase)?\s*{_TRIG}\s*(?:shortcut|command)?[.!]?\s*$", re.I),
    re.compile(rf"^\s*{_TRIG}\s+(?:wala\s+|waala\s+)?(?:shortcut|command)?\s*(?:delete|hata|hatao|bhool)\s*(?:do|dena|jao|karo|kar do)?[.!]?\s*$", re.I),
]
_CORRECT = [
    re.compile(rf"^\s*(?:change|update|correct)\s+(?:the\s+|my\s+)?(?:shortcut\s+|command\s+)?{_TRIG}\s+(?:so that it should|so that it|so it|to|:)\s+(?P<act>.+)$", re.I),
    re.compile(rf"^\s*{_TRIG}\s+(?:ko|ka action)\s+(?:badal do|badlo|change karo|change kar do)\s*[,;:]?\s*(?P<act>.+)$", re.I),
]
_ALIAS = [
    re.compile(rf"^\s*(?:also\s+)?(?:make|let)\s+{_TRIG}\s+(?:also\s+)?(?:do the same as|mean the same as|work like|mean)\s+{_TRIG.replace('trig', 'target')}[.!]?\s*$", re.I),
    re.compile(rf"^\s*add\s+{_TRIG}\s+as\s+(?:another|an alternative)\s+(?:phrase|way to say)\s+(?:for\s+)?{_TRIG.replace('trig', 'target')}[.!]?\s*$", re.I),
]
_LIST = re.compile(r"^\s*(?:list|show)\s+(?:all\s+)?my\s+(?:shortcuts|commands|personal commands)|mere\s+(?:shortcuts|commands)\s+(?:batao|dikhao)", re.I)
_CONJ = re.compile(r"\s+(?:and then|and also|and|then|aur phir|aur|phir|फिर|और)\s+", re.I)


@dataclass
class TeachRequest:
    kind: str  # teach | delete | correct | alias | list
    trigger: str | None = None
    action_text: str | None = None
    target: str | None = None


def _clean(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip().strip(_Q).strip(" ,.;:!?")
    return " ".join(s.split()) or None


def parse_request(text: str) -> TeachRequest | None:
    if _LIST.search(text):
        return TeachRequest("list")
    for pat in _TEACH:
        m = pat.match(text)
        if m:
            return TeachRequest("teach", _clean(m.group("trig")), _clean(m.group("act")))
    for pat in _ALIAS:
        m = pat.match(text)
        if m:
            return TeachRequest("alias", _clean(m.group("trig")), target=_clean(m.group("target")))
    for pat in _CORRECT:
        m = pat.match(text)
        if m:
            return TeachRequest("correct", _clean(m.group("trig")), _clean(m.group("act")))
    for pat in _DELETE:
        m = pat.match(text)
        if m:
            return TeachRequest("delete", _clean(m.group("trig")))
    return None


class TeachError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def interpret_action_clause(clause: str, intent_model, policy) -> VehicleAction:
    """Classify the action clause with the trained model and validate it. Raises TeachError."""
    if not clause:
        raise TeachError("missing_action", "What should the shortcut do?")
    parts = [p for p in _CONJ.split(clause) if p.strip()]
    if len(parts) > 1:
        preds = [intent_model.predict(p)[0] for p in parts]
        execs = {p.intent for p in preds if p.intent in EXECUTABLE_INTENTS and p.score >= policy.th["min_score"]}
        if len(execs) > 1:
            raise TeachError("compound_action_unsupported",
                             "I can save only one action per shortcut for now. Which single action should it do?")
    pred, _ = intent_model.predict(clause)
    if pred.intent not in EXECUTABLE_INTENTS:
        raise TeachError("action_not_supported", "I can't link that to a car function I'm allowed to control.")
    if policy.is_negated(pred):
        raise TeachError("negated_action", "That sounds like something not to do. What should the shortcut do?")
    if pred.score < policy.th["min_score"] or pred.margin < policy.th["min_margin"]:
        raise TeachError("low_confidence", f"I'm not sure what the shortcut should do. Did you mean: {DESCRIPTIONS[pred.intent]}?")
    try:
        return validate_action(build_action(pred.intent, clause), policy.limits)
    except ActionValidationError as e:
        raise TeachError(e.code, f"I can't save that: {e.message}.") from e


def check_trigger(trigger: str | None, intent_model, policy) -> str:
    if not trigger:
        raise TeachError("missing_trigger", "What phrase do you want to use?")
    words = trigger.split()
    if not (1 <= len(words) <= 6) or len(trigger) > 40:
        raise TeachError("invalid_trigger", "Please use a short phrase of one to six words.")
    pred, _ = intent_model.predict(trigger)
    if pred.intent in EXECUTABLE_INTENTS and pred.score >= policy.th["min_score"] and pred.margin >= policy.th["min_margin"]:
        raise TeachError("trigger_conflicts_builtin",
                         f"'{trigger}' already sounds like a built-in command ({DESCRIPTIONS[pred.intent]}). Please pick another phrase.")
    return trigger
