"""Deterministic decision policy. Sits between interpretation and actuation.

The LLM never decides whether something executes. This layer does, using the
validated frame, the resolved context and configured limits. When in doubt it
asks or declines rather than acting.
"""
from __future__ import annotations

from ..config import VehicleLimits
from ..schemas import Decision, Intent, PolicyResult, ResolvedRequest, SemanticFrame, SpeechAct, Verification
from ..nlu.semantic_parser import VERIFIER_FUNCTIONS
from .validation import ValidationIssue, validate_command

_EXPLICIT_ACTS = {SpeechAct.DIRECT_COMMAND, SpeechAct.FOLLOW_UP}
_IMPLICIT_ACTS = {SpeechAct.INDIRECT_REQUEST, SpeechAct.STATEMENT, SpeechAct.QUESTION}


def _reject(code: str, question: str | None = None, command=None) -> PolicyResult:
    return PolicyResult(decision=Decision.CLARIFY_OR_REJECT, reason_code=code, clarifying_question=question, command=command)


class CommandPolicy:
    def __init__(self, limits: VehicleLimits | None = None):
        self.limits = limits or VehicleLimits()

    def decide(self, frame: SemanticFrame | None, resolved: ResolvedRequest | None, nlu_error: str | None = None,
               verification: Verification | None = None, context_intents: set | None = None) -> PolicyResult:
        """verification: independent second opinion for standalone requests (None = not run).
        context_intents: intents of the pending/last command, which can legitimately fill in an omitted function."""
        if nlu_error or frame is None or resolved is None:
            return _reject("model_error", "Sorry, I could not understand that. Could you say it again?")

        if resolved.context_action == "missing_context":
            return _reject("missing_context", "What would you like me to adjust?")

        cmd = resolved.command
        if cmd is None:
            if resolved.context_action == "cleared_pending":
                return _reject("cancelled_by_user")
            if frame.intent == Intent.UNSUPPORTED or frame.speech_act in (
                SpeechAct.UNSUPPORTED_REQUEST, SpeechAct.DIRECT_COMMAND, SpeechAct.INDIRECT_REQUEST):
                # A request with no whitelisted intent is out of scope (e.g. brakes, windows).
                return _reject("unsupported_request")
            return _reject("no_action", frame.clarifying_question if frame.ambiguous else None)

        if frame.negated and resolved.context_action in ("none", "cleared_pending"):
            # The driver asked us NOT to do something: never act on it.
            return _reject("negated_no_action", command=cmd)

        if verification is not None and cmd.intent.value not in VERIFIER_FUNCTIONS[verification.function]:
            context_explains = verification.function == "none" and cmd.intent in (context_intents or set())
            if not context_explains:
                # Two independently prompted interpretations disagree: treat as uncertain, do not act.
                return _reject("interpretation_disagreement", "Sorry, which setting would you like me to change?")

        if frame.ambiguous and not resolved.confirmed_by_user:
            return _reject("ambiguous", frame.clarifying_question or "Could you tell me exactly what to change?")

        try:
            cmd = validate_command(cmd, self.limits)
        except ValidationIssue as issue:
            question = {
                "missing_parameter": f"I need a bit more detail: {issue.message}.",
                "invalid_parameter": f"I can't do that: {issue.message}.",
            }.get(issue.code)
            return _reject(issue.code, question)

        if resolved.confirmed_by_user:
            return PolicyResult(decision=Decision.EXECUTE, reason_code="confirmed_by_user", command=cmd)
        if resolved.context_action == "used_last":
            return PolicyResult(decision=Decision.EXECUTE, reason_code="explicit_follow_up", command=cmd)
        explicit = frame.speech_act in _EXPLICIT_ACTS
        if explicit and verification is not None and verification.request_type != "explicit_request":
            explicit = False  # the two readings disagree on explicitness: confirm instead of acting
        if explicit:
            return PolicyResult(decision=Decision.EXECUTE, reason_code="explicit_command", command=cmd)
        if frame.speech_act in _IMPLICIT_ACTS or frame.speech_act in _EXPLICIT_ACTS:
            return PolicyResult(
                decision=Decision.ASK_CONFIRMATION,
                reason_code="implicit_request",
                command=cmd,
                clarifying_question=f"Would you like me to {cmd.describe()}?",
            )
        return _reject("unsupported_request")
