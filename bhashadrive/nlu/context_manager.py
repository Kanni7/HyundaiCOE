"""Bounded conversation context and deterministic follow-up resolution.

The LLM labels *how* an utterance relates to the conversation (context_relation);
this module decides deterministically *what* that means given the stored state.
A follow-up is only attached to earlier context when compatible state exists.
Only the minimum state is kept: the pending proposal, the last executed command
and a few recent transcript/response pairs. No audio is stored.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from ..config import ContextConfig
from ..schemas import (
    ACTIONABLE_INTENTS,
    ADJUSTABLE_INTENTS,
    Amount,
    Command,
    ContextRelation,
    Direction,
    ResolvedRequest,
    SemanticFrame,
)


@dataclass
class _Stamped:
    command: Command
    at: float


class ConversationContext:
    def __init__(self, config: ContextConfig | None = None, clock=time.monotonic):
        self.config = config or ContextConfig()
        self._clock = clock
        self._pending: _Stamped | None = None
        self._last: _Stamped | None = None
        self.history: deque[tuple[str, str]] = deque(maxlen=self.config.max_turns)

    # ---- state access -------------------------------------------------
    @property
    def pending(self) -> Command | None:
        if self._pending and self._clock() - self._pending.at > self.config.pending_ttl_s:
            self._pending = None
        return self._pending.command if self._pending else None

    @property
    def last_executed(self) -> Command | None:
        if self._last and self._clock() - self._last.at > self.config.followup_ttl_s:
            self._last = None
        return self._last.command if self._last else None

    def set_pending(self, command: Command | None) -> None:
        self._pending = _Stamped(command, self._clock()) if command else None

    def set_last_executed(self, command: Command) -> None:
        self._last = _Stamped(command, self._clock())

    def reset(self) -> None:
        self._pending = self._last = None
        self.history.clear()

    def summary(self) -> str:
        lines = [
            f"Pending proposal awaiting the driver's yes/no: {self.pending.describe() if self.pending else 'none'}",
            f"Last action executed: {self.last_executed.describe() if self.last_executed else 'none'}",
        ]
        if self.history:
            lines.append("Recent turns:")
            lines += [f"  Driver: {u}\n  Assistant: {a}" for u, a in list(self.history)[-3:]]
        return "\n".join(lines)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.append((user_text[:200], assistant_text[:200]))

    # ---- resolution ---------------------------------------------------
    def resolve(self, frame: SemanticFrame) -> ResolvedRequest:
        rel = frame.context_relation
        own = Command(intent=frame.intent, parameters=frame.parameters) if frame.intent in ACTIONABLE_INTENTS else None
        pending, last = self.pending, self.last_executed

        if rel == ContextRelation.CONFIRM_PENDING:
            if pending:
                params = pending.parameters.model_copy()
                if frame.parameters.amount:
                    params.amount = frame.parameters.amount
                if frame.parameters.target_temperature_celsius is not None and pending.intent.value.startswith("climate"):
                    params.target_temperature_celsius = frame.parameters.target_temperature_celsius
                return ResolvedRequest(command=Command(intent=pending.intent, parameters=params),
                                       confirmed_by_user=True, context_action="used_pending")
            if own:
                return ResolvedRequest(command=own, note="confirmation without pending proposal; using stated action")
            return ResolvedRequest(context_action="missing_context", note="nothing pending to confirm")

        if rel in (ContextRelation.REJECT_PENDING, ContextRelation.CANCEL):
            if pending:
                return ResolvedRequest(context_action="cleared_pending", note="pending proposal cancelled")
            return ResolvedRequest(context_action="missing_context", note="nothing pending to cancel")

        if rel in (ContextRelation.REPEAT_LAST, ContextRelation.REVERSE_LAST):
            reverse = rel == ContextRelation.REVERSE_LAST
            base, source = None, None
            if last and last.intent in ADJUSTABLE_INTENTS and last.parameters.direction:
                base, source = last, "used_last"
            elif not reverse and pending and pending.intent in ADJUSTABLE_INTENTS and pending.parameters.direction:
                base, source = pending, "used_pending"
            if base is None:
                return ResolvedRequest(context_action="missing_context", note="no earlier adjustment to refer to")
            direction = base.parameters.direction
            if reverse:
                direction = Direction.DECREASE if direction == Direction.INCREASE else Direction.INCREASE
            if own and (own.intent != base.intent or (own.parameters.direction and own.parameters.direction != direction)):
                # The utterance carries its own complete meaning that conflicts with the follow-up reading:
                # treat it as a standalone request instead of forcing it onto earlier context.
                if own.parameters.direction or own.intent not in ADJUSTABLE_INTENTS:
                    return ResolvedRequest(command=own, note="follow-up label conflicted with stated action; treated as standalone")
                return ResolvedRequest(context_action="missing_context", note="follow-up does not match earlier action")
            amount = frame.parameters.amount or Amount.SMALL
            cmd = Command(intent=base.intent, parameters={"direction": direction, "amount": amount})
            return ResolvedRequest(command=cmd, confirmed_by_user=source == "used_pending", context_action=source)

        if rel == ContextRelation.CORRECT_PREVIOUS:
            if own:
                return ResolvedRequest(command=own, context_action="cleared_pending" if pending else "none",
                                       note="driver corrected the previous target")
            return ResolvedRequest(context_action="missing_context", note="correction without a new action")

        return ResolvedRequest(command=own)

    def record_outcome(self, command: Command | None, decision: str, executed: bool) -> None:
        if executed and command:
            self.set_last_executed(command)
            self.set_pending(None)
        elif decision == "ask_confirmation" and command:
            self.set_pending(command)
        else:
            self.set_pending(None)
