from bhashadrive.config import ContextConfig
from bhashadrive.nlu.context_manager import ConversationContext
from bhashadrive.schemas import Command, Direction

from .conftest import frame

WARM = Command(intent="climate_adjust_temperature", parameters={"direction": "increase"})
VOL = Command(intent="volume_adjust", parameters={"direction": "increase", "amount": "small"})


def test_confirm_pending_with_amount_override():
    ctx = ConversationContext()
    ctx.set_pending(WARM)
    r = ctx.resolve(frame("none", act="follow_up", rel="confirm_pending", amount="small"))
    assert r.confirmed_by_user and r.command.intent == WARM.intent
    assert r.command.parameters.amount.value == "small" and r.command.parameters.direction == Direction.INCREASE


def test_confirm_without_pending_is_missing_context():
    r = ConversationContext().resolve(frame("none", act="follow_up", rel="confirm_pending"))
    assert r.context_action == "missing_context" and r.command is None


def test_repeat_last():
    ctx = ConversationContext()
    ctx.set_last_executed(VOL)
    r = ctx.resolve(frame("none", act="follow_up", rel="repeat_last"))
    assert r.context_action == "used_last" and r.command.parameters.direction == Direction.INCREASE


def test_reverse_last_flips_direction():
    ctx = ConversationContext()
    ctx.set_last_executed(VOL)
    r = ctx.resolve(frame("volume_adjust", act="follow_up", rel="reverse_last"))
    assert r.command.parameters.direction == Direction.DECREASE


def test_repeat_without_context_is_not_attached():
    r = ConversationContext().resolve(frame("climate_adjust_temperature", act="follow_up", rel="repeat_last",
                                            direction="increase"))
    assert r.context_action == "missing_context" and r.command is None


def test_repeat_with_non_adjustable_last_is_missing_context():
    ctx = ConversationContext()
    ctx.set_last_executed(Command(intent="media_next_track"))
    assert ctx.resolve(frame("none", act="follow_up", rel="repeat_last")).context_action == "missing_context"


def test_conflicting_follow_up_label_treated_as_standalone():
    # Model labels a complete new request as reverse_last; the stated meaning wins and no context is used.
    ctx = ConversationContext()
    ctx.set_last_executed(Command(intent="climate_adjust_temperature", parameters={"direction": "increase"}))
    r = ctx.resolve(frame("climate_adjust_temperature", act="indirect_request", rel="reverse_last", direction="increase"))
    assert r.context_action == "none" and r.command.parameters.direction == Direction.INCREASE


def test_cancel_clears_pending():
    ctx = ConversationContext()
    ctx.set_pending(WARM)
    r = ctx.resolve(frame("none", act="follow_up", rel="cancel"))
    assert r.context_action == "cleared_pending" and r.command is None


def test_cancel_with_nothing_pending():
    assert ConversationContext().resolve(frame("none", rel="cancel")).context_action == "missing_context"


def test_correction_replaces_pending():
    ctx = ConversationContext()
    ctx.set_pending(WARM)
    r = ctx.resolve(frame("fan_adjust_speed", rel="correct_previous", direction="decrease"))
    assert r.command.intent.value == "fan_adjust_speed" and r.context_action == "cleared_pending"


def test_pending_expires():
    now = [0.0]
    ctx = ConversationContext(ContextConfig(pending_ttl_s=10), clock=lambda: now[0])
    ctx.set_pending(WARM)
    now[0] = 11
    assert ctx.pending is None


def test_history_is_bounded():
    ctx = ConversationContext(ContextConfig(max_turns=3))
    for i in range(10):
        ctx.add_turn(f"u{i}", f"a{i}")
    assert len(ctx.history) == 3


def test_record_outcome():
    ctx = ConversationContext()
    ctx.record_outcome(WARM, "ask_confirmation", executed=False)
    assert ctx.pending == WARM
    ctx.record_outcome(WARM, "execute", executed=True)
    assert ctx.pending is None and ctx.last_executed == WARM
    ctx.set_pending(WARM)
    ctx.record_outcome(None, "clarify_or_reject", executed=False)
    assert ctx.pending is None
