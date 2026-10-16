"""Personalised command learning: parsing, SQLite persistence, matching, corrections."""
import sqlite3

import numpy as np
import pytest

from bhashadrive.personalization.matcher import MatchThresholds
from bhashadrive.personalization.store import PersonalCommandStore
from bhashadrive.personalization.teaching import parse_request
from bhashadrive.schemas_v2 import DecisionV2, VehicleAction

from .fakes_v2 import FakeIntentModel, make_pipeline

COZY = VehicleAction(intent="climate_temperature_set", parameters={"target_temperature_celsius": 22})


# ---- parsing ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,trigger,action", [
    ("When I say 'make it cozy', set the temperature to 22 degrees.", "make it cozy", "set the temperature to 22 degrees"),
    ("Jab main 'chill mode' bolun, AC 22 pe set kar dena.", "chill mode", "AC 22 pe set kar dena"),
    ("jab bhi main chill mode bolu to AC 22 pe set kar dena", "chill mode", "AC 22 pe set kar dena"),
    ("जब मैं 'आराम' बोलूँ, तो तापमान 22 कर दो", "आराम", "तापमान 22 कर दो"),
    ("if I say party time then turn the volume up", "party time", "turn the volume up"),
])
def test_parse_teach(text, trigger, action):
    r = parse_request(text)
    assert (r.kind, r.trigger, r.action_text) == ("teach", trigger, action)


def test_parse_manage():
    assert parse_request("delete 'chill mode'").kind == "delete"
    assert parse_request("chill mode wala shortcut hata do").trigger == "chill mode"
    r = parse_request("change 'chill mode' to set the temperature to 20")
    assert (r.kind, r.trigger, r.action_text) == ("correct", "chill mode", "set the temperature to 20")
    r = parse_request("make 'relax' do the same as 'chill mode'")
    assert (r.kind, r.trigger, r.target) == ("alias", "relax", "chill mode")
    assert parse_request("list my shortcuts").kind == "list"
    assert parse_request("turn the volume up") is None


# ---- store -----------------------------------------------------------------------------
def test_store_persists_across_reopen(tmp_path):
    db = tmp_path / "p.sqlite"
    s = PersonalCommandStore(db)
    cid = s.add_command("u1", "make it cozy", COZY, np.ones(4, np.float32), "m1")
    s.close()
    s = PersonalCommandStore(db)
    c = s.get("u1", cid)
    assert c.trigger == "make it cozy" and c.action == COZY and c.user_confirmed
    assert s.get("u2", cid) is None  # profiles are isolated
    assert s.find_by_trigger("u1", "Make it COZY!").id == cid
    row = s.db.execute("SELECT created_at, updated_at, taxonomy_version, embedding_model FROM personal_commands").fetchone()
    assert all(row)


def test_store_delete_and_unique_trigger(tmp_path):
    s = PersonalCommandStore(tmp_path / "p.sqlite")
    cid = s.add_command("u1", "chill mode", COZY, None, None)
    with pytest.raises(sqlite3.IntegrityError):
        s.add_command("u1", "Chill  Mode", COZY, None, None)
    assert s.delete_command("u1", cid) and s.list_commands("u1") == []
    assert s.db.execute("SELECT COUNT(*) FROM command_phrases").fetchone()[0] == 0  # cascade


def test_store_never_loads_invalid_action(tmp_path):
    s = PersonalCommandStore(tmp_path / "p.sqlite")
    cid = s.add_command("u1", "x", COZY, None, None)
    with s.db:  # simulate tampering / an old taxonomy
        s.db.execute("UPDATE personal_commands SET intent='open_the_doors' WHERE id=?", (cid,))
    assert s.get("u1", cid) is None and s.list_commands("u1") == []


def test_store_recomputes_stale_embeddings(tmp_path):
    s = PersonalCommandStore(tmp_path / "p.sqlite")
    s.add_command("u1", "chill mode", COZY, np.zeros(3, np.float32), "old-model")
    calls = []

    def embed(texts):
        calls.append(texts)
        return np.ones((len(texts), 3), np.float32)

    pairs, embs = s.phrase_index("u1", "new-model", embed)
    assert calls == [["chill mode"]] and embs.shape == (1, 3) and embs[0, 0] == 1
    s.phrase_index("u1", "new-model", embed)
    assert len(calls) == 1  # cached now


def test_correction_history_is_opt_in(tmp_path):
    new = VehicleAction(intent="volume_up")
    s = PersonalCommandStore(tmp_path / "a.sqlite")
    cid = s.add_command("u1", "x", COZY, None, None)
    s.update_action("u1", cid, new)
    assert s.correction_count("u1") == 0 and s.get("u1", cid).action == new
    s2 = PersonalCommandStore(tmp_path / "b.sqlite", store_correction_history=True)
    cid = s2.add_command("u1", "x", COZY, None, None)
    s2.update_action("u1", cid, new)
    assert s2.correction_count("u1") == 1


def test_no_audio_columns():
    s = PersonalCommandStore(":memory:")
    cols = {r[1] for t in ("personal_commands", "command_phrases", "corrections")
            for r in s.db.execute(f"PRAGMA table_info({t})")}
    assert not any("audio" in c for c in cols)


# ---- end-to-end through the pipeline -----------------------------------------------------
def teach(p, text="Jab main 'chill mode' bolun, AC 22 pe set kar dena."):
    before = len(p.store.list_commands("test-user"))
    r = p.process_text(text)
    assert r.decision == DecisionV2.ASK_CONFIRMATION, r.response_text
    assert len(p.store.list_commands("test-user")) == before  # nothing saved before confirmation
    assert p.process_text("yes").decision == DecisionV2.SAVED
    return r


def test_teach_confirm_and_use():
    p = make_pipeline()
    r = teach(p)
    assert r.action.intent.value == "climate_temperature_set" and r.action.parameters.target_temperature_celsius == 22
    p.vehicle.state["cabin_temperature_c"] = 25
    u = p.process_text("Chill mode, yaar.")
    assert u.decision == DecisionV2.EXECUTE and u.source == "personal" and p.vehicle.state["cabin_temperature_c"] == 22


def test_teach_cancel_saves_nothing():
    p = make_pipeline()
    p.process_text("When I say 'party time', turn the volume up.")
    assert p.process_text("no").reason_code == "cancelled_by_user"
    assert p.store.list_commands("test-user") == []


def test_teach_rejects_unsupported_compound_and_invalid():
    p = make_pipeline()
    assert p.process_text("When I say 'open sesame', open the sunroof").reason_code == "action_not_supported"
    assert p.process_text("When I say 'night', pause the music and turn the volume down").reason_code == "compound_action_unsupported"
    assert p.process_text("When I say 'sauna', set the temperature to 40").reason_code == "invalid_parameter"
    assert p.process_text("When I say 'hmm', set the temperature").reason_code == "missing_parameter"
    assert p.process_text("When I say 'volume up', pause the music").reason_code == "trigger_conflicts_builtin"
    assert p.process_text("When I say 'blowy', fan").reason_code == "low_confidence"
    assert p.store.list_commands("test-user") == []


def test_negated_and_quoted_uses_do_not_execute():
    p = make_pipeline()
    teach(p)
    log_len = len(p.vehicle.log)
    r = p.process_text("Don't use chill mode")
    assert r.decision == DecisionV2.REJECT and r.reason_code == "negated_no_action"
    r = p.process_text("what does 'chill mode' do?")
    assert r.decision == DecisionV2.ASK_CONFIRMATION
    assert len(p.vehicle.log) == log_len


def test_unrelated_utterance_does_not_match():
    p = make_pipeline()
    teach(p)
    r = p.process_text("pause the music")
    assert r.source == "builtin" and r.action.intent.value == "media_pause"


def test_ambiguous_between_two_commands():
    p = make_pipeline(thresholds=MatchThresholds(0.5, 0.5, 0.3, True))
    teach(p, "When I say 'mode alpha', turn the volume up.")
    teach(p, "When I say 'mode beta', pause the music.")
    r = p.process_text("mode")
    assert r.decision != DecisionV2.EXECUTE


def test_duplicate_trigger_rejected():
    p = make_pipeline()
    teach(p)
    assert p.process_text("When I say 'chill mode', turn the volume up.").reason_code == "trigger_exists"


def test_correction_alias_delete_flow():
    p = make_pipeline()
    teach(p)
    r = p.process_text("change 'chill mode' to set the temperature to 20")
    assert r.decision == DecisionV2.ASK_CONFIRMATION
    assert p.process_text("yes").decision == DecisionV2.SAVED
    p.process_text("chill mode")
    assert p.vehicle.state["cabin_temperature_c"] == 20
    p.process_text("make 'relax' do the same as 'chill mode'")
    p.process_text("yes")
    p.vehicle.state["cabin_temperature_c"] = 25
    assert p.process_text("relax").decision == DecisionV2.EXECUTE and p.vehicle.state["cabin_temperature_c"] == 20
    p.process_text("delete 'chill mode'")
    assert p.process_text("yes").reason_code == "delete_saved"
    assert p.store.list_commands("test-user") == []
    assert p.process_text("chill mode").source != "personal"


def test_complaint_after_personal_command_is_not_learned():
    p = make_pipeline()
    teach(p)
    p.process_text("chill mode")
    r = p.process_text("no")
    assert r.reason_code == "personal_feedback_not_learned"
    assert p.store.get("test-user", p.store.list_commands("test-user")[0].id).action.parameters.target_temperature_celsius == 22


def test_unknown_shortcut_management():
    p = make_pipeline()
    assert p.process_text("delete 'ghost mode'").reason_code == "unknown_shortcut"


def test_persisted_command_survives_restart(tmp_path):
    db = tmp_path / "profiles.sqlite"
    p = make_pipeline(store_path=db)
    teach(p)
    p.store.close()
    p2 = make_pipeline(store_path=db)
    p2.vehicle.state["cabin_temperature_c"] = 26
    assert p2.process_text("chill mode").decision == DecisionV2.EXECUTE
    assert p2.vehicle.state["cabin_temperature_c"] == 22
