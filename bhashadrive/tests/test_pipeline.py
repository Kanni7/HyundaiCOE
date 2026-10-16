import numpy as np

from bhashadrive.audio.asr import MockASR
from bhashadrive.config import AppConfig
from bhashadrive.nlu.base import LLMOutputError, LLMTimeoutError, LLMUnavailableError
from bhashadrive.pipeline import BhashaDrivePipeline, normalize_transcript
from bhashadrive.schemas import Decision
from bhashadrive.vehicle.hyundai_adapter import HyundaiVehicleAdapter

from .conftest import ScriptedParser, V, frame


def test_normalization_keeps_negation_and_script():
    assert normalize_transcript("  AC  band\tmat   karna. ") == "AC band mat karna."
    assert normalize_transcript("यहाँ बहुत ठंड") == "यहाँ बहुत ठंड"
    assert normalize_transcript("Don't  warm it") == "Don't warm it"


def test_cold_then_yes_a_little_dialogue(make_pipeline):
    p = make_pipeline(
        {"It's too cold.": frame("climate_adjust_temperature", act="indirect_request", direction="increase"),
         "Yes, just a little.": frame("none", act="follow_up", rel="confirm_pending", amount="small"),
         "A bit more.": frame("none", act="follow_up", rel="repeat_last"),
         "No, that's too much.": frame("climate_adjust_temperature", act="follow_up", rel="reverse_last"),
         "Cancel that.": frame("none", act="follow_up", rel="cancel")},
        {"It's too cold.": V("cabin_temperature", "describes_situation")},
    )
    r = p.process_text("It's too cold.")
    assert r.policy.decision == Decision.ASK_CONFIRMATION and p.vehicle.state["cabin_temperature_c"] == 22
    r = p.process_text("Yes, just a little.")
    assert r.policy.decision == Decision.EXECUTE and p.vehicle.state["cabin_temperature_c"] == 23
    p.process_text("A bit more.")
    assert p.vehicle.state["cabin_temperature_c"] == 24
    p.process_text("No, that's too much.")
    assert p.vehicle.state["cabin_temperature_c"] == 23
    r = p.process_text("Cancel that.")  # nothing pending now
    assert r.policy.reason_code == "missing_context"


def test_correction_switches_target(make_pipeline):
    p = make_pipeline(
        {"cold": frame("climate_adjust_temperature", act="indirect_request", direction="increase"),
         "Not the temperature, reduce the fan speed.": frame("fan_adjust_speed", rel="correct_previous", direction="decrease")},
        {"cold": V("cabin_temperature", "describes_situation"),
         "Not the temperature, reduce the fan speed.": V("fan_speed")},
    )
    p.process_text("cold")
    r = p.process_text("Not the temperature, reduce the fan speed.")
    assert r.policy.decision == Decision.EXECUTE and p.vehicle.state["fan_speed"] == 2
    assert p.vehicle.state["cabin_temperature_c"] == 22 and p.context.pending is None


def test_no_execution_after_ambiguous_request(make_pipeline):
    p = make_pipeline({"Change it.": frame("none", ambiguous=True),
                       "Yes.": frame("none", act="follow_up", rel="confirm_pending")})
    p.process_text("Change it.")
    r = p.process_text("Yes.")
    assert r.policy.decision == Decision.CLARIFY_OR_REJECT and p.vehicle.log == []


def test_rejecting_pending_does_nothing(make_pipeline):
    p = make_pipeline({"hot": frame("climate_adjust_temperature", act="statement", direction="decrease"),
                       "No.": frame("none", act="follow_up", rel="reject_pending")},
                      {"hot": V("cabin_temperature", "describes_situation")})
    p.process_text("hot")
    r = p.process_text("No.")
    assert r.policy.reason_code == "cancelled_by_user" and p.vehicle.log == []


def test_model_errors_never_execute_and_never_fall_back():
    for exc in (LLMTimeoutError("t"), LLMUnavailableError("down"), LLMOutputError("bad json")):
        p = BhashaDrivePipeline(ScriptedParser({"Volume up": exc}), _vehicle(), None, AppConfig())
        r = p.process_text("Volume up")
        assert r.policy.reason_code == "model_error" and r.action is None and r.frame is None


def test_verifier_failure_blocks_action():
    p = BhashaDrivePipeline(ScriptedParser({"Volume up": frame("volume_adjust", direction="increase")},
                                           {"Volume up": LLMTimeoutError("t")}), _vehicle(), None, AppConfig())
    r = p.process_text("Volume up")
    assert r.policy.reason_code == "model_error" and p.vehicle.log == []


def test_empty_transcript_and_asr_silence():
    p = BhashaDrivePipeline(ScriptedParser({}), _vehicle(), MockASR("", status="silence"), AppConfig())
    r = p.process_audio_array(np.zeros(16000, dtype=np.float32))
    assert r.policy.reason_code == "silence" and r.action is None
    assert p.process_text("   ").policy.reason_code == "empty_transcript"


def test_audio_path_through_mock_asr():
    p = BhashaDrivePipeline(ScriptedParser({"Pause the music.": frame("media_pause")},
                                           {"Pause the music.": V("media_playback")}),
                            _vehicle(), MockASR("Pause the music."), AppConfig())
    r = p.process_audio_file("ignored.wav")
    assert r.policy.decision == Decision.EXECUTE and r.action.success and "asr_s" in r.timings_s


def test_hyundai_boundary_reports_failure_not_crash():
    p = BhashaDrivePipeline(ScriptedParser({"next": frame("media_next_track")}, {"next": V("media_playback")}),
                            HyundaiVehicleAdapter(), None, AppConfig())
    r = p.process_text("next")
    assert r.action is not None and not r.action.success and "couldn't" in r.response_text


def test_audio_not_stored_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = BhashaDrivePipeline(ScriptedParser({}), _vehicle(), MockASR("", status="silence"), AppConfig())
    p.process_audio_array(np.ones(1600, dtype=np.float32))
    assert not (tmp_path / "recordings").exists()


def _vehicle():
    from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

    return MockVehicleAdapter()
