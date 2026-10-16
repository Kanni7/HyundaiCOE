"""Integration tests with the real local V2 models.  Run:  pytest -m integration

Require: models downloaded (scripts/download_models.py), encoder exported and heads trained,
runtime promoted (see README). The offline test blocks every non-loopback socket and forces
Hugging Face offline mode before loading anything.
"""
import socket
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture
def no_internet(monkeypatch):
    real_connect, attempts = socket.socket.connect, []

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            attempts.append(host)
            raise OSError(f"Network access blocked by test: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    yield attempts


@pytest.fixture(scope="module")
def pipe_factory():
    from bhashadrive.nlu.encoder import EncoderUnavailableError
    from bhashadrive.pipeline_v2 import build_v2_pipeline

    def make(tmp_path, with_asr=False):
        try:
            return build_v2_pipeline(with_asr=with_asr, db_path=str(tmp_path / "p.sqlite"))
        except EncoderUnavailableError as exc:
            pytest.skip(f"V2 artifacts not built: {exc}")

    return make


@pytest.mark.integration
def test_v2_end_to_end_audio_offline(no_internet, tmp_path, pipe_factory):
    sample = ROOT / "samples" / "tts_v2" / "v13_hg_volume.wav"
    if not sample.exists():
        pytest.skip("run scripts/make_tts_samples.ps1 -OutDir samples\\tts_v2 -Manifest scripts\\tts_samples_v2.json")
    p = pipe_factory(tmp_path, with_asr=True)
    r = p.process_audio_file(sample)
    assert r.asr.status == "ok" and r.intent.value == "volume_up"
    assert r.decision.value == "execute" and p.vehicle.state["volume"] > 10
    assert no_internet == [], f"unexpected network attempts: {no_internet}"


@pytest.mark.integration
@pytest.mark.parametrize("text,intent", [
    ("Volume badha do, yaar.", "volume_up"), ("Set the cabin to 23 degrees.", "climate_temperature_set"),
    ("Take me to Connaught Place.", "navigation_destination_request"), ("Lock the car.", "unsupported"),
])
def test_v2_real_model_intents(no_internet, tmp_path, pipe_factory, text, intent):
    r = pipe_factory(tmp_path).process_text(text)
    assert r.intent.value == intent


@pytest.mark.integration
def test_v2_negation_never_executes_real_model(no_internet, tmp_path, pipe_factory):
    p = pipe_factory(tmp_path)
    for t in ("Don't turn the AC off.", "AC band mat karna.", "एसी को बंद मत करना।"):
        assert p.process_text(t).decision.value != "execute"
    assert p.vehicle.state["ac_on"] is True


@pytest.mark.integration
def test_v2_personal_command_real_model(no_internet, tmp_path, pipe_factory):
    p = pipe_factory(tmp_path)
    r = p.process_text("Jab main 'chill mode' bolun, AC 22 pe set kar dena.")
    assert r.decision.value == "ask_confirmation", r.response_text
    assert p.process_text("yes").decision.value == "saved"
    p.vehicle.state["cabin_temperature_c"] = 26
    assert p.process_text("Chill mode, yaar.").decision.value == "execute"
    assert p.vehicle.state["cabin_temperature_c"] == 22
    assert p.process_text("Don't use chill mode").decision.value == "reject"
