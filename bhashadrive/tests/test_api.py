from fastapi.testclient import TestClient

from bhashadrive.api.server import create_app
from bhashadrive.audio.asr import MockASR
from bhashadrive.config import AppConfig
from bhashadrive.pipeline import BhashaDrivePipeline
from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

from .conftest import ScriptedParser, V, frame


def client():
    parser = ScriptedParser({"Pause the music.": frame("media_pause")}, {"Pause the music.": V("media_playback")})
    pipe = BhashaDrivePipeline(parser, MockVehicleAdapter(), MockASR("Pause the music."), AppConfig())
    return TestClient(create_app(AppConfig(), pipe))


def test_text_endpoint_executes_and_state_updates():
    c = client()
    c.post("/text", json={"text": "Pause the music."})
    j = c.post("/text", json={"text": "Pause the music."}).json()
    assert j["policy"]["decision"] == "execute"
    assert c.get("/state").json()["vehicle"]["media_playing"] is False


def test_audio_upload_and_reset(tmp_path):
    c = client()
    j = c.post("/audio", files={"file": ("a.wav", b"RIFF0000", "audio/wav")}).json()
    assert j["transcript"] == "Pause the music." and j["action"]["success"]
    assert c.post("/reset").json() == {"ok": True}


def test_index_page_served():
    assert "BhashaDrive" in client().get("/").text


def test_text_length_limited():
    assert client().post("/text", json={"text": "x" * 1000}).status_code == 422
