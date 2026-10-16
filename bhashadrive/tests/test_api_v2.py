from fastapi.testclient import TestClient

from bhashadrive.api.server import create_app
from bhashadrive.audio.asr import MockASR
from bhashadrive.config import AppConfig

from .fakes_v2 import make_pipeline


def test_v2_api_text_audio_personal_reset():
    pipe = make_pipeline()
    pipe.asr = MockASR("pause the music")
    c = TestClient(create_app(AppConfig(), pipe))
    assert "fake" in c.get("/health").json()["nlu"]
    j = c.post("/text", json={"text": "volume badha do"}).json()
    assert j["decision"] == "execute" and c.get("/state").json()["vehicle"]["volume"] == 12
    j = c.post("/audio", files={"file": ("a.wav", b"RIFF0000", "audio/wav")}).json()
    assert j["action"]["intent"] == "media_pause"
    c.post("/text", json={"text": "When I say 'party time', turn the volume up."})
    c.post("/text", json={"text": "yes"})
    assert [p["trigger"] for p in c.get("/personal").json()] == ["party time"]
    assert c.post("/reset").json() == {"ok": True}
