"""Integration tests with the real local models.  Run:  pytest -m integration

The offline test blocks every non-loopback socket connection inside this process
and forces Hugging Face offline mode, then runs ASR and LLM inference. It proves
the Python side makes no network calls after setup. It cannot prove the Ollama
server process itself is offline; see README "Offline operation" for the
machine-level check (disconnect network, rerun).
"""
import os
import socket
from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "samples" / "tts"
_LOOPBACK = ("127.0.0.1", "::1", "localhost")


@pytest.fixture
def no_internet(monkeypatch):
    real_connect = socket.socket.connect
    attempts = []

    def guarded(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in _LOOPBACK:
            attempts.append(host)
            raise OSError(f"Network access blocked by test: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    yield attempts


@pytest.mark.integration
def test_end_to_end_offline(no_internet):
    from bhashadrive.config import AppConfig
    from bhashadrive.pipeline import build_default_pipeline

    sample = SAMPLES / "s09_pause_en.wav"
    if not sample.exists():
        pytest.skip("run scripts/make_tts_samples.ps1 first")
    cfg = AppConfig()
    cfg.asr.local_files_only = True
    pipe = build_default_pipeline(cfg, with_asr=True)
    r = pipe.process_audio_file(sample)
    assert r.asr.status == "ok" and "pause" in r.asr.text.lower()
    assert r.frame is not None and r.frame.intent.value == "media_pause"
    assert no_internet == [], f"unexpected network attempts: {no_internet}"


@pytest.mark.integration
def test_asr_missing_and_silent_audio(tmp_path):
    import numpy as np

    from bhashadrive.audio.asr import FasterWhisperASR

    asr = FasterWhisperASR()
    assert asr.transcribe_file(tmp_path / "missing.wav").status == "no_audio"
    (tmp_path / "empty.wav").write_bytes(b"")
    assert asr.transcribe_file(tmp_path / "empty.wav").status == "no_audio"
    (tmp_path / "junk.wav").write_bytes(b"not audio at all")
    assert asr.transcribe_file(tmp_path / "junk.wav").status == "error"
    assert asr.transcribe_array(np.zeros(16000, dtype=np.float32)).status == "silence"
