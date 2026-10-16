"""Model-loading failures must fail closed (clear error, no vehicle action, no fallback)."""
import json

import numpy as np
import pytest

from bhashadrive.audio.asr import ASRUnavailableError
from bhashadrive.audio.sherpa_asr import SherpaWhisperASR
from bhashadrive.config import AppConfig, SherpaASRConfig
from bhashadrive.nlu.encoder import EncoderUnavailableError, OnnxSentenceEncoder
from bhashadrive.nlu.intent_model import IntentModel
from bhashadrive.nlu.taxonomy import TAXONOMY_VERSION


def test_missing_asr_model_raises(tmp_path):
    cfg = SherpaASRConfig()
    cfg.model_dir = tmp_path
    with pytest.raises(ASRUnavailableError):
        SherpaWhisperASR(cfg)


def test_missing_nlu_artifacts_raise(tmp_path):
    with pytest.raises(EncoderUnavailableError):
        IntentModel.load(tmp_path, tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"encoder": {"onnx": "nope.onnx", "tokenizer_dir": "x"}}))
    with pytest.raises(EncoderUnavailableError):
        IntentModel.load(tmp_path, tmp_path)


def test_missing_encoder_files_raise(tmp_path):
    with pytest.raises(EncoderUnavailableError):
        OnnxSentenceEncoder(tmp_path / "model.onnx", tmp_path)


def test_taxonomy_mismatch_rejected():
    manifest = {"taxonomy_version": "0.0.1", "classes": [], "model_version": "x"}
    with pytest.raises(EncoderUnavailableError):
        IntentModel(manifest, {}, encoder=None)


def test_build_pipeline_fails_closed_without_artifacts(tmp_path, monkeypatch):
    from bhashadrive.pipeline_v2 import build_v2_pipeline

    cfg = AppConfig()
    cfg.runtime.root = tmp_path
    with pytest.raises(EncoderUnavailableError):
        build_v2_pipeline(cfg, with_asr=False)


def test_sherpa_input_validation_without_model(tmp_path):
    asr = SherpaWhisperASR.__new__(SherpaWhisperASR)  # bypass model loading to test input guards
    asr.config = SherpaASRConfig()
    assert asr.transcribe_array(np.zeros(0, np.float32)).status == "no_audio"
    assert asr.transcribe_array(np.zeros(16000, np.float32)).status == "silence"
    assert asr.transcribe_array(np.ones(100, np.float32), 8000).status == "error"
    assert asr.transcribe_array(np.ones(16000 * 60, np.float32) * 0.1).status == "error"  # too long
    assert asr.transcribe_file(tmp_path / "missing.wav").status == "no_audio"
    (tmp_path / "junk.wav").write_bytes(b"not audio at all")
    assert asr.transcribe_file(tmp_path / "junk.wav").status == "error"


def test_sherpa_timeout_returns_error():
    import time
    from concurrent.futures import ThreadPoolExecutor

    asr = SherpaWhisperASR.__new__(SherpaWhisperASR)
    asr.config = SherpaASRConfig()
    asr.config.timeout_s = 0.05
    asr._pool = ThreadPoolExecutor(max_workers=1)
    asr._decode = lambda audio: time.sleep(0.5) or "late"
    r = asr.transcribe_array(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)
    assert r.status == "error" and "timeout" in r.error


def test_taxonomy_version_constant():
    assert TAXONOMY_VERSION.count(".") == 2
