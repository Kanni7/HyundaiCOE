"""On-device ASR through sherpa-onnx (Whisper-architecture ONNX exports, e.g. hi-hinglish-swift).

The `SpeechRecognizer` interface is runtime-agnostic: an Android/embedded build would
implement the same contract with the sherpa-onnx Kotlin/C++ API and the same model files.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

import numpy as np

from ..config import SherpaASRConfig
from ..schemas import ASRResult
from .asr import ASRUnavailableError, SpeechRecognizer
from .preprocessing import TARGET_SR, is_silent, preprocess


class SherpaWhisperASR(SpeechRecognizer):
    def __init__(self, config: SherpaASRConfig | None = None):
        self.config = c = config or SherpaASRConfig()
        d = Path(c.model_dir)
        files = {k: d / f for k, f in (("encoder", "encoder.int8.onnx"), ("decoder", "decoder.int8.onnx"), ("tokens", "tokens.txt"))}
        missing = [str(p) for p in files.values() if not p.is_file()]
        if missing:
            raise ASRUnavailableError(f"ASR model files missing: {missing}. Run `python scripts/download_models.py` once.")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise ASRUnavailableError("sherpa-onnx is not installed. Run: pip install -r requirements.txt") from exc
        try:
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=str(files["encoder"]), decoder=str(files["decoder"]), tokens=str(files["tokens"]),
                num_threads=c.threads, decoding_method="greedy_search", language=c.language, task="transcribe")
        except Exception as exc:
            raise ASRUnavailableError(f"Could not load sherpa-onnx model from {d}: {exc}") from exc
        self.name = f"sherpa-onnx:{d.name}:int8:{c.threads}t"
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="asr")

    def _decode(self, audio: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(TARGET_SR, audio)
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1) if audio is not None else np.zeros(0, np.float32)
        if audio.size == 0:
            return ASRResult(text="", status="no_audio", error="No audio samples received")
        if sample_rate != TARGET_SR:
            return ASRResult(text="", status="error", error=f"Expected {TARGET_SR} Hz audio, got {sample_rate}")
        duration = audio.shape[0] / sample_rate
        if duration > self.config.max_audio_s:
            return ASRResult(text="", status="error", duration_s=round(duration, 2),
                             error=f"Audio longer than the {self.config.max_audio_s:g}s limit")
        if is_silent(audio, self.config.min_rms):
            return ASRResult(text="", status="silence", duration_s=round(duration, 2))
        t0 = time.perf_counter()
        fut = self._pool.submit(self._decode, preprocess(audio))
        try:
            text = fut.result(timeout=self.config.timeout_s)
        except FutureTimeout:
            return ASRResult(text="", status="error", error=f"ASR timeout after {self.config.timeout_s:g}s",
                             duration_s=round(duration, 2))
        except Exception as exc:
            return ASRResult(text="", status="error", error=f"ASR inference failed: {exc}")
        return ASRResult(text=text, language=self.config.language, duration_s=round(duration, 2),
                         status="ok" if text else "silence", latency_s=round(time.perf_counter() - t0, 3))
