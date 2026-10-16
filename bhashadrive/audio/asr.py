"""Speech recognition interfaces and the local faster-whisper implementation."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from ..config import ASRConfig
from ..schemas import ASRResult
from .preprocessing import TARGET_SR, decode_file, is_silent, preprocess


def _register_cuda_dlls() -> None:
    """On Windows, expose CUDA/cuDNN DLLs installed via the nvidia-* pip wheels to CTranslate2."""
    import os
    import sys

    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        root = base / "nvidia"
        if root.is_dir():
            for bin_dir in root.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class ASRUnavailableError(RuntimeError):
    """The local speech model could not be loaded. There is no cloud fallback."""


class SpeechRecognizer(ABC):
    name = "abstract"

    @abstractmethod
    def transcribe_array(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> ASRResult: ...

    def transcribe_file(self, path: str | Path) -> ASRResult:
        path = Path(path)
        if not path.exists() or path.stat().st_size == 0:
            return ASRResult(text="", status="no_audio", error=f"Audio file missing or empty: {path}")
        try:
            raw = decode_file(path, TARGET_SR)
        except Exception as exc:
            return ASRResult(text="", status="error", error=f"Could not decode audio: {exc}")
        return self.transcribe_array(raw, TARGET_SR)


class FasterWhisperASR(SpeechRecognizer):
    """Multilingual Whisper via CTranslate2, executed on the local machine."""

    def __init__(self, config: ASRConfig | None = None):
        self.config = c = config or ASRConfig()
        self.name = f"faster-whisper:{c.model_size}:{c.device}:{c.compute_type}"
        if c.device != "cpu":
            _register_cuda_dlls()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRUnavailableError("faster-whisper is not installed. Run: pip install -r requirements.txt") from exc
        try:
            self.model = WhisperModel(
                c.model_size,
                device=c.device,
                compute_type=c.compute_type,
                download_root=c.download_root,
                local_files_only=c.local_files_only,
            )
        except Exception as exc:
            hint = f"Run `python scripts/download_models.py --asr {c.model_size}` once with internet access."
            if c.device != "cpu":
                hint += " For GPU/CUDA errors set BD_ASR_DEVICE=cpu and BD_ASR_COMPUTE_TYPE=int8."
            raise ASRUnavailableError(f"Could not load Whisper model '{c.model_size}' on {c.device}: {exc}. {hint}") from exc

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> ASRResult:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1) if audio is not None else np.zeros(0, np.float32)
        if audio.size == 0:
            return ASRResult(text="", status="no_audio", error="No audio samples received")
        if sample_rate != TARGET_SR:
            return ASRResult(text="", status="error", error=f"Expected {TARGET_SR} Hz audio, got {sample_rate}")
        duration = audio.shape[0] / sample_rate
        if is_silent(audio, self.config.min_rms):
            return ASRResult(text="", status="silence", duration_s=round(duration, 2))
        t0 = time.perf_counter()
        audio = preprocess(audio)
        allowed_langs = self.config.allowed_languages
        try:
            segs, info = self._decode(audio, self.config.language)  # None -> automatic detection
            detected = info.language
            if self.config.language is None and allowed_langs and detected not in allowed_langs:
                # Detected an unsupported language (e.g. Hinglish misread as Arabic): re-decode with the
                # most probable supported language. Whisper uses one language per decode, so
                # code-switched speech is still transcribed in a single pass.
                probs = [(l, p) for l, p in (info.all_language_probs or []) if l in allowed_langs]
                if probs:
                    segs, info = self._decode(audio, max(probs, key=lambda x: x[1])[0])
        except Exception as exc:
            return ASRResult(text="", status="error", error=f"Whisper inference failed: {exc}")
        latency = time.perf_counter() - t0
        text = " ".join(s.text.strip() for s in segs).strip()
        return ASRResult(
            text=text,
            language=info.language,
            language_probability=round(float(info.language_probability), 3),
            duration_s=round(duration, 2),
            raw_detected_language=detected,
            segments=[
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
                 "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3)}
                for s in segs
            ],
            avg_logprob=round(float(np.mean([s.avg_logprob for s in segs])), 3) if segs else None,
            no_speech_prob=round(float(np.max([s.no_speech_prob for s in segs])), 3) if segs else None,
            status="ok" if text else "silence",
            latency_s=round(latency, 3),
        )


    def _decode(self, audio: np.ndarray, language: str | None):
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",  # transcribe, never translate
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            initial_prompt=self.config.initial_prompt,
            condition_on_previous_text=False,
        )
        return list(segments), info


class MockASR(SpeechRecognizer):
    """Returns a scripted transcript. For unit tests only."""

    name = "mock-asr"

    def __init__(self, text: str = "", status: str = "ok"):
        self.text, self.status = text, status

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = TARGET_SR) -> ASRResult:
        return ASRResult(text=self.text, status=self.status, language="en", latency_s=0.0)

    def transcribe_file(self, path: str | Path) -> ASRResult:
        return self.transcribe_array(np.zeros(1, dtype=np.float32))
