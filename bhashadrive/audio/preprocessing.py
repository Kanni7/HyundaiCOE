"""Audio loading and light preprocessing (mono, 16 kHz, float32, peak-normalised)."""
from __future__ import annotations

import numpy as np

TARGET_SR = 16000


class AudioError(RuntimeError):
    pass


def _resample(audio: np.ndarray, sr: int, target: int) -> np.ndarray:
    if sr == target or audio.size == 0:
        return audio
    # Light anti-alias smoothing then linear interpolation; adequate for speech recognition.
    if sr > target:
        k = max(1, int(round(sr / target)))
        if k > 1:
            audio = np.convolve(audio, np.ones(k, dtype=np.float32) / k, mode="same")
    n_out = int(round(audio.shape[0] * target / sr))
    x_old = np.linspace(0.0, 1.0, audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def decode_file(path, sample_rate: int = TARGET_SR) -> np.ndarray:
    """Decode an audio file to mono float32 at `sample_rate`.

    Uses libsndfile (wav/flac/ogg) first, then PyAV for compressed formats (mp3/m4a/webm).
    Our own PyAV call avoids a keyword that newer PyAV releases removed.
    """
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return _resample(data.mean(axis=1), sr, sample_rate)
    except Exception:
        pass
    try:
        import av
    except ImportError as exc:
        raise AudioError("PyAV is required for compressed audio formats") from exc
    try:
        with av.open(str(path), mode="r") as container:
            resampler = av.AudioResampler(format="flt", layout="mono", rate=sample_rate)
            chunks = []
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame) or []:
                    chunks.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None) or []:
                chunks.append(out.to_ndarray().reshape(-1))
    except Exception as exc:
        raise AudioError(f"Could not decode audio file {path}: {exc}") from exc
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def preprocess(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size:
        audio = audio - float(np.mean(audio))  # remove DC offset
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-6:
            audio = audio / peak * 0.95
    return audio


def rms(audio: np.ndarray) -> float:
    audio = np.asarray(audio, dtype=np.float32)
    return float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0


def is_silent(raw_audio: np.ndarray, min_rms: float) -> bool:
    """Silence check on the raw (pre-normalisation) signal."""
    return np.asarray(raw_audio).size == 0 or rms(raw_audio) < min_rms
