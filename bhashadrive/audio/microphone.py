"""Push-to-talk microphone capture (press Enter to start, Enter to stop)."""
from __future__ import annotations

import queue
import threading

import numpy as np

from .preprocessing import TARGET_SR


class MicrophoneError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
    except Exception as exc:
        raise MicrophoneError(f"sounddevice/PortAudio unavailable: {exc}. Run: pip install sounddevice") from exc
    return sd


def record_push_to_talk(sample_rate: int = TARGET_SR, max_seconds: float = 15.0) -> np.ndarray:
    sd = _sd()
    input("Press Enter to START speaking...")
    chunks: queue.Queue = queue.Queue()
    stop = threading.Event()

    def on_audio(indata, frames, time_info, status):
        chunks.put(indata.copy())

    def wait_for_enter():
        input()
        stop.set()

    try:
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=on_audio):
            print(f"Recording... press Enter to STOP (max {max_seconds:g}s).")
            threading.Thread(target=wait_for_enter, daemon=True).start()
            stop.wait(timeout=max_seconds)
    except Exception as exc:
        raise MicrophoneError(f"Could not open microphone: {exc}") from exc

    data = []
    while not chunks.empty():
        data.append(chunks.get())
    return np.concatenate(data, axis=0).reshape(-1) if data else np.zeros(0, dtype=np.float32)


def record_fixed(seconds: float, sample_rate: int = TARGET_SR) -> np.ndarray:
    sd = _sd()
    try:
        audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        raise MicrophoneError(f"Could not record from microphone: {exc}") from exc
    return audio.reshape(-1)
