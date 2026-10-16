"""Validate the local setup: packages, Whisper weights, Ollama runtime and model, GPU.

  python scripts/check_setup.py            # checks only, no downloads
  python scripts/check_setup.py --infer    # also runs one ASR and one LLM inference
"""
from __future__ import annotations

import argparse
import importlib
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bhashadrive.config import AppConfig  # noqa: E402

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer", action="store_true")
    args = ap.parse_args()
    cfg, failures = AppConfig(), 0
    print(f"Python {platform.python_version()} on {platform.platform()}")
    try:
        import psutil

        print(f"RAM {psutil.virtual_memory().total / 2**30:.1f} GiB, CPU cores {psutil.cpu_count()}")
    except ImportError:
        pass

    for mod in ("faster_whisper", "ctranslate2", "pydantic", "httpx", "fastapi", "uvicorn", "numpy", "soundfile", "sounddevice"):
        try:
            m = importlib.import_module(mod)
            print(f"{OK} {mod} {getattr(m, '__version__', '')}")
        except Exception as exc:
            failures += mod not in ("sounddevice",)
            print(f"{BAD if mod != 'sounddevice' else WARN} {mod}: {exc}")

    try:
        import ctranslate2

        n = ctranslate2.get_cuda_device_count()
        print(f"{OK if n else WARN} CUDA devices visible to CTranslate2: {n} (ASR device configured: {cfg.asr.device})")
    except Exception as exc:
        print(f"{WARN} CUDA check failed: {exc}")

    from bhashadrive.audio.asr import ASRUnavailableError, FasterWhisperASR

    asr = None
    try:
        asr = FasterWhisperASR(cfg.asr)
        print(f"{OK} Whisper '{cfg.asr.model_size}' loaded from local cache ({cfg.asr.device}/{cfg.asr.compute_type})")
    except ASRUnavailableError as exc:
        failures += 1
        print(f"{BAD} {exc}")

    from bhashadrive.nlu.base import LLMError
    from bhashadrive.nlu.local_llm import OllamaLLM

    llm = None
    try:
        llm = OllamaLLM(cfg.llm)
        llm.health_check()
        print(f"{OK} Ollama at {cfg.llm.endpoint} with model {llm.model_info()}")
    except LLMError as exc:
        failures += 1
        llm = None
        print(f"{BAD} {exc}")

    if args.infer:
        if asr:
            import numpy as np

            t = np.linspace(0, 1, 16000, dtype=np.float32)
            r = asr.transcribe_array(0.1 * np.sin(2 * np.pi * 220 * t))
            print(f"{OK} ASR inference ran (tone input -> status={r.status})")
        if llm:
            from bhashadrive.nlu.semantic_parser import LLMSemanticParser

            llm.warmup()
            f = LLMSemanticParser(llm).parse("volume thoda badha do", "Pending proposal awaiting the driver's yes/no: none\nLast action executed: none")
            print(f"{OK} LLM inference ran -> intent={f.intent.value} params={f.parameters.model_dump(mode='json', exclude_none=True)}")

    print("\nSetup OK" if not failures else f"\n{failures} problem(s) found. See README 'Troubleshooting'.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
