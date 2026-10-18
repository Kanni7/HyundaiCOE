"""Runtime configuration.

All settings come from environment variables (optionally loaded from a `.env`
file in the working directory). Nothing machine-specific is hardcoded, so model
size, device, endpoint and limits can be changed without editing source files.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional
    pass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


DEFAULT_INCAR_PROMPT = (
    "Bhai, cabin thoda thanda kar do, cabin thoda warm kar do, cabin cooling badhao, "
    "AC thoda thanda karo, AC warm karo, AC on karo, AC band kar do, AC chala do, "
    "temperature 22 degree pe set karo, fan speed badha do, fan speed kam karo, blower tez karo, hawa dheemi karo, "
    "volume badha do, volume kam karo, awaaz badhao, awaaz dheemi karo, mute karo, "
    "gaana chalao, gaana roko, music pause karo, next song, previous song, "
    "thand lag rahi hai, garmi lag rahi hai, bahut thand hai, bahut garmi hai, "
    "navigate to Connaught Place, Delhi le chalo, rasta dikhao."
)


@dataclass
class ASRConfig:
    model_size: str = field(default_factory=lambda: _env("BD_ASR_MODEL", "base"))
    device: str = field(default_factory=lambda: _env("BD_ASR_DEVICE", "cpu"))  # cpu | cuda | auto
    compute_type: str = field(default_factory=lambda: _env("BD_ASR_COMPUTE_TYPE", "int8"))
    beam_size: int = field(default_factory=lambda: _env_int("BD_ASR_BEAM_SIZE", 1))  # 1 = greedy decode for low latency
    cpu_threads: int = field(default_factory=lambda: _env_int("BD_ASR_THREADS", 4))
    language: str | None = field(default_factory=lambda: _env("BD_ASR_LANGUAGE", "") or None)  # None = auto-detect among allowed_languages
    # Auto-detection is restricted to these languages; empty = any language Whisper knows.
    allowed_languages: tuple = field(default_factory=lambda: tuple(
        x.strip() for x in _env("BD_ASR_ALLOWED_LANGUAGES", "en,hi").split(",") if x.strip()))
    vad_filter: bool = field(default_factory=lambda: _env_bool("BD_ASR_VAD", True))
    vad_parameters: dict = field(default_factory=lambda: {"min_silence_duration_ms": 600, "threshold": 0.35, "speech_pad_ms": 400})
    no_speech_threshold: float = field(default_factory=lambda: _env_float("BD_ASR_NO_SPEECH_TH", 0.6))
    initial_prompt: str | None = field(default_factory=lambda: os.environ.get("BD_ASR_INITIAL_PROMPT", DEFAULT_INCAR_PROMPT))
    download_root: str | None = field(default_factory=lambda: os.environ.get("BD_ASR_MODEL_DIR") or None)
    # After the one-time download, keep this true so startup never touches the network.
    local_files_only: bool = field(default_factory=lambda: _env_bool("BD_ASR_LOCAL_ONLY", True))
    sample_rate: int = 16000
    min_rms: float = field(default_factory=lambda: _env_float("BD_ASR_MIN_RMS", 0.003))


@dataclass
class LLMConfig:
    backend: str = field(default_factory=lambda: _env("BD_LLM_BACKEND", "ollama"))
    model: str = field(default_factory=lambda: _env("BD_LLM_MODEL", "qwen2.5:3b"))
    endpoint: str = field(default_factory=lambda: _env("BD_LLM_ENDPOINT", "http://127.0.0.1:11434"))
    timeout_s: float = field(default_factory=lambda: _env_float("BD_LLM_TIMEOUT_S", 30.0))
    temperature: float = field(default_factory=lambda: _env_float("BD_LLM_TEMPERATURE", 0.0))
    num_ctx: int = field(default_factory=lambda: _env_int("BD_LLM_NUM_CTX", 4096))
    keep_alive: str = field(default_factory=lambda: _env("BD_LLM_KEEP_ALIVE", "30m"))
    # Only loopback endpoints are allowed unless explicitly overridden.
    allow_remote_endpoint: bool = field(default_factory=lambda: _env_bool("BD_LLM_ALLOW_REMOTE", False))


@dataclass
class VehicleLimits:
    temp_min_c: float = field(default_factory=lambda: _env_float("BD_TEMP_MIN_C", 16.0))
    temp_max_c: float = field(default_factory=lambda: _env_float("BD_TEMP_MAX_C", 30.0))
    fan_min: int = field(default_factory=lambda: _env_int("BD_FAN_MIN", 0))
    fan_max: int = field(default_factory=lambda: _env_int("BD_FAN_MAX", 7))
    volume_min: int = field(default_factory=lambda: _env_int("BD_VOLUME_MIN", 0))
    volume_max: int = field(default_factory=lambda: _env_int("BD_VOLUME_MAX", 30))
    # Step sizes for qualitative amounts.
    temp_steps: dict = field(default_factory=lambda: {"small": 1.0, "medium": 2.0, "large": 3.0})
    fan_steps: dict = field(default_factory=lambda: {"small": 1, "medium": 2, "large": 3})
    volume_steps: dict = field(default_factory=lambda: {"small": 2, "medium": 4, "large": 8})
    max_destination_chars: int = 120


@dataclass
class ContextConfig:
    max_turns: int = field(default_factory=lambda: _env_int("BD_CONTEXT_MAX_TURNS", 6))
    pending_ttl_s: float = field(default_factory=lambda: _env_float("BD_PENDING_TTL_S", 60.0))
    followup_ttl_s: float = field(default_factory=lambda: _env_float("BD_FOLLOWUP_TTL_S", 120.0))


_ROOT = Path(__file__).resolve().parents[1]


def _abs(p: str) -> Path:
    """Relative paths in settings are resolved against the project root, not the CWD."""
    return Path(p) if Path(p).is_absolute() else _ROOT / p


@dataclass
class SherpaASRConfig:
    """V2 on-device ASR (sherpa-onnx Whisper export). Only local files are read."""

    model_dir: Path = field(default_factory=lambda: _abs(_env("BD_SHERPA_MODEL_DIR", "models/asr/hi-hinglish-swift")))
    language: str = field(default_factory=lambda: _env("BD_SHERPA_LANGUAGE", "hi"))
    threads: int = field(default_factory=lambda: _env_int("BD_ASR_THREADS", 4))
    timeout_s: float = field(default_factory=lambda: _env_float("BD_ASR_TIMEOUT_S", 10.0))
    max_audio_s: float = field(default_factory=lambda: _env_float("BD_MAX_AUDIO_S", 15.0))
    min_rms: float = field(default_factory=lambda: _env_float("BD_ASR_MIN_RMS", 0.003))


@dataclass
class RuntimeConfig:
    """V2 semantic runtime: trained NLU artifacts, personalisation storage."""

    root: Path = field(default_factory=lambda: Path(_env("BD_ROOT", str(_ROOT))))
    nlu_artifacts: str = field(default_factory=lambda: _env("BD_NLU_ARTIFACTS", "artifacts/nlu/runtime"))
    nlu_threads: int = field(default_factory=lambda: _env_int("BD_NLU_THREADS", 2))
    profile_db: str = field(default_factory=lambda: _env("BD_PROFILE_DB", "data/profiles/bhashadrive.sqlite"))
    profile_id: str = field(default_factory=lambda: _env("BD_PROFILE_ID", "default"))
    personal_thresholds: str = field(default_factory=lambda: _env("BD_PERSONAL_THRESHOLDS", "artifacts/personalization.json"))
    # Opt-in: keep an audit trail of old -> new mappings when the user corrects a shortcut.
    store_corrections: bool = field(default_factory=lambda: _env_bool("BD_STORE_CORRECTIONS", False))


@dataclass
class AppConfig:
    asr: ASRConfig = field(default_factory=ASRConfig)
    sherpa: SherpaASRConfig = field(default_factory=SherpaASRConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    limits: VehicleLimits = field(default_factory=VehicleLimits)
    context: ContextConfig = field(default_factory=ContextConfig)
    store_audio: bool = field(default_factory=lambda: _env_bool("BD_STORE_AUDIO", False))
    audio_store_dir: Path = field(default_factory=lambda: Path(_env("BD_AUDIO_STORE_DIR", "recordings")))
    api_host: str = field(default_factory=lambda: _env("BD_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("BD_API_PORT", 8000))
