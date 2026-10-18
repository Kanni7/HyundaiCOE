"""Command-line interface.

  python -m bhashadrive.cli text "Bhai, volume thoda badha do"
  python -m bhashadrive.cli file samples/tts/s02_volume_hinglish.wav
  python -m bhashadrive.cli mic            # push-to-talk loop
  python -m bhashadrive.cli chat           # typed multi-turn conversation
  python -m bhashadrive.cli serve          # local API + test page on 127.0.0.1
  python -m bhashadrive.cli --legacy-llm text "..."   # V1 dev-only benchmark path (needs Ollama)

The default runtime is V2: sherpa-onnx ASR + ONNX encoder + trained intent heads, no LLM.
"""
from __future__ import annotations

import argparse
import json
import sys

from .config import AppConfig


def _print_v2(r, as_json: bool) -> None:
    if as_json:
        print(r.model_dump_json(indent=2, exclude_none=True))
        return
    if r.asr:
        a = r.asr
        print(f"[ASR] '{a.text}' status={a.status} t={a.latency_s}s" + (f" error={a.error}" if a.error else ""))
    p = r.prediction
    if p:
        neg = f" negation_cue={p.negation_cue!r}" if p.negation_cue else ""
        print(f"[NLU] intent={p.intent.value} score={p.score} margin={p.margin} top={p.top_k} "
              f"P(neg)={p.negation_score} P(implicit)={p.implicit_score}{neg}")
    if r.personal_match:
        m = r.personal_match
        print(f"[PER] '{m.trigger}' sim={m.similarity} margin={m.margin} accepted={m.accepted} ({m.reason})")
    print(f"[POL] {r.decision.value} ({r.reason_code}) source={r.source} params={r.parameters}")
    if r.action_result:
        print(f"[CAR] {'OK' if r.action_result.success else 'FAILED'}: {r.action_result.message}")
    print(f"[SAY] {r.response_text}")
    print(f"[T]   {r.timings_s}")


def _print(result, as_json: bool) -> None:
    if not hasattr(result, "frame"):
        return _print_v2(result, as_json)
    if as_json:
        print(result.model_dump_json(indent=2, exclude_none=True))
        return
    f = result.frame
    if result.asr:
        a = result.asr
        print(f"[ASR] '{a.text}' lang={a.language} p={a.language_probability} status={a.status} t={a.latency_s}s"
              + (f" error={a.error}" if a.error else ""))
    if f:
        print(f"[NLU] intent={f.intent.value} params={f.parameters.model_dump(mode='json', exclude_none=True)} "
              f"act={f.speech_act.value} rel={f.context_relation.value} negated={f.negated} lang={f.language_pattern}")
        print(f"      meaning: {f.meaning}")
    if result.verification:
        print(f"[CHK] {result.verification.function} / {result.verification.request_type}")
    if result.nlu_error:
        print(f"[ERR] {result.nlu_error}")
    print(f"[POL] {result.policy.decision.value} ({result.policy.reason_code})")
    if result.action:
        print(f"[CAR] {'OK' if result.action.success else 'FAILED'}: {result.action.message}")
    print(f"[SAY] {result.response_text}")
    print(f"[T]   {result.timings_s}")


def _configure(args) -> AppConfig:
    cfg = AppConfig()
    if getattr(args, "asr_model", None):
        cfg.asr.model_size = args.asr_model
    if getattr(args, "llm_model", None):
        cfg.llm.model = args.llm_model
    if getattr(args, "device", None):
        cfg.asr.device = args.device
        cfg.asr.compute_type = "float16" if args.device == "cuda" else "int8"
    return cfg


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # Devanagari on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser(prog="bhashadrive")
    ap.add_argument("--json", action="store_true", help="print full structured result")
    ap.add_argument("--legacy-llm", action="store_true", help="V1 Ollama/Qwen pipeline (development benchmark only)")
    ap.add_argument("--mock", action="store_true", help="use mock NLU/ASR pipeline (no heavy model weights required)")
    ap.add_argument("--profile", help="personalisation profile id (overrides BD_PROFILE_ID)")
    ap.add_argument("--asr-model", help="tiny | base | small | medium (overrides BD_ASR_MODEL)")
    ap.add_argument("--llm-model", help="Ollama tag, e.g. qwen2.5:1.5b (overrides BD_LLM_MODEL)")
    ap.add_argument("--device", choices=["cpu", "cuda"], help="ASR device (overrides BD_ASR_DEVICE)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("text"); p.add_argument("utterance", nargs="+")
    p = sub.add_parser("file"); p.add_argument("paths", nargs="+")
    sub.add_parser("mic")
    sub.add_parser("chat")
    p = sub.add_parser("serve"); p.add_argument("--host"); p.add_argument("--port", type=int)
    args = ap.parse_args(argv)
    cfg = _configure(args)

    if args.profile:
        cfg.runtime.profile_id = args.profile
    if args.cmd == "serve":
        from .api.server import run

        run(cfg, host=args.host, port=args.port, legacy=args.legacy_llm, mock=args.mock)
        return 0

    from .audio.asr import ASRUnavailableError
    from .nlu.base import LLMUnavailableError
    from .nlu.encoder import EncoderUnavailableError

    try:
        if args.mock:
            from .audio.asr import MockASR
            from .tests.fakes_v2 import make_pipeline

            pipe = make_pipeline()
            pipe.asr = MockASR("pause the music")
        elif args.legacy_llm:
            from .pipeline import build_default_pipeline

            pipe = build_default_pipeline(cfg, with_asr=args.cmd in ("file", "mic"))
        else:
            from .pipeline_v2 import build_v2_pipeline

            pipe = build_v2_pipeline(cfg, with_asr=args.cmd in ("file", "mic"))
    except (LLMUnavailableError, ASRUnavailableError, EncoderUnavailableError, FileNotFoundError, Exception) as exc:
        if not args.mock and not args.legacy_llm:
            print(f"[NOTE] Model weights not found ({exc}). Starting in mock mode for testing.", file=sys.stderr)
            from .audio.asr import MockASR
            from .tests.fakes_v2 import make_pipeline

            pipe = make_pipeline()
            pipe.asr = MockASR("pause the music")
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    if args.cmd == "text":
        _print(pipe.process_text(" ".join(args.utterance)), args.json)
    elif args.cmd == "file":
        for path in args.paths:
            print(f"=== {path}")
            _print(pipe.process_audio_file(path), args.json)
    elif args.cmd == "chat":
        print("Type what the driver says (Ctrl+C or 'quit' to exit, 'reset' to clear context, 'state' for car state).")
        while True:
            try:
                line = input("driver> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if line in ("quit", "exit"):
                break
            if line == "reset":
                (pipe.reset() if hasattr(pipe, "reset") else pipe.context.reset()); pipe.vehicle.reset(); continue
            if line == "state":
                print(json.dumps(pipe.vehicle.get_state(), indent=2)); continue
            if line:
                _print(pipe.process_text(line), args.json)
    elif args.cmd == "mic":
        from .audio.microphone import MicrophoneError, record_push_to_talk

        print("Push-to-talk. Ctrl+C to exit.")
        while True:
            try:
                audio = record_push_to_talk(cfg.asr.sample_rate)
            except MicrophoneError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 3
            except (EOFError, KeyboardInterrupt):
                break
            _print(pipe.process_audio_array(audio, cfg.asr.sample_rate), args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
