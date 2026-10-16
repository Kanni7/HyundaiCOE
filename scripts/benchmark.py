"""Benchmark local inference configurations. Reports measured numbers only.

  python scripts/benchmark.py                                   # default matrix
  python scripts/benchmark.py --asr small:cpu base:cpu --llm qwen2.5:3b qwen2.5:1.5b
  python scripts/benchmark.py --skip-eval                       # latency only

Measures: ASR latency per audio sample, NLU (parser + verifier) latency and accuracy on the
labelled dataset, end-to-end latency on audio samples, process RSS, Ollama runner RSS and
GPU memory (nvidia-smi) where available.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import psutil  # noqa: E402

from bhashadrive.config import AppConfig  # noqa: E402

SAMPLES = ROOT / "samples" / "tts"


def gpu_mem_used_mib() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def ollama_rss_mib() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        if p.info["name"] and "ollama" in p.info["name"].lower() and p.info["memory_info"]:
            total += p.info["memory_info"].rss
    return round(total / 2**20, 1)


def rss_mib() -> float:
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def stats(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return {"n": len(xs), "mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
            "p90": round(xs[int(0.9 * (len(xs) - 1))], 3), "max": round(xs[-1], 3)}


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
            "cpu_cores_logical": psutil.cpu_count(), "ram_gib": round(psutil.virtual_memory().total / 2**30, 1)}
    exe = shutil.which("nvidia-smi")
    if exe:
        info["gpu"] = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True).stdout.strip()
    return info


def bench_asr(size: str, device: str, samples: list[dict]) -> dict:
    from bhashadrive.audio.asr import FasterWhisperASR, ASRUnavailableError

    cfg = AppConfig().asr
    cfg.model_size, cfg.device = size, device
    cfg.compute_type = "int8_float16" if device == "cuda" else "int8"
    gpu0, rss0 = gpu_mem_used_mib(), rss_mib()
    t0 = time.perf_counter()
    try:
        asr = FasterWhisperASR(cfg)
    except ASRUnavailableError as exc:
        return {"config": f"{size}:{device}", "error": str(exc)}
    load_s = time.perf_counter() - t0
    asr.transcribe_file(samples[0]["path"])  # warm-up, excluded
    rows, lat = [], []
    for s in samples:
        r = asr.transcribe_file(s["path"])
        lat.append(r.latency_s or 0)
        rows.append({"id": s["id"], "reference": s["text"], "transcript": r.text, "language": r.language,
                     "raw_detected": r.raw_detected_language, "latency_s": r.latency_s, "audio_s": r.duration_s})
    out = {"config": f"{size}:{device}:{cfg.compute_type}", "load_s": round(load_s, 2), "latency_s": stats(lat),
           "rss_delta_mib": round(rss_mib() - rss0, 1), "gpu_delta_mib": (gpu_mem_used_mib() - gpu0) if gpu0 is not None else None,
           "samples": rows}
    out["_asr"] = asr
    return out


def bench_llm(model: str, skip_eval: bool) -> dict:
    from bhashadrive.evaluation.evaluate import load_dataset, run
    from bhashadrive.nlu.base import LLMError
    from bhashadrive.nlu.local_llm import OllamaLLM
    from bhashadrive.nlu.semantic_parser import LLMSemanticParser

    cfg = AppConfig()
    cfg.llm.model = model
    try:
        llm = OllamaLLM(cfg.llm)
        llm.health_check()
    except LLMError as exc:
        return {"model": model, "error": str(exc)}
    gpu0 = gpu_mem_used_mib()
    load_s = llm.warmup()
    out = {"model": model, "info": llm.model_info(), "load_s": round(load_s, 2),
           "ollama_rss_mib": ollama_rss_mib(), "gpu_used_after_load_mib": gpu_mem_used_mib(), "gpu_used_before_mib": gpu0}
    try:
        out["ollama_ps"] = subprocess.run([shutil.which("ollama") or "ollama", "ps"], capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    parser = LLMSemanticParser(llm)
    if not skip_eval:
        rep = run(parser, load_dataset(), cfg)
        out["eval"] = rep["summary"]
        out["_rows"] = rep["rows"]
    out["_parser"] = parser
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr", nargs="*", default=["small:cpu", "base:cpu"], help="size:device")
    ap.add_argument("--llm", nargs="*", default=["qwen2.5:3b", "qwen2.5:1.5b"])
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "reports" / "benchmark.json"))
    args = ap.parse_args()

    samples = json.loads((ROOT / "scripts" / "tts_samples.json").read_text(encoding="utf-8"))
    samples = [dict(s, path=SAMPLES / f"{s['id']}.wav") for s in samples if (SAMPLES / f"{s['id']}.wav").exists()]
    if not samples:
        print("No audio samples. Run scripts/make_tts_samples.ps1 (Windows) or record your own into samples/tts.")
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"), "hardware": hardware(),
              "audio_samples": len(samples), "audio_samples_note": "synthetic TTS (Windows OneCore voices), not real drivers",
              "asr": [], "llm": [], "end_to_end": []}

    llm_results = []
    for model in args.llm:
        print(f"LLM {model} ...", flush=True)
        r = bench_llm(model, args.skip_eval)
        llm_results.append(r)
        report["llm"].append({k: v for k, v in r.items() if not k.startswith("_")})
        print(json.dumps(report["llm"][-1].get("eval", report["llm"][-1]), indent=1, ensure_ascii=False)[:1500], flush=True)

    for spec in args.asr:
        size, device = (spec.split(":") + ["cpu"])[:2]
        print(f"ASR {size} on {device} ...", flush=True)
        a = bench_asr(size, device, samples) if samples else {"config": spec, "error": "no samples"}
        asr = a.pop("_asr", None)
        report["asr"].append(a)
        print(json.dumps({k: v for k, v in a.items() if k != "samples"}, ensure_ascii=False), flush=True)
        if asr is None:
            continue
        # End-to-end: audio -> ASR -> parser + verifier -> policy -> mock vehicle, for each LLM.
        from bhashadrive.pipeline import BhashaDrivePipeline
        from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

        for lr in llm_results:
            if "_parser" not in lr:
                continue
            e2e, rows = [], []
            for s in samples:
                pipe = BhashaDrivePipeline(lr["_parser"], MockVehicleAdapter(), asr, AppConfig())
                res = pipe.process_audio_file(s["path"])
                e2e.append(res.timings_s.get("end_to_end_s", res.timings_s.get("asr_s", 0)))
                got = res.policy.command.intent.value if res.policy.command else (res.frame.intent.value if res.frame else "none")
                rows.append({"id": s["id"], "transcript": res.transcript, "expected_intent": s["expected_intent"],
                             "pred_intent": got, "decision": res.policy.decision.value, "timings_s": res.timings_s})
            ok = sum(r["pred_intent"] == r["expected_intent"] or (r["expected_intent"] == "none" and r["decision"] == "clarify_or_reject") for r in rows)
            report["end_to_end"].append({"asr": a["config"], "llm": lr["model"], "latency_s": stats(e2e),
                                         "intent_correct": f"{ok}/{len(rows)}", "rows": rows})
            print(f"  e2e {a['config']} + {lr['model']}: {stats(e2e)} intent {ok}/{len(rows)}", flush=True)
        del asr
        gc.collect()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
