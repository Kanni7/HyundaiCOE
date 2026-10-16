"""ASR benchmark: accuracy (WER/CER), latency, peak memory, noise robustness, end-to-end intent.

  python -m bhashadrive.evaluation.eval_asr --models swift apex
  python -m bhashadrive.evaluation.eval_asr --models swift --manifest samples/real/manifest.json

Each model runs in its own subprocess so peak memory is attributable to that model.
Noisy conditions add SYNTHETIC cabin-like noise (brown noise + engine harmonics) at fixed SNRs;
this is a proxy, not a substitute for recordings made in a moving car.
Metrics: WER and CER on lower-cased, punctuation-stripped Latin text against the manifest's
`text` (Romanized references; the Hinglish models emit Romanized output).
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = [(ROOT / "scripts" / "tts_samples_v2.json", ROOT / "samples" / "tts_v2"),
             (ROOT / "scripts" / "tts_samples.json", ROOT / "samples" / "tts")]
SNRS = [None, 10, 5]
_NUM = {"twenty two": "22", "twenty three": "23", "twenty four": "24", "twenty five": "25", "twenty": "20"}


def norm_text(s: str) -> str:
    s = s.lower().replace("'", "")
    for k, v in _NUM.items():
        s = s.replace(k, v)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return d[len(b)]


def cabin_noise(n: int, sr: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    brown = np.cumsum(rng.standard_normal(n))
    brown -= np.convolve(brown, np.ones(801) / 801, mode="same")  # remove drift
    t = np.arange(n) / sr
    hum = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)) / k for k, f in enumerate((38, 76, 114, 152), 1))
    x = brown / (np.std(brown) + 1e-9) + 0.6 * hum * (1 + 0.2 * np.sin(2 * np.pi * 0.7 * t))
    return (x / (np.std(x) + 1e-9)).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, seed: int, sr: int = 16000) -> np.ndarray:
    speech = audio[np.abs(audio) > 0.02 * np.max(np.abs(audio))] if audio.size else audio
    p_s = float(np.mean(speech ** 2)) if speech.size else float(np.mean(audio ** 2))
    noise = cabin_noise(audio.size, sr, seed)
    return (audio + noise * np.sqrt(p_s / (10 ** (snr_db / 10)))).astype(np.float32)


def load_items(manifests) -> list[dict]:
    items = []
    for mf, d in manifests:
        if not Path(mf).is_file():
            continue
        for s in json.loads(Path(mf).read_text(encoding="utf-8")):
            p = Path(d) / f"{s['id']}.wav"
            if p.is_file():
                items.append({**s, "path": str(p), "provenance": s.get("provenance", "synthetic_tts_windows_onecore")})
    return items


def peak_rss_mb() -> float:
    import psutil

    mi = psutil.Process().memory_info()
    return round(getattr(mi, "peak_wset", getattr(mi, "rss", 0)) / 2**20, 1)  # Windows peak working set; else current RSS


def child(model: str, items_json: str, threads: int) -> None:
    """Runs in a subprocess: load one model, transcribe every item/condition, print JSON."""
    import soundfile as sf

    from bhashadrive.audio.preprocessing import decode_file

    items = json.loads(Path(items_json).read_text(encoding="utf-8"))
    rss0 = peak_rss_mb()
    t0 = time.perf_counter()
    if model in ("swift", "apex"):
        from bhashadrive.audio.sherpa_asr import SherpaWhisperASR
        from bhashadrive.config import SherpaASRConfig

        cfg = SherpaASRConfig()
        cfg.model_dir, cfg.threads = ROOT / "models" / "asr" / f"hi-hinglish-{model}", threads
        asr = SherpaWhisperASR(cfg)
        files = [ROOT / "models" / "asr" / f"hi-hinglish-{model}" / f for f in ("encoder.int8.onnx", "decoder.int8.onnx", "tokens.txt")]
    else:  # legacy V1 faster-whisper, e.g. "whisper-small"
        from bhashadrive.audio.asr import FasterWhisperASR
        from bhashadrive.config import ASRConfig

        c = ASRConfig()
        c.model_size, c.local_files_only = model.split("-", 1)[1], True
        asr = FasterWhisperASR(c)
        files = []
    load_s = time.perf_counter() - t0
    out = []
    for it in items:
        audio = decode_file(it["path"], 16000)
        for snr in SNRS:
            a = audio if snr is None else add_noise(audio, snr, seed=zlib.crc32(it["id"].encode()))
            t1 = time.perf_counter()
            r = asr.transcribe_array(a, 16000)
            out.append({"id": it["id"], "snr_db": snr, "hyp": r.text, "status": r.status,
                        "latency_s": round(time.perf_counter() - t1, 4), "duration_s": round(a.size / 16000, 3)})
    print(json.dumps({"model": model, "load_s": round(load_s, 2), "peak_rss_mb": peak_rss_mb(), "rss_before_load_mb": rss0,
                      "files_mb": round(sum(f.stat().st_size for f in files) / 2**20, 1) if files else None, "results": out}))


def summarise(items, run, nlu_pipe) -> dict:
    by_id = {i["id"]: i for i in items}
    res = {"load_s": run["load_s"], "peak_rss_mb": run["peak_rss_mb"], "model_files_mb": run["files_mb"], "conditions": {}}
    for snr in SNRS:
        rows = [r for r in run["results"] if r["snr_db"] == snr]
        we = wn = ce = cn = 0
        per_lang, lat, rtf, e2e = {}, [], [], []
        for r in rows:
            ref, hyp = norm_text(by_id[r["id"]]["text"]), norm_text(r["hyp"])
            w = edit_distance(ref.split(), hyp.split())
            c = edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
            we, wn, ce, cn = we + w, wn + len(ref.split()), ce + c, cn + len(ref.replace(" ", ""))
            lg = by_id[r["id"]].get("lang", "english")
            pl = per_lang.setdefault(lg, [0, 0])
            pl[0] += w
            pl[1] += len(ref.split())
            lat.append(r["latency_s"])
            rtf.append(r["latency_s"] / max(r["duration_s"], 1e-3))
            exp = by_id[r["id"]].get("expected_intent", "")
            if nlu_pipe is not None and exp and ":" not in exp and exp != "teach" and r["hyp"]:
                nlu_pipe.reset()
                pr = nlu_pipe.process_text(r["hyp"])
                e2e.append(pr.intent is not None and pr.intent.value == exp)
        key = "clean" if snr is None else f"snr_{snr}db"
        res["conditions"][key] = {
            "n": len(rows), "wer": round(we / max(1, wn), 4), "cer": round(ce / max(1, cn), 4),
            "wer_by_language": {k: round(v[0] / max(1, v[1]), 4) for k, v in sorted(per_lang.items())},
            "latency_s_median": round(statistics.median(lat), 3), "latency_s_p95": round(float(np.percentile(lat, 95)), 3),
            "rtf_median": round(statistics.median(rtf), 3),
            "end_to_end_intent_accuracy": round(sum(e2e) / len(e2e), 4) if e2e else None, "end_to_end_n": len(e2e),
            "transcripts": [(r["id"], r["hyp"]) for r in rows],
        }
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["swift"], help="swift | apex | whisper-small (legacy V1)")
    ap.add_argument("--manifest", help="extra manifest JSON (e.g. real recordings); wavs next to it")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--items", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        child(args.child, args.items, args.threads)
        return 0
    manifests = list(MANIFESTS)
    if args.manifest:
        manifests.append((Path(args.manifest), Path(args.manifest).parent))
    items = load_items(manifests)
    items_path = ROOT / "reports" / "_asr_items.json"
    items_path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    try:
        from bhashadrive.pipeline_v2 import build_v2_pipeline

        nlu = build_v2_pipeline(with_asr=False, with_personalization=False)
    except Exception as exc:  # NLU artifacts not trained yet: report ASR only
        print(f"NLU unavailable for end-to-end check: {exc}", file=sys.stderr)
        nlu = None
    import psutil

    report = {"hardware": f"{platform.processor()} | {psutil.cpu_count()} logical CPUs | "
                          f"{psutil.virtual_memory().total / 2**30:.1f} GiB | {platform.platform()}",
              "threads": args.threads, "n_items": len(items),
              "provenance": sorted({i["provenance"] for i in items}),
              "metric": "WER/CER vs Romanized reference after lower-casing and punctuation removal", "models": {}}
    for m in args.models:
        proc = subprocess.run([sys.executable, "-m", "bhashadrive.evaluation.eval_asr", "--child", m, "--items",
                               str(items_path), "--threads", str(args.threads)], capture_output=True, text=True,
                              encoding="utf-8", cwd=ROOT)
        line = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        if proc.returncode != 0 or not line:
            report["models"][m] = {"error": proc.stderr[-800:]}
            print(f"{m}: FAILED {proc.stderr[-400:]}")
            continue
        report["models"][m] = summarise(items, json.loads(line[-1]), nlu)
        for k, v in report["models"][m]["conditions"].items():
            print(f"{m:14s} {k:9s} WER={v['wer']:.3f} CER={v['cer']:.3f} lat_med={v['latency_s_median']}s "
                  f"p95={v['latency_s_p95']}s RTF={v['rtf_median']} e2e_intent={v['end_to_end_intent_accuracy']}")
        print(f"{m:14s} load={report['models'][m]['load_s']}s peak_rss={report['models'][m]['peak_rss_mb']}MB")
    items_path.unlink(missing_ok=True)
    (ROOT / "reports" / "asr_eval.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
