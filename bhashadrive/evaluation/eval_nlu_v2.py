"""Evaluate trained V2 intent models (text-only NLU) and compare with baselines.

  python -m bhashadrive.evaluation.eval_nlu_v2 --models minilm_onnx_int8 minilm-ft_onnx_int8
  python -m bhashadrive.evaluation.eval_nlu_v2 --models minilm-ft_onnx_int8 --promote minilm-ft_onnx_int8

Test sets (never used for training or calibration): `test` (template-group held-out),
`heldout_unseen` (hand-written paraphrases), `v1_mapped` (V1 sets, written before V2).
Writes reports/nlu_eval_v2.json, reports/nlu_eval_v2.md and per-model confusion matrices.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

from ..config import VehicleLimits
from ..nlu.intent_model import IntentModel
from ..nlu.slots import extract_amount, extract_destination, extract_temperature
from ..nlu.taxonomy import EXECUTABLE_INTENTS, IntentV2
from ..policy.policy_v2 import PolicyV2
from ..schemas_v2 import DecisionV2
from ..training.train_classifier import CLASSES, ROOT, load_split
from . import keyword_baseline_v2

TEST_SETS = ("test", "heldout_unseen", "v1_mapped")
EXEC = {i.value for i in EXECUTABLE_INTENTS}


def prf(gold: list[str], pred: list[str]) -> dict:
    labels = sorted(set(gold) | set(pred))
    per, f1s = {}, []
    for c in labels:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        if c in gold:
            f1s.append(f1)
        per[c] = {"precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "support": gold.count(c)}
    acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
    return {"n": len(gold), "accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4), "per_intent": per}


def norm_dest(s: str | None) -> str:
    s = (s or "").casefold()
    for w in ("the ", "nearest ", "closest ", "a ", "नज़दीकी "):
        s = s.replace(w, "")
    return " ".join(s.replace("?", "").split())


def gold_decision(r: dict) -> str:
    return r["decision"]


def eval_model(key: str, policy_limits: VehicleLimits) -> dict:
    model = IntentModel.load(ROOT / "artifacts" / "nlu" / key, ROOT, threads=2)
    m = model.manifest
    policy = PolicyV2(policy_limits, m["thresholds"], m.get("negation_threshold", 0.5), m.get("implicit_threshold", 0.5))
    out: dict = {"model_version": m["model_version"], "encoder": m["encoder"], "thresholds": m["thresholds"], "sets": {}}
    for split in TEST_SETS:
        rows = load_split(split)
        texts = [r["text"] for r in rows]
        X = model.embed(texts)
        preds = [model.predict_embedding(t, x) for t, x in zip(texts, X)]
        gold, pi = [r["intent"] for r in rows], [p.intent.value for p in preds]
        res = prf(gold, pi)
        # by language pattern
        by_lang = defaultdict(list)
        for r, p in zip(rows, pi):
            by_lang[r["language_pattern"]].append(r["intent"] == p)
        res["accuracy_by_language"] = {k: {"n": len(v), "accuracy": round(sum(v) / len(v), 3)} for k, v in sorted(by_lang.items())}
        # decisions & safety
        dec = [policy.decide(t, p) for t, p in zip(texts, preds)]
        counts = Counter()
        failures = []
        for r, p, d in zip(rows, preds, dec):
            g = gold_decision(r)
            if g == "context_dependent":
                continue
            counts["n"] += 1
            ours = d.decision.value
            if ours == "execute":
                counts["executed"] += 1
                wrong_intent = d.action.intent.value != r["intent"]
                gp = r["parameters"]
                wrong_param = ("target_temperature_celsius" in gp and d.action.parameters.target_temperature_celsius != gp["target_temperature_celsius"])
                if g != "execute" or wrong_intent or wrong_param:
                    counts["unsafe_execution"] += 1
                    counts[f"unsafe_execution_gold_{g}"] += 1
                    failures.append({"text": r["text"], "gold_intent": r["intent"], "gold_decision": g, "pred": p.intent.value,
                                     "score": p.score, "decision": ours, "kind": "UNSAFE_EXECUTION"})
            if g == "execute":
                counts["gold_execute"] += 1
                counts["gold_execute_executed_correctly"] += int(ours == "execute" and d.action.intent.value == r["intent"])
            if r["negated"]:
                counts["negated"] += 1
                counts["negated_blocked"] += int(ours != "execute")
            if r["intent"] == "unsupported":
                counts["unsupported"] += 1
                counts["unsupported_rejected"] += int(ours == "reject")
            if r["intent"] == "ambiguous":
                counts["ambiguous"] += 1
                counts["ambiguous_not_executed"] += int(ours != "execute")
                counts["ambiguous_clarified"] += int(ours == "clarify")
            if r["speech_act"] == "implicit_request" and not r["negated"]:
                counts["implicit"] += 1
                counts["implicit_confirmed_not_executed"] += int(ours == "ask_confirmation")
            if ours == g:
                counts["decision_correct"] += 1
            elif len(failures) < 400 and not (failures and failures[-1]["text"] == r["text"]):
                failures.append({"text": r["text"], "gold_intent": r["intent"], "gold_decision": g, "pred": p.intent.value,
                                 "score": p.score, "decision": ours, "reason": d.reason_code})
        c = counts
        res["decision"] = {
            "n": c["n"], "decision_accuracy": round(c["decision_correct"] / max(1, c["n"]), 4),
            "unsafe_execution_rate": round(c["unsafe_execution"] / max(1, c["executed"]), 4),
            "unsafe_executions": c["unsafe_execution"],
            "unsafe_by_gold_decision": {k.split("gold_")[1]: v for k, v in c.items() if k.startswith("unsafe_execution_gold_")},
            "correct_execution_recall": round(c["gold_execute_executed_correctly"] / max(1, c["gold_execute"]), 4),
            "negated_blocked": f"{c['negated_blocked']}/{c['negated']}",
            "unsupported_rejected": f"{c['unsupported_rejected']}/{c['unsupported']}",
            "ambiguous_not_executed": f"{c['ambiguous_not_executed']}/{c['ambiguous']}",
            "ambiguous_clarified": f"{c['ambiguous_clarified']}/{c['ambiguous']}",
            "implicit_asked_confirmation": f"{c['implicit_confirmed_not_executed']}/{c['implicit']}",
        }
        res["failures"] = failures[:60]
        # confusion matrix
        idx = {c_: i for i, c_ in enumerate(CLASSES)}
        cm = np.zeros((len(CLASSES), len(CLASSES)), int)
        for g_, p_ in zip(gold, pi):
            cm[idx[g_], idx[p_]] += 1
        res["confusion_matrix"] = {"labels": CLASSES, "matrix": cm.tolist()}
        out["sets"][split] = res
    # latency (single utterance, as at runtime)
    texts = [r["text"] for r in load_split("heldout_unseen")]
    for t in texts[:10]:
        model.predict(t)
    lat = []
    for t in texts:
        t0 = time.perf_counter()
        model.predict(t)
        lat.append((time.perf_counter() - t0) * 1000)
    out["latency_ms"] = {"n": len(lat), "median": round(statistics.median(lat), 2),
                         "p95": round(float(np.percentile(lat, 95)), 2), "threads": 2}
    return out


def extractor_accuracy() -> dict:
    """Slot extraction given the gold intent, on all evaluation rows with gold parameters."""
    res = {}
    rows = [r for s in TEST_SETS for r in load_split(s)]
    t = [r for r in rows if "target_temperature_celsius" in r["parameters"] and not r["negated"]]
    res["temperature"] = {"n": len(t), "accuracy": round(sum(extract_temperature(r["text"]) == r["parameters"]["target_temperature_celsius"] for r in t) / max(1, len(t)), 4)}
    d = [r for r in rows if r["parameters"].get("destination") and not r["negated"]]
    ok = [norm_dest(extract_destination(r["text"])) == norm_dest(r["parameters"]["destination"]) for r in d]
    res["destination"] = {"n": len(d), "exact_match_normalised": round(sum(ok) / max(1, len(d)), 4),
                          "errors": [(r["text"], extract_destination(r["text"]), r["parameters"]["destination"]) for r, k in zip(d, ok) if not k][:15]}
    a = [r for r in load_split("heldout_unseen") if r["parameters"].get("amount")]
    res["amount_heldout"] = {"n": len(a), "accuracy": round(sum(extract_amount(r["text"]) == r["parameters"]["amount"] for r in a) / max(1, len(a)), 4)}
    return res


def baselines() -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tr = load_split("train")
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform([r["text"].lower() for r in tr])
    clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(Xtr, [r["intent"] for r in tr])
    out = {}
    for split in TEST_SETS:
        rows = load_split(split)
        gold = [r["intent"] for r in rows]
        kw = [keyword_baseline_v2.predict(r["text"]) for r in rows]
        tf = list(clf.predict(vec.transform([r["text"].lower() for r in rows])))
        out[split] = {"keyword_rules": {k: v for k, v in prf(gold, kw).items() if k != "per_intent"},
                      "tfidf_char_logreg": {k: v for k, v in prf(gold, tf).items() if k != "per_intent"}}
    return out


def to_markdown(rep: dict) -> str:
    L = ["# V2 NLU evaluation (text only, synthetic data)", "",
         f"Hardware: {rep['hardware']}", "",
         "| model | set | n | accuracy | macro-F1 | decision acc | unsafe exec | correct-exec recall | negated blocked | unsupported rejected | ambiguous not executed |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, r in rep["models"].items():
        for s, v in r["sets"].items():
            d = v["decision"]
            L.append(f"| {key} | {s} | {v['n']} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | {d['decision_accuracy']:.3f} | "
                     f"{d['unsafe_executions']} ({d['unsafe_execution_rate']:.1%}) | {d['correct_execution_recall']:.3f} | "
                     f"{d['negated_blocked']} | {d['unsupported_rejected']} | {d['ambiguous_not_executed']} |")
    L += ["", "| baseline | set | accuracy | macro-F1 |", "|---|---|---|---|"]
    for s, b in rep["baselines"].items():
        for name, v in b.items():
            L.append(f"| {name} | {s} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} |")
    L += ["", "Latency (single utterance, encoder + heads, 2 threads):", ""]
    for key, r in rep["models"].items():
        L.append(f"- {key}: median {r['latency_ms']['median']} ms, p95 {r['latency_ms']['p95']} ms")
    L += ["", f"Parameter extraction (gold intent): {json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'errors'} for k, v in rep['extraction'].items()})}"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="artifact keys under artifacts/nlu/")
    ap.add_argument("--promote", help="copy this artifact dir to artifacts/nlu/runtime")
    args = ap.parse_args(argv)
    proc = psutil.Process()
    rep = {"hardware": f"{platform.processor()} | {psutil.cpu_count()} logical CPUs | "
                       f"{psutil.virtual_memory().total / 2**30:.1f} GiB RAM | {platform.platform()} | Python {platform.python_version()}",
           "models": {}, "note": "All evaluation text is synthetic; see data/nlu_v2 provenance fields."}
    for key in args.models:
        rss0 = proc.memory_info().rss
        rep["models"][key] = eval_model(key, VehicleLimits())
        rep["models"][key]["process_rss_mb_after_load"] = round(proc.memory_info().rss / 2**20, 1)
        rep["models"][key]["rss_increase_mb"] = round((proc.memory_info().rss - rss0) / 2**20, 1)
        for s, v in rep["models"][key]["sets"].items():
            print(f"{key:24s} {s:15s} n={v['n']:4d} acc={v['accuracy']:.3f} macroF1={v['macro_f1']:.3f} "
                  f"unsafe={v['decision']['unsafe_executions']} recall_exec={v['decision']['correct_execution_recall']:.3f}")
    rep["extraction"] = extractor_accuracy()
    rep["baselines"] = baselines()
    for s, b in rep["baselines"].items():
        print(f"baselines {s:15s} " + "  ".join(f"{k}: acc={v['accuracy']:.3f} F1={v['macro_f1']:.3f}" for k, v in b.items()))
    out = ROOT / "reports" / "nlu_eval_v2.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    (ROOT / "reports" / "nlu_eval_v2.md").write_text(to_markdown(rep), encoding="utf-8")
    if args.promote:
        src, dst = ROOT / "artifacts" / "nlu" / args.promote, ROOT / "artifacts" / "nlu" / "runtime"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        print(f"promoted {args.promote} -> {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
