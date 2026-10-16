"""Calibrate (validation profiles) and test (test profiles) personalised-command matching.

  python -m bhashadrive.evaluation.eval_personalization

End-to-end through the V2 pipeline: each teach utterance is parsed and interpreted by the
trained model, confirmed with "yes", stored in SQLite (in-memory), then later utterances are
matched. Thresholds are chosen on the validation profiles only, then frozen and applied to
the test profiles. Writes artifacts/personalization.json and reports/personalization_eval.json.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time

import numpy as np

from ..config import AppConfig
from ..nlu.intent_model import IntentModel
from ..nlu.slots import negation_cue, quoted_reference
from ..personalization.matcher import MatchThresholds, PersonalCommandMatcher
from ..personalization.store import PersonalCommandStore
from ..pipeline_v2 import BhashaDrivePipelineV2
from ..schemas_v2 import DecisionV2
from ..training.train_classifier import ROOT
from ..vehicle.mock_adapter import MockVehicleAdapter

DATA = json.loads((ROOT / "data" / "nlu_v2" / "personal_eval.json").read_text(encoding="utf-8"))


def build(model: IntentModel, th: MatchThresholds) -> BhashaDrivePipelineV2:
    store = PersonalCommandStore(":memory:")
    matcher = PersonalCommandMatcher(store, model.embed, model.version, th)
    return BhashaDrivePipelineV2(model, MockVehicleAdapter(), None, AppConfig(), store, matcher, "eval")


def teach_all(pipe, profiles) -> list[dict]:
    out = []
    for p in profiles:
        r1 = pipe.process_text(p["teach"])
        ok_parse = r1.decision == DecisionV2.ASK_CONFIRMATION and r1.action is not None
        intent_ok = ok_parse and r1.action.intent.value == p["intent"]
        params_ok = ok_parse and all(getattr(r1.action.parameters, k) == v for k, v in p["params"].items())
        saved = False
        if ok_parse:
            saved = pipe.process_text("yes").decision == DecisionV2.SAVED
        out.append({"teach": p["teach"], "trigger_expected": p["trigger"], "proposal": r1.response_text,
                    "intent_ok": intent_ok, "params_ok": params_ok, "saved": saved})
    return out


def features(pipe, text: str):
    emb = pipe.model.embed([text])[0]
    c = pipe.matcher.candidates(pipe.profile_id, text, emb)
    if not c:
        return None
    return {"cid": c[0][0], "sim": c[0][2], "contained": c[0][3], "margin": c[0][2] - c[1][2] if len(c) > 1 else 1.0,
            "guard": bool(negation_cue(text) or quoted_reference(text))}


def accepts(f, th: MatchThresholds) -> bool:
    if f is None or f["guard"]:
        return False
    strong = f["sim"] >= th.min_similarity or (f["contained"] and f["sim"] >= th.min_similarity_contained)
    return strong and (f["margin"] >= th.min_margin or f["contained"])


def collect(pipe, profiles, unrelated) -> list[dict]:
    trig_to_cid = {c.trigger.lower(): c.id for c in pipe.store.list_commands(pipe.profile_id)}
    items = []
    for p in profiles:
        cid = trig_to_cid.get(p["trigger"].lower())
        for kind in ("uses", "paraphrases", "negatives"):
            for t in p[kind]:
                items.append({"text": t, "kind": kind, "expected_cid": cid if kind != "negatives" else None,
                              "f": features(pipe, t)})
    for t in unrelated:
        items.append({"text": t, "kind": "unrelated", "expected_cid": None, "f": features(pipe, t)})
    return items


def score(items, th: MatchThresholds) -> dict:
    res = {}
    for kind in ("uses", "paraphrases", "negatives", "unrelated"):
        sel = [i for i in items if i["kind"] == kind]
        if kind in ("uses", "paraphrases"):
            ok = [accepts(i["f"], th) and i["f"]["cid"] == i["expected_cid"] for i in sel]
            res[kind] = {"n": len(sel), "correct_match_rate": round(sum(ok) / max(1, len(sel)), 4),
                         "misses": [i["text"] for i, k in zip(sel, ok) if not k]}
        else:
            fa = [accepts(i["f"], th) for i in sel]
            res[kind] = {"n": len(sel), "false_acceptance_rate": round(sum(fa) / max(1, len(sel)), 4),
                         "false_acceptances": [i["text"] for i, k in zip(sel, fa) if k]}
    neg_quoted = [i for i in items if i["kind"] == "negatives" and i["f"] is not None and i["f"]["guard"]]
    res["negated_or_quoted_blocked"] = f"{sum(not accepts(i['f'], th) for i in neg_quoted)}/{len(neg_quoted)}"
    return res


def calibrate(items) -> MatchThresholds:
    best = None
    for s, sc, m in itertools.product(np.arange(0.5, 0.96, 0.01), np.arange(0.3, 0.91, 0.02), np.arange(0.0, 0.31, 0.02)):
        th = MatchThresholds(float(round(s, 2)), float(round(sc, 2)), float(round(m, 2)), True)
        r = score(items, th)
        fa = r["negatives"]["false_acceptance_rate"] + r["unrelated"]["false_acceptance_rate"]
        pos = r["uses"]["correct_match_rate"] + r["paraphrases"]["correct_match_rate"]
        # zero false acceptances first, then most positives, then the most conservative thresholds
        key = (fa == 0, pos, -fa, s, sc, m)
        if best is None or key > best[0]:
            best = (key, th)
    return best[1]


def correction_test(pipe, profiles) -> dict:
    ok_corr = ok_del = ok_alias = 0
    for p in profiles:
        trig = p["trigger"]
        new_clause = "set the temperature to 20 degrees" if p["intent"] != "climate_temperature_set" else "turn the volume down"
        r = pipe.process_text(f"change '{trig}' to {new_clause}")
        if r.decision == DecisionV2.ASK_CONFIRMATION:
            pipe.process_text("yes")
        after = pipe.process_text(trig)
        expect = "climate_temperature_set" if p["intent"] != "climate_temperature_set" else "volume_down"
        ok_corr += int(after.decision == DecisionV2.EXECUTE and after.action.intent.value == expect)
        alias = f"{trig} wala mode"
        r = pipe.process_text(f"make '{alias}' do the same as '{trig}'")
        if r.decision == DecisionV2.ASK_CONFIRMATION:
            pipe.process_text("yes")
        ok_alias += int(pipe.store.find_by_trigger("eval", alias) is not None)
        r = pipe.process_text(f"delete '{trig}'")
        if r.decision == DecisionV2.ASK_CONFIRMATION:
            pipe.process_text("yes")
        gone = pipe.process_text(trig)
        ok_del += int(gone.source != "personal" or gone.decision != DecisionV2.EXECUTE)
    n = len(profiles)
    return {"n": n, "action_retrieval_after_correction": f"{ok_corr}/{n}", "alias_added": f"{ok_alias}/{n}",
            "not_executed_after_delete": f"{ok_del}/{n}"}


def main(argv=None) -> int:
    cfg = AppConfig()
    model = IntentModel.load(cfg.runtime.root / cfg.runtime.nlu_artifacts, cfg.runtime.root)
    report = {"model_version": model.version}
    pipe_val = build(model, MatchThresholds())
    report["teach_val"] = teach_all(pipe_val, DATA["val"])
    val_items = collect(pipe_val, DATA["val"], DATA["unrelated"][: len(DATA["unrelated"]) // 2])
    th = calibrate(val_items)
    report["thresholds"] = th.__dict__
    report["val"] = score(val_items, th)

    pipe_test = build(model, th)
    report["teach_test"] = teach_all(pipe_test, DATA["test"])
    test_items = collect(pipe_test, DATA["test"], DATA["unrelated"][len(DATA["unrelated"]) // 2:])
    report["test"] = score(test_items, th)
    # Match latency (per utterance, embedding + similarity), test profiles
    lat = []
    for i in test_items:
        t0 = time.perf_counter()
        features(pipe_test, i["text"])
        lat.append((time.perf_counter() - t0) * 1000)
    report["match_latency_ms"] = {"median": round(statistics.median(lat), 2), "p95": round(float(np.percentile(lat, 95)), 2)}
    report["corrections_test"] = correction_test(pipe_test, DATA["test"])
    report["note"] = ("Synthetic phrases; thresholds chosen on validation profiles with the objective of zero false "
                      "acceptances, then frozen for the test profiles. Small sample: treat as a smoke-level estimate.")
    (ROOT / "artifacts").mkdir(exist_ok=True)
    (ROOT / "artifacts" / "personalization.json").write_text(json.dumps(th.__dict__, indent=2), encoding="utf-8")
    (ROOT / "reports" / "personalization_eval.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("thresholds", "val", "test", "match_latency_ms", "corrections_test")},
                     indent=1, ensure_ascii=False))
    print("teach:", [(t["teach"][:40], t["intent_ok"], t["params_ok"], t["saved"]) for t in report["teach_val"] + report["teach_test"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
