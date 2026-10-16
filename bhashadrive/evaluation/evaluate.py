"""Repeatable evaluation of the text-level pipeline against dataset.jsonl.

Examples:
  python -m bhashadrive.evaluation.evaluate                       # local LLM (BD_LLM_MODEL)
  python -m bhashadrive.evaluation.evaluate --model qwen2.5:1.5b
  python -m bhashadrive.evaluation.evaluate --baseline            # keyword baseline (comparison only)
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ..config import AppConfig
from ..nlu.context_manager import ConversationContext
from ..pipeline import BhashaDrivePipeline
from ..schemas import Command, Decision
from ..vehicle.mock_adapter import MockVehicleAdapter

DATASET = Path(__file__).with_name("dataset.jsonl")
NO_ACTION = {"none", "unsupported"}


def load_dataset(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _param_match(expected: dict, cmd: Command | None) -> bool:
    if not expected:
        return True
    if cmd is None:
        return False
    got = cmd.parameters.model_dump(mode="json")
    for k, v in expected.items():
        g = got.get(k)
        if k == "destination":
            if not g or v.lower() not in g.lower():
                return False
        elif isinstance(v, (int, float)):
            if g is None or abs(float(g) - float(v)) > 1e-6:
                return False
        elif g != v:
            return False
    return True


def score_case(case: dict, result) -> dict:
    exp = case["expected"]
    cmd = result.resolved.command if result.resolved and result.resolved.command else None
    if cmd is None and result.policy.command is not None:
        cmd = result.policy.command
    if cmd is None and result.frame is not None and result.frame.intent.value not in NO_ACTION:
        cmd = Command(intent=result.frame.intent, parameters=result.frame.parameters)  # interpretation only
    pred_intent = cmd.intent.value if cmd else (result.frame.intent.value if result.frame else "error")
    if exp["intent"] in NO_ACTION:
        intent_ok = pred_intent in NO_ACTION or result.policy.reason_code in ("ambiguous", "missing_context", "interpretation_disagreement", "direction_unclear")
    else:
        intent_ok = pred_intent == exp["intent"]
    executed = result.policy.decision == Decision.EXECUTE
    params_ok = intent_ok and _param_match(exp["parameters"], cmd)
    decision_ok = result.policy.decision.value == exp["decision"]
    unintended = executed and (exp["decision"] != "execute" or not intent_ok or not params_ok)
    return {
        "id": case["id"], "category": case["category"], "lang": case["lang"], "text": case["text"],
        "expected_intent": exp["intent"], "pred_intent": pred_intent,
        "expected_decision": exp["decision"], "pred_decision": result.policy.decision.value,
        "reason_code": result.policy.reason_code,
        "pred_params": cmd.parameters.model_dump(mode="json", exclude_none=True) if cmd else {},
        "intent_ok": intent_ok, "params_ok": params_ok, "decision_ok": decision_ok,
        "executed": executed, "unintended_execution": unintended,
        "missed_execution": exp["decision"] == "execute" and not executed,
        "nlu_error": result.nlu_error, "verification": result.verification.model_dump() if result.verification else None,
        "nlu_s": round(result.timings_s.get("nlu_s", 0) + result.timings_s.get("verify_s", 0), 3) if "nlu_s" in result.timings_s else None,
        "meaning": result.frame.meaning if result.frame else None,
    }


def run(parser, cases: list[dict], config: AppConfig | None = None, verbose: bool = False) -> dict:
    config = config or AppConfig()
    rows = []
    for case in cases:
        ctx = ConversationContext(config.context)
        setup = case.get("setup", {})
        if setup.get("pending"):
            ctx.set_pending(Command.model_validate(setup["pending"]))
        if setup.get("last"):
            ctx.set_last_executed(Command.model_validate(setup["last"]))
        pipe = BhashaDrivePipeline(parser, MockVehicleAdapter(config.limits), None, config, ctx)
        result = pipe.process_text(case["text"])
        row = score_case(case, result)
        rows.append(row)
        if verbose:
            flag = "OK " if row["intent_ok"] and row["params_ok"] and row["decision_ok"] else "XX "
            print(f"{flag}{row['id']} [{row['category']}] {row['text']!r}: {row['pred_intent']} {row['pred_params']} "
                  f"-> {row['pred_decision']}/{row['reason_code']} (expected {row['expected_intent']} -> {row['expected_decision']})")
    return {"summary": summarize(rows), "rows": rows}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    lat = [r["nlu_s"] for r in rows if r["nlu_s"] is not None]
    by_cat = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    pct = lambda xs: round(100 * sum(xs) / len(xs), 1) if xs else None  # noqa: E731
    return {
        "cases": n,
        "intent_accuracy_pct": pct([r["intent_ok"] for r in rows]),
        "parameter_accuracy_pct": pct([r["params_ok"] for r in rows]),
        "decision_accuracy_pct": pct([r["decision_ok"] for r in rows]),
        "fully_correct_pct": pct([r["intent_ok"] and r["params_ok"] and r["decision_ok"] for r in rows]),
        "unintended_execution_rate_pct": pct([r["unintended_execution"] for r in rows]),
        "unintended_execution_count": sum(r["unintended_execution"] for r in rows),
        "missed_execution_count": sum(r["missed_execution"] for r in rows),
        "clarify_or_reject_count": sum(r["pred_decision"] == "clarify_or_reject" for r in rows),
        "ask_confirmation_count": sum(r["pred_decision"] == "ask_confirmation" for r in rows),
        "nlu_errors": sum(bool(r["nlu_error"]) for r in rows),
        "nlu_latency_s": {
            "mean": round(statistics.mean(lat), 3) if lat else None,
            "median": round(statistics.median(lat), 3) if lat else None,
            "p90": round(sorted(lat)[int(0.9 * (len(lat) - 1))], 3) if lat else None,
            "max": round(max(lat), 3) if lat else None,
        },
        "by_category_fully_correct_pct": {
            c: pct([r["intent_ok"] and r["params_ok"] and r["decision_ok"] for r in rs]) for c, rs in sorted(by_cat.items())
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true", help="keyword baseline (comparison only)")
    ap.add_argument("--model", help="override BD_LLM_MODEL")
    ap.add_argument("--out", help="write JSON report here")
    ap.add_argument("--dataset", default=str(DATASET), help="JSONL dataset path")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    config = AppConfig()
    if args.baseline:
        from .keyword_baseline import KeywordBaselineParser

        parser, label = KeywordBaselineParser(), "keyword-baseline"
    else:
        from ..nlu.local_llm import OllamaLLM
        from ..nlu.semantic_parser import LLMSemanticParser

        if args.model:
            config.llm.model = args.model
        llm = OllamaLLM(config.llm)
        llm.health_check()
        llm.warmup()
        parser, label = LLMSemanticParser(llm), llm.name
    t0 = time.perf_counter()
    report = run(parser, load_dataset(Path(args.dataset)), config, verbose=args.verbose)
    report["parser"] = label
    report["wall_time_s"] = round(time.perf_counter() - t0, 1)
    print(json.dumps({"parser": label, **report["summary"]}, indent=2, ensure_ascii=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
