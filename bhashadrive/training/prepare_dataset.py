"""Build the labelled V2 NLU dataset from the seed templates.

  python -m bhashadrive.training.prepare_dataset            # writes data/nlu_v2/{train,val,test}.jsonl

Splits are made by seed template (group), so paraphrase expansions of one template never
appear in two splits. The hand-written `heldout_unseen.jsonl` (see handwritten.py) and the
V1 sets mapped to V2 labels are additional, independently written test sets.
All examples are SYNTHETIC text; provenance is recorded per row.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import re
from pathlib import Path

from ..nlu.taxonomy import DIALOG_INTENTS, EXECUTABLE_INTENTS, TAXONOMY_VERSION, IntentV2
from . import seeds as S
from .handwritten import HANDWRITTEN, PERSONAL_EVAL

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "nlu_v2"
LANG = {"en": "english", "hi": "hindi", "rh": "romanized_hindi", "hg": "hinglish"}
_ALT = re.compile(r"\{([^{}]*\|[^{}]*)\}")


def expand_alternations(t: str) -> list[str]:
    parts = _ALT.split(t)
    # parts alternates literal, options, literal, ...
    choices = [[p] if i % 2 == 0 else p.split("|") for i, p in enumerate(parts)]
    return [" ".join("".join(c).split()) for c in itertools.product(*choices)]


def decision_for(intent: str, speech_act: str, negated: bool, params_ok: bool = True) -> str:
    it = IntentV2(intent)
    if negated or it == IntentV2.UNSUPPORTED:
        return "reject"
    if it == IntentV2.AMBIGUOUS or not params_ok:
        return "clarify"
    if it in DIALOG_INTENTS:
        return "context_dependent"
    return "ask_confirmation" if speech_act == "implicit_request" else "execute"


def make_row(rid, text, intent, lang, speech_act, negated, params, group, split, provenance):
    return {"id": rid, "text": text, "intent": intent, "parameters": params, "language_pattern": lang,
            "speech_act": speech_act, "negated": negated, "decision": decision_for(intent, speech_act, negated),
            "group": group, "split": split, "provenance": provenance, "taxonomy_version": TAXONOMY_VERSION}


def expand_seed(intent: str, seed: str, rng: random.Random, max_variants: int) -> list[dict]:
    negated, implicit = seed.startswith("!"), False
    seed = seed.lstrip("!")
    code, body = seed.split(":", 1)
    if body.startswith("~"):
        implicit, body = True, body[1:]
    dev = code == "hi"
    texts = expand_alternations(body)
    rows = []
    for _ in range(max_variants * 3):
        if len(rows) >= max_variants:
            break
        t = rng.choice(texts)
        params: dict = {}
        if "{T}" in t:
            pool = S.TEMPS_HI if dev else (S.TEMPS_RH if code in ("rh", "hg") else S.TEMPS_EN)
            surf, val = rng.choice(pool)
            t = t.replace("{T}", surf)
            params["target_temperature_celsius"] = val
        if "{D}" in t:
            d = rng.choice(S.DESTS_DEV if dev else S.DESTS_LATIN)
            t = t.replace("{D}", d)
            params["destination"] = d
        if intent not in (IntentV2.DIALOG_CONFIRM.value, IntentV2.DIALOG_CANCEL.value):
            pre, suf = (S.PREFIX_DEV, S.SUFFIX_DEV) if dev else (S.PREFIX_LATIN, S.SUFFIX_LATIN)
            if not negated:
                t = rng.choice(pre) + (t[0].lower() + t[1:] if t and not dev and rng.random() < 0.3 else t) + rng.choice(suf)
        if rng.random() < 0.5 and not dev:
            t = t.lower()
        t = " ".join(t.split())
        if any(r["text"] == t for r in rows):
            continue
        act = "implicit_request" if implicit else ("dialog" if IntentV2(intent) in DIALOG_INTENTS else
                                                     "direct_command" if IntentV2(intent) in EXECUTABLE_INTENTS else "other")
        rows.append({"text": t, "intent": intent, "lang": LANG[code], "speech_act": act, "negated": negated,
                     "params": params})
    return rows


def build(seed: int = 13, max_variants: int = 6, val_frac: float = 0.15, test_frac: float = 0.15) -> dict[str, list]:
    rng = random.Random(seed)
    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for intent, seed_list in S.SEEDS.items():
        IntentV2(intent)  # validate taxonomy
        groups = list(enumerate(seed_list))
        rng.shuffle(groups)
        n = len(groups)
        n_val, n_test = max(1, round(n * val_frac)), max(1, round(n * test_frac))
        assign = {g: ("test" if k < n_test else "val" if k < n_test + n_val else "train") for k, (g, _) in enumerate(groups)}
        for g, s in groups:
            split = assign[g]
            for j, r in enumerate(expand_seed(intent, s, rng, max_variants)):
                splits[split].append(make_row(f"{intent}:{g}:{j}", r["text"], intent, r["lang"], r["speech_act"],
                                              r["negated"], r["params"], f"{intent}:{g}", split, "synthetic_template"))
    for k in splits:
        rng.shuffle(splits[k])
    return splits


def handwritten_rows() -> list[dict]:
    rows = []
    for i, (text, intent, lang, act, neg, params) in enumerate(HANDWRITTEN):
        rows.append(make_row(f"hw{i:03d}", text, intent, lang, act, neg, params, f"hw{i:03d}", "heldout_unseen",
                             "synthetic_handwritten_unseen"))
    return rows


# --- V1 sets mapped onto the V2 taxonomy (independently written before V2 existed) ---------------
def map_v1(row: dict) -> dict | None:
    if row.get("setup"):
        return None  # needs conversational context; covered by the dialog tests instead
    e = row["expected"]
    it, p = e.get("intent"), e.get("parameters", {}) or {}
    neg = row.get("category") == "negation" and e.get("decision") == "clarify_or_reject"
    if row.get("category") == "ambiguous":
        it = "ambiguous"
    m = {
        "ambiguous": "ambiguous",
        "climate_adjust_temperature": "climate_temperature_up" if p.get("direction") == "increase" else "climate_temperature_down",
        "climate_set_temperature": "climate_temperature_set",
        "climate_ac_power": "climate_on" if p.get("power") == "on" else "climate_off",
        "fan_adjust_speed": "climate_fan_speed_up" if p.get("direction") == "increase" else "climate_fan_speed_down",
        "volume_adjust": "volume_up" if p.get("direction") == "increase" else "volume_down",
        "media_play": "media_play", "media_pause": "media_pause", "media_next_track": "media_next",
        "media_previous_track": "media_previous", "navigation_search": "navigation_destination_request",
        "unsupported": "unsupported", "none": "unsupported",
    }.get(it)
    if m is None or (it in ("climate_adjust_temperature", "fan_adjust_speed", "volume_adjust") and not p.get("direction")):
        return None
    if it == "climate_ac_power" and not p.get("power"):
        return None
    params = {}
    if p.get("target_temperature_celsius") is not None:
        params["target_temperature_celsius"] = p["target_temperature_celsius"]
    if p.get("destination"):
        params["destination"] = p["destination"]
    lang = {"en": "english", "hi": "hindi", "hinglish": "hinglish", "rom_hi": "romanized_hindi"}.get(row.get("lang"), "unknown")
    act = "implicit_request" if e.get("decision") == "ask_confirmation" else ("other" if m == "unsupported" else "direct_command")
    return make_row(f"v1:{row['id']}", row["text"], m, lang, act, bool(neg), params, f"v1:{row['id']}", "v1_mapped",
                    "synthetic_v1_handwritten")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv=None) -> int:
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--variants", type=int, default=6)
    args = ap.parse_args(argv)
    splits = build(args.seed, args.variants)
    for k, rows in splits.items():
        write_jsonl(OUT / f"{k}.jsonl", rows)
    hw = handwritten_rows()
    write_jsonl(OUT / "heldout_unseen.jsonl", hw)
    v1 = []
    for name in ("dataset.jsonl", "heldout.jsonl"):
        p = ROOT / "bhashadrive" / "evaluation" / name
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = map_v1(json.loads(line))
                if r:
                    v1.append(r)
    seen = {r["text"].lower() for r in splits["train"]}
    dropped = [r["text"] for r in v1 if r["text"].lower() in seen]
    v1 = [r for r in v1 if r["text"].lower() not in seen]  # exact duplicates of training text are not "unseen"
    if dropped:
        print(f"v1_mapped: dropped {len(dropped)} row(s) identical to training text: {dropped}")
    write_jsonl(OUT / "v1_mapped.jsonl", v1)
    (OUT / "personal_eval.json").write_text(json.dumps(PERSONAL_EVAL, ensure_ascii=False, indent=1), encoding="utf-8")
    # Leakage check: no exact text overlap between train and any evaluation split.
    train_txt = {r["text"].lower() for r in splits["train"]}
    for name, rows in (("val", splits["val"]), ("test", splits["test"]), ("heldout_unseen", hw), ("v1_mapped", v1)):
        overlap = sum(r["text"].lower() in train_txt for r in rows)
        print(f"{name:15s} {len(rows):5d} rows   exact-text overlap with train: {overlap}")
    print(f"train           {len(splits['train']):5d} rows   groups: {len({r['group'] for r in splits['train']})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
