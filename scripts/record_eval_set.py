"""Record REAL speech for ASR / end-to-end evaluation (push-to-talk, local only).

  python scripts/record_eval_set.py --speaker s01 [--out samples/real]

Prompts each phrase, records it, and writes <out>/<speaker>_<id>.wav plus manifest.json
(provenance "real_recording"). Then run:
  python -m bhashadrive.evaluation.eval_asr --models swift apex --manifest samples/real/manifest.json
Recordings are personal data: keep them out of Git (samples/real/ is git-ignored) and get
each speaker's consent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Devanagari prompts carry a Romanized reference because the Hinglish ASR models emit Roman script.
EXTRA = [
    {"id": "d01_hi_cold", "prompt": "यहाँ बहुत ठंड लग रही है।", "text": "yahan bahut thand lag rahi hai",
     "lang": "hindi", "expected_intent": "climate_temperature_up"},
    {"id": "d02_hi_ac_neg", "prompt": "एसी को बंद मत करना।", "text": "AC ko band mat karna", "lang": "hindi",
     "expected_intent": "climate_off", "negated": True},
    {"id": "d03_hi_song", "prompt": "अगला गाना चलाओ।", "text": "agla gaana chalao", "lang": "hindi", "expected_intent": "media_next"},
    {"id": "d04_hi_temp", "prompt": "तापमान इक्कीस डिग्री रखो।", "text": "taapmaan ikkis degree rakho", "lang": "hindi",
     "expected_intent": "climate_temperature_set"},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True)
    ap.add_argument("--out", default=str(ROOT / "samples" / "real"))
    ap.add_argument("--noise", default="quiet_room", help="describe the condition, e.g. car_idle, car_60kmh_ac_on")
    args = ap.parse_args()
    import soundfile as sf

    from bhashadrive.audio.microphone import record_push_to_talk

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((ROOT / "scripts" / "tts_samples_v2.json").read_text(encoding="utf-8")) + EXTRA
    mf = out / "manifest.json"
    manifest = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else []
    for p in prompts:
        print(f"\nSay: {p.get('prompt', p['text'])}")
        audio = record_push_to_talk()
        rid = f"{args.speaker}_{p['id']}"
        sf.write(out / f"{rid}.wav", audio, 16000)
        manifest = [m for m in manifest if m["id"] != rid] + [{
            "id": rid, "text": p["text"], "lang": p.get("lang", "english"), "expected_intent": p.get("expected_intent"),
            "negated": p.get("negated", False), "speaker": args.speaker, "condition": args.noise,
            "provenance": "real_recording"}]
        mf.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(prompts)} recordings to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
