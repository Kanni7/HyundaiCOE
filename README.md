# BhashaDrive V2 — edge-oriented Hindi / English / Hinglish in-car voice commands

BhashaDrive maps spoken English, Hindi (Devanagari), Romanized Hindi and Hindi-English
code-switched requests onto a **finite, whitelisted set of vehicle actions**, fully offline.
V2 replaces the mandatory Qwen2.5-3B/Ollama runtime with a compact pipeline:

```
microphone / WAV
  → local ASR            sherpa-onnx + hi-hinglish-swift (Whisper-base fine-tune, INT8 ONNX)   audio/sherpa_asr.py
  → normalisation        Unicode NFC + whitespace only (never drops words)                    pipeline.py
  → intent model         multilingual sentence encoder (INT8 ONNX) + our trained linear heads nlu/intent_model.py
                         (intent, negation, implicit-request)                                  
  → personal commands    SQLite + cosine matching with calibrated thresholds                  personalization/
  → dialog context       one pending proposal (TTL), confirmations, cancellations             pipeline_v2.py
  → policy + validation  negation / ambiguity / confidence / per-intent schema / bounds       policy/policy_v2.py
  → vehicle adapter      MockVehicleAdapter (dev) · OEM adapter boundary (prod)                vehicle/
  → structured result    ResultV2 (Pydantic)                                                   schemas_v2.py
```

No LLM, cloud API or network call is used at runtime. If a required local model cannot load,
the pipeline raises a clear error and never executes a vehicle command. The V1 LLM path is kept
**only** as an opt-in development benchmark (`--legacy-llm`); it is never used as a fallback.

> Evidence status: all accuracy numbers below come from **synthetic** text and **synthetic TTS** audio
> on a development laptop. No real driver recordings and no automotive hardware were available.
> See `AUTOMOTIVE_FEASIBILITY.md` for what this does and does not show.

## Intent taxonomy (v2.0.0, `nlu/taxonomy.py`)

| intent | parameters (validated) |
|---|---|
| `climate_temperature_up` / `climate_temperature_down` | direction fixed by intent; optional `amount` small/medium/large |
| `climate_temperature_set` | **required** `target_temperature_celsius`, range-checked (16–30 °C default) |
| `climate_fan_speed_up` / `climate_fan_speed_down` | optional `amount` |
| `climate_on` / `climate_off` | — |
| `volume_up` / `volume_down` | optional `amount` |
| `media_play` / `media_pause` / `media_next` / `media_previous` | — |
| `navigation_destination_request` | **required** `destination` (length/charset-checked; search request only) |
| `dialog_confirm` / `dialog_cancel` | only meaningful with a pending proposal |
| `unsupported` | always rejected (doors, brakes, windows, calls, chit-chat …) |
| `ambiguous` | always a clarification question |

Decisions: `execute`, `ask_confirmation` (implicit requests such as "I'm freezing"), `clarify`
(ambiguous, low-confidence, missing parameter), `reject` (unsupported, negated, out-of-range), `saved`
(personalisation change committed).

## Setup (Windows or Ubuntu, Python ≥ 3.10; tested on 3.14)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_models.py --asr swift apex --encoder minilm e5   # one-time, ~2.2 GB; swift+minilm only: ~0.6 GB
```

Model files go to `./models` and trained artifacts to `./artifacts` (both git-ignored). After the
download every step runs with `HF_HUB_OFFLINE=1`.

## Reproducing the results

```bash
python scripts/reproduce_v2.py          # all steps below, in order (~30 min on a 6-core laptop CPU)
```

or step by step:

```bash
python -m bhashadrive.training.prepare_dataset                       # data/nlu_v2/*.jsonl (group-split, synthetic)
python -m bhashadrive.training.export_onnx --encoder minilm e5        # ONNX fp32 + dynamic INT8, agreement check
python -m bhashadrive.training.train_classifier --encoder minilm e5 --backend torch onnx_int8   # frozen encoder + heads
python -m bhashadrive.training.finetune_encoder --encoder minilm e5   # EXPERIMENT: supervised encoder fine-tuning
python -m bhashadrive.training.export_onnx --encoder minilm-ft e5-ft
python -m bhashadrive.training.train_classifier --encoder minilm-ft e5-ft --backend onnx_fp32 onnx_int8
python -m bhashadrive.evaluation.eval_nlu_v2 --models minilm_onnx_int8 e5_onnx_int8 minilm-ft_onnx_int8 e5-ft_onnx_fp32 e5-ft_onnx_int8 --promote e5-ft_onnx_int8
python -m bhashadrive.evaluation.eval_personalization                 # calibrates artifacts/personalization.json
python -m bhashadrive.evaluation.eval_asr --models swift apex         # WER/CER, latency, memory, noise, e2e intent
```

Synthetic audio (Windows only, offline OneCore voices):
`powershell -ExecutionPolicy Bypass -File scripts\make_tts_samples.ps1 -OutDir samples\tts_v2 -Manifest scripts\tts_samples_v2.json`.
Real recordings: `python scripts/record_eval_set.py --speaker s01 --noise car_idle`, then
`python -m bhashadrive.evaluation.eval_asr --models swift apex --manifest samples/real/manifest.json`.

## Running

```bash
python -m bhashadrive.cli text "Bhai, AC thoda warm kar do"
python -m bhashadrive.cli file samples/tts_v2/v13_hg_volume.wav
python -m bhashadrive.cli chat        # multi-turn; try the personalisation dialogue below
python -m bhashadrive.cli mic         # push-to-talk
python -m bhashadrive.cli serve       # http://127.0.0.1:8000 (text, audio, /state, /personal)
python -m bhashadrive.cli --json text "Don't turn the AC off"
```

Personalised commands (stored in `data/profiles/bhashadrive.sqlite`, per `BD_PROFILE_ID`):

```
driver> Jab main 'chill mode' bolun, AC 22 pe set kar dena.
[SAY] Should I save this: when you say 'chill mode', I will set the cabin temperature to 22°C?
driver> haan
[SAY] Saved. Say 'chill mode' to set the cabin temperature to 22°C.
driver> Chill mode, yaar.           -> executes (personal command, re-validated)
driver> Don't use chill mode        -> rejected (negation)
driver> what does 'chill mode' do?  -> asks before running (quoted / meta reference)
driver> change 'chill mode' to set the temperature to 20   -> confirm -> updated
driver> make 'relax' do the same as 'chill mode'           -> confirm -> alias added
driver> delete 'chill mode'                                -> confirm -> deleted
driver> list my shortcuts
```

Teaching never fine-tunes a model; it writes one validated database row after explicit
confirmation. One action per shortcut; compound actions are rejected explicitly. Raw audio is not
stored. Correction history is kept only with `BD_STORE_CORRECTIONS=true`.

## Tests

```bash
pytest                    # unit/regression tests (fake model; no weights needed)
pytest -m integration     # real models: offline end-to-end audio, real-model intents, negation, personalisation
```

## Layout

| path | content |
|---|---|
| `bhashadrive/audio/` | `sherpa_asr.py` (V2), `asr.py` (interface + V1 faster-whisper), preprocessing, microphone |
| `bhashadrive/nlu/` | `taxonomy.py`, `encoder.py` (ONNX/torch), `intent_model.py`, `slots.py`; V1 LLM parser kept |
| `bhashadrive/training/` | seeds, hand-written eval set, dataset builder, ONNX export, head training, fine-tuning |
| `bhashadrive/personalization/` | SQLite store, matcher, teach/manage parsing |
| `bhashadrive/policy/` | `policy_v2.py` (V2), V1 policy kept |
| `bhashadrive/vehicle/` | adapter interface, `dispatch_v2`, mock adapter, documented OEM boundary |
| `bhashadrive/evaluation/` | NLU, personalisation and ASR benchmarks; V1 evaluation kept |
| `data/nlu_v2/` | generated dataset splits (text, synthetic) |
| `reports/` | measured results (JSON + Markdown) |

## Android / embedded path

See `AUTOMOTIVE_FEASIBILITY.md` §5. In short: the same `encoder.int8.onnx`/`decoder.int8.onnx`
run with the sherpa-onnx Android (Kotlin/JNI) or C++ API; the sentence encoder runs with ONNX
Runtime Mobile (or is converted for the SoC NPU); the linear heads are a 384×18 matrix and the
policy/validation/personalisation logic is small enough to port to Kotlin/C++ directly. The
vehicle adapter becomes an Android Automotive `CarPropertyManager`/VHAL client or an OEM
middleware client, subject to manufacturer authorisation.
