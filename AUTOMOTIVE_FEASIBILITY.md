# BhashaDrive: Automotive Feasibility Notes

This document describes what the prototype does today, what it needs, and what is still unverified for an
in-vehicle deployment. All figures come from one developer laptop. They do not show that the system is
suitable for an infotainment head unit.

## 1. What runs locally today

| Component | Implementation | Where it runs |
|---|---|---|
| Speech recognition | `faster-whisper` (CTranslate2), multilingual Whisper `small` (also `base`/`tiny`), int8 on CPU | Python process |
| Semantic parser | `qwen2.5:3b` (Q4_K_M GGUF, ~1.9 GB file) served by Ollama, JSON-schema constrained decoding | Ollama server on 127.0.0.1 |
| Verifier (second opinion) | Same local model, separate short prompt | Ollama server on 127.0.0.1 |
| Context, policy, validation | Deterministic Python | Python process |
| Vehicle | `MockVehicleAdapter` simulator | Python process |

There are no remote calls in the inference path. The Python process accepts only loopback LLM endpoints
(`BD_LLM_ALLOW_REMOTE=false`). Whisper loads with `local_files_only=True`. The integration test
`test_end_to_end_offline` blocks all non-loopback sockets in the Python process and runs audio through the
full pipeline. That test passes on the development machine. Section 6 lists what the test does not cover.

## 2. Measured resources (development laptop)

Hardware: AMD Ryzen 5 4600H (12 logical cores), 15.4 GiB RAM, NVIDIA GTX 1650 4 GB, Windows 11.
`reports/benchmark.json` holds the full measured values. A summary is in the README. Main observations:

- `qwen2.5:3b` takes about 2.2 GB in Ollama (`ollama ps`, 100% GPU). It needs two LLM calls per standalone
  utterance: a parser call (about 2 s) and a verifier call (about 0.4 s).
- Whisper `small` int8 on CPU takes about 4 s for a 2–3 s utterance on this CPU. On a head-unit SoC it will
  be slower unless an NPU/GPU path is used.
- The laptop has a discrete GPU with 4 GB of memory. Most head units do not.

## 3. Path to an embedded target

The interfaces were built so that models can be swapped:

- `SpeechRecognizer` (ASR) and `SemanticParser` / `LanguageModel` (NLU) are abstract interfaces. The policy,
  context and vehicle layers do not depend on the model.
- Options for the target, in rough order of effort:
  1. **Smaller off-the-shelf models.** Whisper `base`/`tiny`, `qwen2.5:1.5b` or `0.5b`. These need only a
     configuration change (`BD_ASR_MODEL`, `BD_LLM_MODEL`). Accuracy loss must be measured with
     `scripts/benchmark.py`.
  2. **Distilled intent model.** The schema is fixed and small (12 intents, 5 parameters, 7 context
     relations), so a compact encoder can replace the generative LLM. Examples: a multilingual
     MiniLM/IndicBERT-class model with classification and span heads, about 20–120 M parameters. Train it
     on LLM-labelled and human-verified Hinglish data, and keep the same `SemanticFrame` output. The
     deterministic policy layer stays the same.
  3. **Streaming / on-device ASR.** Use an Indic-tuned ASR model (for example an IndicConformer-class model
     or a Hindi-English fine-tuned Whisper). Export it to the SoC's NPU runtime (QNN/SNPE, TensorRT,
     OpenVINO, NNAPI) with int8 weights.
- Expected trade-offs:
  - Model size goes down, so memory and latency go down.
  - Hindi and code-switched accuracy, and reasoning about indirect requests, also go down.
  - Smaller generative models rely more on the verifier and the policy layer to stay safe. This raises the
    clarification rate. It must not raise the unintended-execution rate.
  - GPU/NPU offload lowers latency. It raises power and thermal load, and it shares resources with
    navigation rendering.

## 4. Interfaces Hyundai would need to expose

`bhashadrive/vehicle/hyundai_adapter.py` is a documented boundary. It does not implement anything. A real
integration needs manufacturer-authorised services for these functions:

| Function | Required interface (OEM-owned) |
|---|---|
| Temperature step / set-point | Climate HMI service: per-zone set-point read/write, with limits enforced by the HVAC controller |
| AC on/off | Climate service request that the HVAC controller arbitrates |
| Fan level | Climate service: blower level step and read-back |
| Media volume | Audio management service: media-stream volume, with audio-focus/arbitration rules (chimes, calls and ADAS warnings take priority) |
| Play/pause/next/previous | Media session service for the active source |
| Navigation search | Navigation app search intent. Route guidance stays inside the navigation application |
| State read-back | Current values for all of the above, so the assistant reports real outcomes |

Examples of such platforms are Android Automotive VHAL/CarPropertyManager, SOME/IP services, or an OEM
middleware. The voice application must get permission only for these functions. It must never write raw
CAN frames, bypass the gateway, or reach steering, braking, propulsion or other safety-relevant functions.
The prototype has no code path to such functions, and the policy rejects them.

## 5. Unverified production claims

None of these items has been verified:

- Latency, memory and power on any automotive SoC.
- Accuracy on real in-cabin audio: road/engine/HVAC noise, far-field microphones, multiple speakers,
  real code-switching. The ASR tests use synthetic Windows TTS voices only.
- Accuracy on a large, representative, independently labelled Hindi/Hinglish corpus. The local datasets
  have 91 development cases and 37 held-out cases, written by one author.
- Calibration. Model confidence is not used. The safety behaviour comes from policy rules plus agreement
  between two prompts, and has been measured only on the small datasets above.
- Robustness to adversarial prompts beyond the few injection cases in the dataset.
- Functional safety (ISO 26262), cybersecurity (ISO/SAE 21434, UN R155/R156) and driver-distraction
  compliance.
- Long-run stability, thermal behaviour and cold-start time.

## 6. Offline operation and network failure

- After the one-time model download, no component needs the network. The in-process test blocks external
  sockets. The Ollama server is a separate process. To confirm that the whole machine works offline,
  disconnect the network and run `pytest -m integration` (README, "Offline operation").
- In a vehicle, network loss must not change behaviour. The design has no cloud dependency. If the local
  model fails (timeout, crash, invalid output), the system answers "could not understand" and does
  nothing. There is no silent keyword fallback.
- Production must add a watchdog for the model runtime, bounded startup time, and a degraded mode in
  which the voice assistant becomes unavailable and the HMI says so.

## 7. Privacy, cybersecurity and action validation

- **Audio:** processed in memory. Audio is not written to disk unless `BD_STORE_AUDIO=true`. Uploaded API
  audio goes to a temporary directory that is deleted after processing.
- **Retained data:**
  - The pending proposal and the last executed command, with time-outs of 60 s and 120 s.
  - Up to 6 recent transcript/response pairs, truncated to 200 characters, held in memory only.
  - Nothing is persisted. The CLI and API print results to the console, and no log file is written.
- **Network exposure:** the API binds to `127.0.0.1` by default. It has no authentication, so do not bind
  it to other interfaces.
- **Untrusted model output:**
  - All LLM output must pass strict JSON schema validation (Pydantic, `extra="forbid"` on parameters).
  - It then goes through the deterministic context resolver and the policy.
  - Commands map to adapter methods through an explicit `match` statement. There is no dynamic dispatch,
    shell or eval.
  - Numeric values are range-checked. Destination strings are length-limited and filtered for control and
    markup characters.
  - Navigation is only a search *request*.
- **Execution policy:**
  - The LLM cannot execute anything directly.
  - Execution needs an explicit request, or explicit confirmation of a pending proposal.
  - Indirect statements lead to a confirmation question.
  - Negation, ambiguity, missing or invalid parameters, unsupported functions, model errors and
    disagreement between the parser and the verifier all lead to no action.
- **Remaining work for production:**
  - Speaker or seat attribution (driver versus passenger).
  - Rate limiting of actuation.
  - Arbitration with the physical controls.
  - Security review of the IPC boundary.
  - Signed model artefacts with integrity checks at load time.
