import sys, time
from bhashadrive.pipeline import build_default_pipeline
p = build_default_pipeline(with_asr=False)
for t in sys.argv[1:]:
    if t == "RESET":
        p.context.reset(); print("--reset--"); continue
    t0=time.time(); r = p.process_text(t); f = r.frame
    print(f"{time.time()-t0:5.2f}s | {t} | {f.intent.value if f else r.nlu_error} {f.parameters.model_dump(exclude_none=True, mode='json') if f else ''} act={f.speech_act.value if f else ''} rel={f.context_relation.value if f else ''} neg={f.negated if f else ''} | {f.meaning if f else ''} -> {r.policy.decision.value}/{r.policy.reason_code} | {r.response_text}")
