"""Train the task-specific heads on top of a frozen pretrained sentence encoder.

  python -m bhashadrive.training.train_classifier --encoder minilm --backend onnx_int8

Steps: embed (cached) -> train intent head (logistic regression; linear SVM compared) ->
temperature-scale on validation -> train negation and implicit-request heads -> calibrate
the acceptance thresholds on validation -> write artifacts/nlu/<encoder>_<backend>/.
The pretrained encoder itself is NOT fine-tuned; only the linear heads are trained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..nlu.taxonomy import EXECUTABLE_INTENTS, TAXONOMY_VERSION, IntentV2

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "nlu_v2"
CACHE = ROOT / "artifacts" / "cache"
ENCODERS = {"minilm": "paraphrase-multilingual-MiniLM-L12-v2", "e5": "multilingual-e5-small",
            "minilm-ft": "paraphrase-multilingual-MiniLM-L12-v2 (fine-tuned on BhashaDrive intents)",
            "e5-ft": "multilingual-e5-small (fine-tuned on BhashaDrive intents)"}


def encoder_dir(name: str) -> Path:
    if name.endswith("-ft"):
        return ROOT / "models" / "finetuned" / name
    return ROOT / "models" / "encoders" / ENCODERS[name]
CLASSES = [i.value for i in IntentV2]
EXEC = {i.value for i in EXECUTABLE_INTENTS}
TARGET_RISK = 0.02  # max tolerated wrong-execution rate among accepted executable predictions (validation)


def load_split(name: str) -> list[dict]:
    p = DATA / f"{name}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_encoder(encoder: str, backend: str, threads: int = 4):
    from ..nlu.encoder import OnnxSentenceEncoder, TorchSentenceEncoder

    src = encoder_dir(encoder)
    if backend == "torch":
        return TorchSentenceEncoder(src)
    return OnnxSentenceEncoder(ROOT / "models" / "onnx" / encoder / f"model.{backend.split('_')[1]}.onnx", src, threads=threads)


def embed_cached(enc, key: str, texts: list[str]) -> np.ndarray:
    h = hashlib.sha1(("\n".join(texts)).encode("utf-8")).hexdigest()[:12]
    p = CACHE / f"{key}_{h}.npy"
    if p.is_file():
        return np.load(p)
    CACHE.mkdir(parents=True, exist_ok=True)
    x = enc.encode(texts)
    np.save(p, x)
    return x


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_nll = 1.0, np.inf
    for t in np.exp(np.linspace(np.log(0.05), np.log(10), 200)):
        p = softmax(logits / t)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def calibrate_thresholds(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    """Pick (score, margin) thresholds on validation: maximise coverage of correct executable
    predictions subject to wrong-execution rate <= TARGET_RISK among accepted executable ones."""
    top = probs.argmax(1)
    srt = np.sort(probs, 1)
    score, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    is_exec = np.array([classes[t] in EXEC for t in top])
    correct = top == y
    best = None
    for ts in np.linspace(0.2, 0.95, 31):
        for tm in np.linspace(0.0, 0.6, 25):
            acc = is_exec & (score >= ts) & (margin >= tm)
            n = acc.sum()
            if n == 0:
                continue
            risk = (acc & ~correct).sum() / n
            cov = (acc & correct).sum() / max(1, (is_exec & correct).sum())
            if risk <= TARGET_RISK:
                cand = (2, cov, -ts, -tm)
            else:
                cand = (1, -risk, cov, -ts, -tm)
            if best is None or cand > best[0]:
                best = (cand, float(ts), float(tm), float(risk), float(cov), int(n))
    _, ts, tm, risk, cov, n = best
    ts = min(max(0.60, ts), 0.75)
    tm = max(0.15, tm)
    return {"min_score": round(ts, 4), "min_margin": round(tm, 4), "val_wrong_execution_rate": round(risk, 4),
            "val_coverage_of_correct_exec": round(cov, 4), "val_accepted": n, "target_risk": TARGET_RISK,
            "met_target": bool(best[0][0])}


def train(encoder: str, backend: str, out_dir: Path | None = None, C_grid=(0.5, 1, 2, 4, 8, 16, 32)) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.svm import LinearSVC

    enc = make_encoder(encoder, backend)
    key = f"{encoder}_{backend}"
    tr, va = load_split("train"), load_split("val")
    Xtr, Xva = embed_cached(enc, key, [r["text"] for r in tr]), embed_cached(enc, key, [r["text"] for r in va])
    ci = {c: i for i, c in enumerate(CLASSES)}
    ytr, yva = np.array([ci[r["intent"]] for r in tr]), np.array([ci[r["intent"]] for r in va])

    # Intent head: choose C on validation macro-F1; compare LR vs linear SVM.
    sel = []
    for C in C_grid:
        lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        sel.append(("logreg", C, f1_score(yva, lr.predict(Xva), average="macro"), lr))
        sv = LinearSVC(C=C / 8, class_weight="balanced", max_iter=20000).fit(Xtr, ytr)
        sel.append(("linear_svm", C / 8, f1_score(yva, sv.predict(Xva), average="macro"), sv))
    comparison = [{"model": m, "C": c, "val_macro_f1": round(f, 4)} for m, c, f, _ in sel]
    best_lr = max((s for s in sel if s[0] == "logreg"), key=lambda s: s[2])
    best_svm = max((s for s in sel if s[0] == "linear_svm"), key=lambda s: s[2])
    # Logistic regression is kept as the runtime head because it yields scores that can be
    # temperature-scaled; the SVM result is reported for comparison only.
    clf = best_lr[3]
    W, b = clf.coef_.astype(np.float32), clf.intercept_.astype(np.float32)
    logits_va = Xva @ W.T + b
    T = fit_temperature(logits_va, yva)
    p_raw, p_cal = softmax(logits_va), softmax(logits_va / T)
    thresholds = calibrate_thresholds(p_cal, yva, CLASSES)

    # Negation head (binary) and implicit-request head (executable intents only).
    neg_tr = np.array([r["negated"] for r in tr], int)
    neg = LogisticRegression(C=4, max_iter=3000).fit(Xtr, neg_tr)
    ex_tr = [i for i, r in enumerate(tr) if r["intent"] in EXEC and not r["negated"]]
    imp_y = np.array([tr[i]["speech_act"] == "implicit_request" for i in ex_tr], int)
    imp = LogisticRegression(C=4, max_iter=3000, class_weight="balanced").fit(Xtr[ex_tr], imp_y)
    ex_va = [i for i, r in enumerate(va) if r["intent"] in EXEC and not r["negated"]]
    imp_va = np.array([va[i]["speech_act"] == "implicit_request" for i in ex_va], int)

    if out_dir:
        out_dir = (ROOT / out_dir).resolve() if not Path(out_dir).is_absolute() else Path(out_dir)
    else:
        out_dir = ROOT / "artifacts" / "nlu" / key
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "heads.npz", intent_W=W, intent_b=b, neg_W=neg.coef_.astype(np.float32),
             neg_b=neg.intercept_.astype(np.float32), imp_W=imp.coef_.astype(np.float32), imp_b=imp.intercept_.astype(np.float32))
    onnx = None if backend == "torch" else f"models/onnx/{encoder}/model.{backend.split('_')[1]}.onnx"
    manifest = {
        "model_version": f"bhashadrive-nlu-{key}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
        "taxonomy_version": TAXONOMY_VERSION, "classes": CLASSES,
        "encoder": {"name": ENCODERS[encoder], "backend": backend, "onnx": onnx,
                    "tokenizer_dir": encoder_dir(encoder).relative_to(ROOT).as_posix(),
                    "fine_tuned": encoder.endswith("-ft")},
        "intent_head": {"type": "logistic_regression", "C": best_lr[1], "temperature": round(T, 4)},
        "confidence_meaning": ("temperature-scaled softmax probability of the top intent from a logistic-regression "
                               "head; temperature fitted on the validation split. It is an estimate, not a guarantee, "
                               "and is only calibrated for text resembling the training distribution."),
        "thresholds": thresholds,
        "negation_threshold": 0.5, "implicit_threshold": 0.5,
        "train_size": len(tr), "val_size": len(va),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = {
        "key": key, "classifier_comparison": comparison,
        "best": {"logreg": {"C": best_lr[1], "val_macro_f1": round(best_lr[2], 4)},
                 "linear_svm": {"C": best_svm[1], "val_macro_f1": round(best_svm[2], 4)}},
        "temperature": round(T, 4), "val_ece_raw": round(ece(p_raw, yva), 4), "val_ece_calibrated": round(ece(p_cal, yva), 4),
        "thresholds": thresholds,
        "val_negation_acc": round(float((neg.predict(Xva) == np.array([r["negated"] for r in va], int)).mean()), 4),
        "val_implicit_acc": round(float((imp.predict(Xva[ex_va]) == imp_va).mean()), 4),
        "artifacts": str(out_dir.relative_to(ROOT)),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs="+", default=["minilm"], choices=list(ENCODERS))
    ap.add_argument("--backend", nargs="+", default=["onnx_int8"], choices=["torch", "onnx_fp32", "onnx_int8"])
    ap.add_argument("--out-dir", type=Path, default=None, help="Directory to save artifacts into")
    args = ap.parse_args(argv)
    reports = []
    for e in args.encoder:
        for b in args.backend:
            r = train(e, b, out_dir=args.out_dir)
            reports.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "classifier_comparison"}, indent=1))
    p = ROOT / "reports" / "nlu_training.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    old.update({r["key"]: r for r in reports})
    p.write_text(json.dumps(old, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
