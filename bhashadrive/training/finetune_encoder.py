"""EXPERIMENT: supervised fine-tuning of the sentence encoder on our labelled intents.

  python -m bhashadrive.training.finetune_encoder --encoder minilm

Frozen pretrained embeddings + a linear head confuse opposite-polarity intents (warmer vs
cooler, play vs pause) because paraphrase embeddings mostly encode topic. This script
fine-tunes the transformer layers (word embeddings frozen) with a mean-pooling + linear
classification objective, selects the epoch by validation macro-F1, and saves a plain
Hugging Face model to models/finetuned/<encoder>-ft/ that the normal export/training
scripts can consume. The test and hand-written sets are never used here.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .train_classifier import CLASSES, ENCODERS, ROOT, load_split


def finetune(encoder: str, epochs: int = 8, lr: float = 4e-5, batch: int = 16, seed: int = 7, threads: int = 8) -> dict:
    import torch
    from sklearn.metrics import f1_score
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    src = ROOT / "models" / "encoders" / ENCODERS[encoder]
    dst = ROOT / "models" / "finetuned" / f"{encoder}-ft"
    prefix = "query: " if encoder == "e5" else ""
    tok = AutoTokenizer.from_pretrained(str(src), local_files_only=True)
    base = AutoModel.from_pretrained(str(src), local_files_only=True)
    for p in base.embeddings.parameters():
        p.requires_grad = False
    head = torch.nn.Linear(base.config.hidden_size, len(CLASSES))
    drop = torch.nn.Dropout(0.1)
    ci = {c: i for i, c in enumerate(CLASSES)}
    tr, va = load_split("train"), load_split("val")
    ytr = torch.tensor([ci[r["intent"]] for r in tr])
    yva = np.array([ci[r["intent"]] for r in va])
    counts = np.bincount(ytr.numpy(), minlength=len(CLASSES)).astype(np.float32)
    weights = torch.tensor(counts.sum() / np.clip(counts, 1, None) / len(CLASSES))

    def embed(texts, train_mode):
        b = tok([prefix + t for t in texts], padding=True, truncation=True, max_length=64, return_tensors="pt")
        h = base(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1) * 20.0  # scale: the head sees normalised embeddings
        return drop(pooled) if train_mode else pooled

    params = [p for p in base.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    steps = epochs * ((len(tr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps)) * max(0.0, 1 - s / steps))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    texts = [r["text"] for r in tr]
    best, history, t0 = (-1.0, -1), [], time.perf_counter()
    for ep in range(epochs):
        base.train(); head.train()
        perm = np.random.permutation(len(tr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            logits = head(embed([texts[j] for j in idx], True))
            loss = loss_fn(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        base.eval(); head.eval()
        with torch.no_grad():
            pv = torch.cat([head(embed([r["text"] for r in va[i:i + 64]], False)) for i in range(0, len(va), 64)]).argmax(1).numpy()
        f1 = float(f1_score(yva, pv, average="macro"))
        history.append({"epoch": ep + 1, "val_macro_f1": round(f1, 4), "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(history[-1], flush=True)
        if f1 > best[0]:
            best = (f1, ep + 1)
            dst.mkdir(parents=True, exist_ok=True)
            base.save_pretrained(str(dst))
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"):
        if (src / f).is_file():
            shutil.copy(src / f, dst / f)
    info = {"encoder": ENCODERS[encoder], "output": str(dst.relative_to(ROOT)), "epochs": epochs, "lr": lr, "batch": batch,
            "seed": seed, "frozen": "word embeddings", "best_epoch": best[1], "best_val_macro_f1": round(best[0], 4),
            "history": history, "train_size": len(tr), "train_time_s": round(time.perf_counter() - t0, 1),
            "note": "Fine-tuned on SYNTHETIC labelled text only; validation used for epoch selection."}
    (dst / "bhashadrive_finetune.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs="+", default=["minilm"], choices=["minilm", "e5"])
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args(argv)
    out = {}
    for e in args.encoder:
        out[f"{e}-ft"] = finetune(e, epochs=args.epochs)
    p = ROOT / "reports" / "finetune.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    old.update(out)
    p.write_text(json.dumps(old, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
