"""Export a sentence encoder's transformer to ONNX and apply dynamic INT8 quantization.

  python -m bhashadrive.training.export_onnx --encoder minilm
  python -m bhashadrive.training.export_onnx --encoder e5

Writes models/onnx/<name>/model.fp32.onnx and model.int8.onnx, then checks numerical
agreement (cosine similarity of pooled embeddings) against the PyTorch reference on the
dataset sentences. Results go to reports/onnx_export_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from .train_classifier import ENCODERS, encoder_dir  # noqa: E402


def export(name: str) -> dict:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel

    from ..nlu.encoder import OnnxSentenceEncoder, TorchSentenceEncoder

    src = encoder_dir(name)
    out = ROOT / "models" / "onnx" / name
    out.mkdir(parents=True, exist_ok=True)
    fp32, int8 = out / "model.fp32.onnx", out / "model.int8.onnx"

    base = AutoModel.from_pretrained(str(src), local_files_only=True).eval()

    class Wrapper(torch.nn.Module):  # keyword call: positional args differ across transformers versions
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    model = Wrapper(base).eval()
    ids = torch.ones(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.long)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(model, (ids, mask), str(fp32), input_names=["input_ids", "attention_mask"],
                          output_names=["last_hidden_state"], opset_version=17, dynamo=False,
                          dynamic_axes={"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"},
                                        "last_hidden_state": {0: "b", 1: "s"}})
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    export_s = time.perf_counter() - t0

    texts = []
    for split in ("train", "val", "test", "heldout_unseen"):
        p = ROOT / "data" / "nlu_v2" / f"{split}.jsonl"
        texts += [json.loads(l)["text"] for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = TorchSentenceEncoder(src).encode(texts)
    res = {"encoder": ENCODERS[name], "n_sentences": len(texts), "export_and_quantize_s": round(export_s, 1),
           "size_mb": {"safetensors": round((src / "model.safetensors").stat().st_size / 2**20, 1),
                       "onnx_fp32": round(fp32.stat().st_size / 2**20, 1), "onnx_int8": round(int8.stat().st_size / 2**20, 1)}}
    for label, path in (("onnx_fp32", fp32), ("onnx_int8", int8)):
        emb = OnnxSentenceEncoder(path, src, threads=4).encode(texts)
        cos = (emb * ref).sum(1)
        res[f"cosine_vs_torch_{label}"] = {"mean": round(float(cos.mean()), 5), "min": round(float(cos.min()), 5),
                                           "p01": round(float(np.percentile(cos, 1)), 5)}
    rp = ROOT / "reports" / f"onnx_export_{name}.json"
    rp.parent.mkdir(exist_ok=True)
    rp.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", nargs="+", default=["minilm"], choices=list(ENCODERS))
    args = ap.parse_args(argv)
    for name in args.encoder:
        print(json.dumps(export(name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
