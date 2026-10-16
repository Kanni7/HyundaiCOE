"""Runtime intent model: local ONNX encoder + trained linear heads (numpy only).

Loads once at startup from `artifacts/nlu/<key>/` (manifest.json + heads.npz). If any
artifact is missing it raises `EncoderUnavailableError`; the pipeline then refuses to act.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..schemas_v2 import IntentPrediction
from .encoder import EncoderUnavailableError, OnnxSentenceEncoder, SentenceEncoder, load_encoder_manifest
from .slots import negation_cue
from .taxonomy import TAXONOMY_VERSION, IntentV2


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class IntentModel:
    def __init__(self, manifest: dict, heads: dict, encoder: SentenceEncoder):
        if manifest.get("taxonomy_version") != TAXONOMY_VERSION:
            raise EncoderUnavailableError(
                f"Model taxonomy {manifest.get('taxonomy_version')} != code taxonomy {TAXONOMY_VERSION}; retrain.")
        self.manifest, self.encoder = manifest, encoder
        self.classes = [IntentV2(c) for c in manifest["classes"]]
        self.W, self.b = heads["intent_W"], heads["intent_b"]
        self.neg_W, self.neg_b = heads["neg_W"], heads["neg_b"]
        self.imp_W, self.imp_b = heads["imp_W"], heads["imp_b"]
        self.T = float(manifest["intent_head"]["temperature"])
        self.thresholds = manifest["thresholds"]
        self.version = manifest["model_version"]

    @classmethod
    def load(cls, artifacts_dir: str | Path, root: str | Path, threads: int = 2) -> "IntentModel":
        artifacts_dir, root = Path(artifacts_dir), Path(root)
        manifest = load_encoder_manifest(artifacts_dir)
        heads_path = artifacts_dir / "heads.npz"
        if not heads_path.is_file():
            raise EncoderUnavailableError(f"Classifier heads not found: {heads_path}")
        enc = manifest["encoder"]
        if not enc.get("onnx"):
            raise EncoderUnavailableError("Runtime requires an ONNX encoder artifact (backend onnx_fp32/onnx_int8).")
        encoder = OnnxSentenceEncoder(root / enc["onnx"], root / enc["tokenizer_dir"], threads=threads)
        with np.load(heads_path) as h:
            heads = {k: h[k] for k in h.files}
        return cls(manifest, heads, encoder)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts)

    def predict_embedding(self, text: str, x: np.ndarray) -> IntentPrediction:
        p = _softmax((x @ self.W.T + self.b) / self.T)
        order = np.argsort(-p)
        top, second = int(order[0]), int(order[1])
        return IntentPrediction(
            intent=self.classes[top], score=round(float(p[top]), 4), margin=round(float(p[top] - p[second]), 4),
            top_k=[(self.classes[int(i)].value, round(float(p[int(i)]), 4)) for i in order[:3]],
            negation_score=round(float(_sigmoid(x @ self.neg_W[0] + self.neg_b[0])), 4),
            negation_cue=negation_cue(text),
            implicit_score=round(float(_sigmoid(x @ self.imp_W[0] + self.imp_b[0])), 4),
            model_version=self.version)

    def predict(self, text: str) -> tuple[IntentPrediction, np.ndarray]:
        x = self.embed([text])[0]
        return self.predict_embedding(text, x), x
