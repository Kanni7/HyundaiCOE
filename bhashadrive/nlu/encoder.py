"""Sentence encoders.

`OnnxSentenceEncoder` is the runtime implementation: onnxruntime + the Rust `tokenizers`
library, mean pooling and L2 normalisation in numpy. It needs no PyTorch, loads once at
startup and only reads local files.

`TorchSentenceEncoder` (sentence-transformers) is the reference implementation used for
training, ONNX export and agreement checks. It is not used by the runtime.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class EncoderUnavailableError(RuntimeError):
    """The local encoder could not be loaded. There is no remote fallback."""


class SentenceEncoder(ABC):
    name = "abstract"
    dim = 0

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (n, dim)."""


def _prefix_for(model_dir: Path) -> str:
    # multilingual-e5 expects a task prefix; MiniLM does not.
    return "query: " if "e5" in model_dir.name.lower() else ""


def _normalise(x: np.ndarray) -> np.ndarray:
    return (x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


class OnnxSentenceEncoder(SentenceEncoder):
    def __init__(self, onnx_path: str | Path, tokenizer_dir: str | Path, threads: int = 2, max_length: int = 64):
        onnx_path, tokenizer_dir = Path(onnx_path), Path(tokenizer_dir)
        if not onnx_path.is_file():
            raise EncoderUnavailableError(f"Encoder ONNX file not found: {onnx_path}. Run training/export_onnx.py.")
        if not (tokenizer_dir / "tokenizer.json").is_file():
            raise EncoderUnavailableError(f"tokenizer.json not found in {tokenizer_dir}. Run scripts/download_models.py.")
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise EncoderUnavailableError(f"onnxruntime/tokenizers not installed: {exc}") from exc
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
            self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        except Exception as exc:
            raise EncoderUnavailableError(f"Could not load encoder {onnx_path}: {exc}") from exc
        self.tokenizer.enable_truncation(max_length)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("<pad>") or 1, pad_token="<pad>")
        self._inputs = {i.name for i in self.session.get_inputs()}
        self.prefix = _prefix_for(tokenizer_dir)
        self.name = f"onnx:{onnx_path.name}"
        self.dim = int(self.session.get_outputs()[0].shape[-1]) if isinstance(
            self.session.get_outputs()[0].shape[-1], int) else 384

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        enc = self.tokenizer.encode_batch([self.prefix + t for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self.session.run(None, feeds)[0]  # (n, seq, dim)
        m = mask[..., None].astype(np.float32)
        pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        return _normalise(pooled)


class TorchSentenceEncoder(SentenceEncoder):
    """Reference encoder using transformers (no sentence-transformers package required)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device)
        self.model.eval()
        self.prefix = _prefix_for(model_dir)
        self.name = f"torch:{model_dir.name}"
        self.dim = getattr(self.model.config, "hidden_size", 384)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        import torch

        all_pooled = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = [self.prefix + t for t in texts[i:i + batch_size]]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                mask = inputs["attention_mask"].unsqueeze(-1)
                pooled = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                all_pooled.append(pooled.cpu().numpy())
        res = np.vstack(all_pooled) if all_pooled else np.zeros((0, self.dim), np.float32)
        return _normalise(res)


def load_encoder_manifest(artifacts_dir: str | Path) -> dict:
    p = Path(artifacts_dir) / "manifest.json"
    if not p.is_file():
        raise EncoderUnavailableError(f"NLU manifest not found: {p}. Run the training pipeline (see README).")
    return json.loads(p.read_text(encoding="utf-8"))
