"""Run the whole V2 build + evaluation chain in order (after scripts/download_models.py).

  python scripts/reproduce_v2.py                 # frozen + fine-tuned MiniLM/e5, eval, promote best
  python scripts/reproduce_v2.py --quick         # MiniLM only (frozen + fine-tuned)

Every step is a normal module you can also run on its own; see README "Reproducing the results".
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True, env={**os.environ, "HF_HUB_OFFLINE": "1"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--promote", default="e5-ft_onnx_int8", help="artifact key to install as the runtime model")
    args = ap.parse_args()
    enc = ["minilm"] if args.quick else ["minilm", "e5"]
    ft = [f"{e}-ft" for e in enc]
    run("bhashadrive.training.prepare_dataset")
    run("bhashadrive.training.export_onnx", "--encoder", *enc)
    run("bhashadrive.training.train_classifier", "--encoder", *enc, "--backend", "torch", "onnx_int8")
    run("bhashadrive.training.finetune_encoder", "--encoder", *enc)
    run("bhashadrive.training.export_onnx", "--encoder", *ft)
    run("bhashadrive.training.train_classifier", "--encoder", *ft, "--backend", "onnx_fp32", "onnx_int8")
    keys = [f"{e}_onnx_int8" for e in enc] + [f"{e}_{b}" for e in ft for b in ("onnx_fp32", "onnx_int8")]
    promote = args.promote if args.promote in keys else keys[-1]
    run("bhashadrive.evaluation.eval_nlu_v2", "--models", *keys, "--promote", promote)
    run("bhashadrive.evaluation.eval_personalization")
    run("bhashadrive.evaluation.eval_asr", "--models", "swift", "apex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
