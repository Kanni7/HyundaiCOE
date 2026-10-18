"""Unit tests for Phonetic Retrieval + Self-Attention Context Rescorer."""
from pathlib import Path
import pytest

from bhashadrive.nlu.intent_model import IntentModel
from bhashadrive.nlu.rescorer import PhoneticAttentionRescorer, get_phonetic_candidates
from bhashadrive.pipeline_v2 import build_v2_pipeline
from bhashadrive.schemas_v2 import DecisionV2


def test_phonetic_candidate_retrieval():
    # 1. Phonetic slip 'bloer' retrieves 'blower' in top candidates
    cands_bloer = [c[0] for c in get_phonetic_candidates("bloer", top_k=5)]
    assert "blower" in cands_bloer

    # 2. Phonetic slip 'volin' retrieves 'volume' in top candidates
    cands_volin = [c[0] for c in get_phonetic_candidates("volin", top_k=5)]
    assert "volume" in cands_volin

    # 3. Phonetic slip 'kaibin' retrieves 'cabin' in top candidates
    cands_kaibin = [c[0] for c in get_phonetic_candidates("kaibin", top_k=5)]
    assert "cabin" in cands_kaibin

    # 4. Phonetic slip 'tempratur' retrieves 'temperature'
    cands_temp = [c[0] for c in get_phonetic_candidates("tempratur", top_k=5)]
    assert "temperature" in cands_temp


def test_self_attention_rescoring():
    root = Path(__file__).resolve().parents[2]
    model = IntentModel.load(root / "artifacts" / "nlu" / "runtime", root)
    rescorer = PhoneticAttentionRescorer(model)

    # Contextual resolution: in "bhai bloer thoda badha do", self-attention
    # determines "blower" is the coherent in-car candidate (fan speed up)
    rescored, meta = rescorer.rescore("bhai bloer thoda badha do")
    assert "blower" in rescored
    assert meta["rescored"] is True
    assert meta["target_word"] == "bloer"

    # Preserves clean sentences untouched
    clean = "turn on the AC"
    rescored_clean, meta_clean = rescorer.rescore(clean)
    assert rescored_clean == clean
    assert meta_clean["rescored"] is False


def test_pipeline_integration_with_rescorer():
    pipe = build_v2_pipeline(with_asr=False)

    # When the driver says a distorted phrase, the rescorer recovers canonical words
    # and executes the intended automotive action seamlessly
    r = pipe.process_text("bhai bloer thoda badha do")
    assert r.decision == DecisionV2.EXECUTE
    assert "Fan speed" in r.action_result.message
