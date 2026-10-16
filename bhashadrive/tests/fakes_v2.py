"""Deterministic stand-ins for the trained V2 model, for fast unit tests."""
from __future__ import annotations

import re
import zlib

import numpy as np

from bhashadrive.config import AppConfig
from bhashadrive.nlu.slots import negation_cue
from bhashadrive.nlu.taxonomy import TAXONOMY_VERSION, IntentV2
from bhashadrive.personalization.matcher import MatchThresholds, PersonalCommandMatcher
from bhashadrive.personalization.store import PersonalCommandStore
from bhashadrive.pipeline_v2 import BhashaDrivePipelineV2
from bhashadrive.schemas_v2 import IntentPrediction
from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

THRESHOLDS = {"min_score": 0.6, "min_margin": 0.2}


class FakeIntentModel:
    """Scripted predictions by regex; bag-of-words hashing embeddings (cosine ~ word overlap)."""

    def __init__(self, rules: list[tuple[str, str, float]] | None = None, implicit: tuple[str, ...] = (), dim: int = 256):
        self.rules = [(re.compile(p, re.I), IntentV2(i), s) for p, i, s in (rules or DEFAULT_RULES)]
        self.implicit = [re.compile(p, re.I) for p in implicit or DEFAULT_IMPLICIT]
        self.dim = dim
        self.version = "fake-1"
        self.manifest = {"thresholds": THRESHOLDS, "negation_threshold": 0.5, "implicit_threshold": 0.5,
                         "taxonomy_version": TAXONOMY_VERSION, "model_version": self.version,
                         "confidence_meaning": "fake"}

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), np.float32)
        for k, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[k, zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(n, 1e-9, None)

    def predict(self, text):
        x = self.embed([text])[0]
        return self.predict_embedding(text, x), x

    def predict_embedding(self, text, x):
        intent, score = IntentV2.UNSUPPORTED, 0.9
        for pat, it, s in self.rules:
            if pat.search(text):
                intent, score = it, s
                break
        return IntentPrediction(intent=intent, score=score, margin=score - (1 - score) / 2,
                                top_k=[(intent.value, score), ("unsupported", round((1 - score) / 2, 3))],
                                negation_score=0.9 if negation_cue(text) else 0.05, negation_cue=negation_cue(text),
                                implicit_score=0.9 if any(p.search(text) for p in self.implicit) else 0.1,
                                model_version=self.version)


DEFAULT_RULES = [
    (r"^(yes|haan|ok)\b", "dialog_confirm", 0.95),
    (r"^(no|nahi|cancel)\b", "dialog_cancel", 0.95),
    (r"set .*temperature|temperature .*\d|ac \d+ pe|\d+ pe set", "climate_temperature_set", 0.9),
    (r"cold|freezing|warm", "climate_temperature_up", 0.9),
    (r"hot|cool", "climate_temperature_down", 0.9),
    (r"ac off|turn the ac off|ac band", "climate_off", 0.9),
    (r"volume up|volume badha|louder", "volume_up", 0.9),
    (r"volume down|quieter", "volume_down", 0.9),
    (r"pause", "media_pause", 0.9),
    (r"navigate|take me", "navigation_destination_request", 0.9),
    (r"fan", "climate_fan_speed_up", 0.55),  # deliberately low confidence
    (r"change it|adjust", "ambiguous", 0.9),
]
DEFAULT_IMPLICIT = (r"freezing|i'm cold|it's hot",)


def make_pipeline(store_path=":memory:", thresholds: MatchThresholds | None = None, model=None):
    model = model or FakeIntentModel()
    store = PersonalCommandStore(store_path)
    matcher = PersonalCommandMatcher(store, model.embed, model.version,
                                     thresholds or MatchThresholds(0.8, 0.5, 0.05, True))
    return BhashaDrivePipelineV2(model, MockVehicleAdapter(), None, AppConfig(), store, matcher, "test-user")
