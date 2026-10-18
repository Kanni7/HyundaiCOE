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
        has_action_cue = bool(re.search(r"\b(karo|kar|kardo|chalao|chala|chalu|badha|kam|set|please|de|do|on|off)\b", text, re.I))
        is_implicit = any(p.search(text) for p in self.implicit) and not has_action_cue
        return IntentPrediction(intent=intent, score=score, margin=score - (1 - score) / 2,
                                top_k=[(intent.value, score), ("unsupported", round((1 - score) / 2, 3))],
                                negation_score=0.9 if negation_cue(text) else 0.05, negation_cue=negation_cue(text),
                                implicit_score=0.9 if is_implicit else 0.1,
                                model_version=self.version)


DEFAULT_RULES = [
    # Dialog confirmations & cancellations
    (r"^(?:yes|haan|ha|theek hai|ok|okay|sure|bilkul|confirm|correct|हाँ|ठीक है)\b", "dialog_confirm", 0.95),
    (r"^(?:no|nahi|na|cancel|rehne do|mat karo|chhod|leave it|abort|नहीं|रहने दो)\b", "dialog_cancel", 0.95),

    # Unsupported commands (out of automotive scope - safety protection against random guesses)
    (r"\b(sunroof|window|khidki|darwaza|door|brakes?|accelerat\w*|speed\s+up\s+the\s+car|steering|horn|wipers?|mirrors?|seats?|massage|airbags?|calls?|phone|weather|jokes?|cricket|score|pizza|emails?|alarm)\b|खिड़की|दरवाज़|ब्रेक|मौसम|चुटकुला", "unsupported", 0.95),

    # Temperature explicit set (e.g. 22 degrees)
    (r"(?:set\s*.*(?:temp|cabin|climate)|temperature\s*.*(?:to|pe|set)|\bac\b.*\b(?:1[6-9]|2[0-9]|3[0-2])\b|\b(?:1[6-9]|2[0-9]|3[0-2])\s*(?:pe|par|degree|degrees|celsius|°c)?\s*(?:set|rakho|kar|karo|fix)|cabin\s*to\s*\d+|temperature\s*set|तापमान.*\d|एसी.*\d.*(?:पर|पे)|\d+\s*डिग्री)", "climate_temperature_set", 0.9),

    # Media next / previous (checked BEFORE play/pause so 'agla gaana' doesn't trigger play)
    (r"\b(next|agla|agle|skip)\b.*(?:song|track|gaana|gana)?|change\s+the\s+song|अगला\s*गाना|गाना\s*बदलो", "media_next", 0.9),
    (r"\b(previous|pichla|pichle|pichhla|back)\b.*(?:song|track|gaana|gana)?|pehle\s*wala|पिछला\s*गाना|पहले\s*वाला", "media_previous", 0.9),

    # Fan speed explicit up / down
    (r"(?:fan|blower|hawa|airflow|pankh\w*).*(?:badha|tez|fast|high|up|increase|full|more)|(?:crank|turn)\s+up.*(?:fan|blower)|(?:more|stronger)\s+airflow|पंखा.*(?:तेज़|बढ़ा)|हवा.*(?:तेज़|बढ़ा)", "climate_fan_speed_up", 0.9),
    (r"(?:fan|blower|hawa|airflow|pankh\w*).*(?:kam|ghata|slow|dheema|dheemi|down|decrease|low|halki|soft|reduce|3d\s*kam)|(?:turn|make).*(?:down|slower|quieter).*(?:fan|blower)|less\s+airflow|पंखा.*(?:धीमा|कम)|हवा.*(?:कम|धीमी)", "climate_fan_speed_down", 0.9),

    # Climate On / Off (off checked first)
    (r"(?:ac|climate|air\s*con\w*|air\s*condition\w*|cooling).*(?:off|band|stop|shut|kill|disable|roko)|(?:turn|switch|shut).*(?:off).*(?:ac|climate|air\s*con)|एसी.*(?:बंद|ऑफ)", "climate_off", 0.9),
    (r"(?:ac|climate|air\s*con\w*|air\s*condition\w*|cooling|thandi\s*hawa).*(?:on|chalu|chala|start|active|lagao|khol)|(?:turn|switch|get|start).*(?:ac|climate|air\s*con|cooling)|एसी.*(?:चालू|ऑन|शुरू)", "climate_on", 0.9),

    # Temperature down (cooler) - includes hot/boiling/garmi complaints or cool/thanda requests (checked before thand)
    (r"\b(cool\w*|chilled|colder|cooling|thanda|thandi)\b|make\s+it\s+(?:cool|cold)|thoda\s+(?:cool|thanda)|cabin.*cool|(?:decrease|kam|ghata|lower|drop|down).*(?:temp|cabin)|(?:temp\w*|cabin).*(?:kam|ghata|lower|down)|\bhot\b|boiling|sweat\w*|garmi\w*|garam\s+hai|gami\b|pasina|गर्मी|उमस|ठंडा|गाड़ी\s+ठंडी|पसीना", "climate_temperature_down", 0.9),

    # Temperature up (warmer) - includes cold/freezing/thand complaints or warm requests
    (r"\b(warm\w*|heater|hotter|heat\s*up)\b|make\s+it\s+(?:warm|hot)|thoda\s+(?:warm|garam)|cabin.*warm|(?:increase|badha|raise|up|high).*(?:temp|cabin)|(?:temp\w*|cabin).*(?:badha|increase|up|high)|\bcold\b|freezing|chilly|\bthand\b|\bthandi\s*lag|\bthand\s*lag|sardi|kaanp|shiver|गाड़ी\s+गरम|gadi\s+garam|ठंड|सर्दी", "climate_temperature_up", 0.9),

    # Volume up / down
    (r"(?:volume|awaaz|aawaz|awaz|sound|audio).*(?:up|badha|bada|tez|high|full|increase|raise|oonchi|more|didi\s*mika)|(?:pump\s*up|raise|increase|turn\s*up).*(?:volume|sound|music)|louder|आवाज़.*(?:बढ़ा|तेज़)|आवाज.*(?:बढ़ा|तेज)|वॉल्यूम.*बढ़ा", "volume_up", 0.9),
    (r"(?:volume|awaaz|aawaz|awaz|sound|audio).*(?:down|kam|ghata|low|decrease|lower|reduce|dheemi|dheema|halka)|(?:lower|decrease|reduce|turn\s*down).*(?:volume|sound|music)|quieter|soft\w*|mute|आवाज़.*(?:कम|धीमी|घटा)|आवाज.*(?:कम|धीमी|घटा)|वॉल्यूम.*कम", "volume_down", 0.9),

    # Media pause / play (pause checked first)
    (r"(?:pause|stop).*(?:music|song|track|playback)?|(?:gaana|gana|music|song|geet|radio).*(?:roko|band|pause|stop|mute)|(?:roko|chup\s*ho|chup\s*raho|shant)|kill\s+the\s+music|गाना.*(?:रोको|रोक|बंद)|संगीत.*बंद", "media_pause", 0.9),
    (r"(?:play|resume|unpause|start).*(?:music|song|track|tunes|playback|radio)?|(?:gaana|gana|music|song|geet|radio|kuch).*(?:chala|chalu|baja|lagao|start|play|resume|shuru|on)|hit\s+play|गाना.*(?:चलाओ|बजाओ|लगाओ|शुरू)|संगीत.*(?:शुरू|बजाओ)", "media_play", 0.9),

    # Navigation
    (r"(?:navigate|take\s+me|le\s+chalo|rasta|route|directions?|directions\s+to|drive\s+to|map\s+to|kahan\s+hai)\b|ले\s*चलो|रास्ता|दिशा", "navigation_destination_request", 0.9),

    # Isolated fan mention (deliberately low confidence for clarification)
    (r"^\s*(?:fan|hawa|blower|पंखा)\s*$", "climate_fan_speed_up", 0.55),

    # Ambiguous commands and isolated domain mentions
    (r"^\s*(?:temperature|taapmaan|cabin\s+temperature|ac\s+temperature|तापमान)(?:\s+(?:please|bhai|yaar))?\s*$", "ambiguous", 0.9),
    (r"^\s*(?:volume|awaaz|aawaz|awaz|sound)(?:\s+(?:please|bhai|yaar))?\s*$", "ambiguous", 0.9),
    (r"^\s*(?:music|gaana|gana|song)(?:\s+(?:please|bhai|yaar))?\s*$", "ambiguous", 0.9),
    (r"^\s*(?:ac|air\s*con\w*)(?:\s+(?:please|bhai|yaar))?\s*$", "ambiguous", 0.9),
    (r"^(?:change it|adjust|something|कुछ बदलो|बदलो)$", "ambiguous", 0.9),
]
DEFAULT_IMPLICIT = (
    r"freezing|i'm cold|it's (?:hot|cold|freezing|chilly|boiling)|sweating|thand lag rahi|garmi lag rahi|sardi lag rahi|pasina aa raha|बहुत ठंड|बहुत गर्मी",
)


def make_pipeline(store_path=":memory:", thresholds: MatchThresholds | None = None, model=None):
    model = model or FakeIntentModel()
    store = PersonalCommandStore(store_path)
    matcher = PersonalCommandMatcher(store, model.embed, model.version,
                                     thresholds or MatchThresholds(0.8, 0.5, 0.05, True))
    return BhashaDrivePipelineV2(model, MockVehicleAdapter(), None, AppConfig(), store, matcher, "test-user")
