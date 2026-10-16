"""Semantic matching of an utterance against a profile's saved trigger phrases.

Decision rule (thresholds calibrated on the validation profiles by
`evaluation/eval_personalization.py`, stored in artifacts/personalization.json):
  accept if NOT negated, NOT a quoted/meta reference, margin to the best competing command
  >= min_margin, and (cosine >= min_similarity  OR  the trigger appears verbatim as whole
  words and cosine >= min_similarity_contained).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..nlu.slots import negation_cue, quoted_reference
from ..schemas_v2 import PersonalMatch
from .store import PersonalCommandStore, normalize_phrase


@dataclass
class MatchThresholds:
    min_similarity: float = 0.80
    min_similarity_contained: float = 0.55
    min_margin: float = 0.05
    calibrated: bool = False

    @classmethod
    def load(cls, path: str | Path) -> "MatchThresholds":
        p = Path(path)
        if not p.is_file():
            return cls()  # conservative uncalibrated defaults; reported as such
        d = json.loads(p.read_text(encoding="utf-8"))
        return cls(d["min_similarity"], d["min_similarity_contained"], d["min_margin"], True)


def contains_phrase(text: str, phrase: str) -> bool:
    t, p = normalize_phrase(text), normalize_phrase(phrase)
    return bool(p) and re.search(rf"(?<!\w){re.escape(p)}(?!\w)", t) is not None


class PersonalCommandMatcher:
    def __init__(self, store: PersonalCommandStore, embed_fn, embedding_model: str, thresholds: MatchThresholds):
        self.store, self.embed_fn, self.model, self.th = store, embed_fn, embedding_model, thresholds

    def candidates(self, profile_id: str, text: str, emb: np.ndarray) -> list[tuple[int, str, float, bool]]:
        """Best (command_id, phrase, cosine, contained) per command, best first."""
        pairs, embs = self.store.phrase_index(profile_id, self.model, self.embed_fn)
        if not pairs:
            return []
        sims = embs @ np.asarray(emb, np.float32)
        best: dict[int, tuple[int, str, float, bool]] = {}
        for (cid, phrase), s in zip(pairs, sims):
            cont = contains_phrase(text, phrase)
            cur = best.get(cid)
            if cur is None or s > cur[2]:
                best[cid] = (cid, phrase, float(s), cont or bool(cur and cur[3]))
            elif cont:
                best[cid] = (*cur[:3], True)
        th = self.th.min_similarity_contained
        return sorted(best.values(), key=lambda c: (c[3] and c[2] >= th, c[2]), reverse=True)

    def match(self, profile_id: str, text: str, emb: np.ndarray) -> PersonalMatch | None:
        cands = self.candidates(profile_id, text, emb)
        if not cands:
            return None
        cid, phrase, sim, contained = cands[0]
        margin = sim - cands[1][2] if len(cands) > 1 else 1.0
        strong = sim >= self.th.min_similarity or (contained and sim >= self.th.min_similarity_contained)
        if not strong:
            return None  # nothing resembling a saved trigger
        cmd = self.store.get(profile_id, cid)
        if cmd is None:
            return None
        reason, accepted = "matched", True
        if negation_cue(text):
            reason, accepted = "negated_reference", False
        elif quoted_reference(text):
            reason, accepted = "quoted_or_meta_reference", False
        elif margin < self.th.min_margin and not contained:
            reason, accepted = "ambiguous_between_commands", False
        return PersonalMatch(command_id=cid, trigger=cmd.trigger, matched_phrase=phrase, similarity=round(sim, 4),
                             margin=round(margin, 4), action=cmd.action, accepted=accepted, reason=reason)
