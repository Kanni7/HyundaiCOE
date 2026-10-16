from __future__ import annotations

import pytest

from bhashadrive.config import AppConfig
from bhashadrive.nlu.semantic_parser import StaticSemanticParser
from bhashadrive.pipeline import BhashaDrivePipeline
from bhashadrive.schemas import SemanticFrame, Verification
from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter


def frame(intent="none", act="direct_command", rel="none", negated=False, ambiguous=False, **params) -> SemanticFrame:
    return SemanticFrame(speech_act=act, intent=intent, context_relation=rel, negated=negated,
                         ambiguous=ambiguous, parameters=params)


class ScriptedParser(StaticSemanticParser):
    """Static frames plus an optional scripted verifier answer per transcript."""

    def __init__(self, frames, verifications=None):
        super().__init__(frames)
        self.verifications = verifications or {}

    def verify(self, transcript):
        v = self.verifications.get(transcript)
        if isinstance(v, Exception):
            raise v
        return v


@pytest.fixture
def make_pipeline():
    def _make(frames, verifications=None):
        cfg = AppConfig()
        vehicle = MockVehicleAdapter(cfg.limits)
        return BhashaDrivePipeline(ScriptedParser(frames, verifications), vehicle, None, cfg)

    return _make


def V(function, request_type="explicit_request"):
    return Verification(function=function, request_type=request_type)
