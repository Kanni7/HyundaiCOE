"""KEYWORD BASELINE — FOR COMPARISON ONLY.

A deliberately simple keyword matcher used to quantify how much the local LLM
adds. It is never used by the application pipeline and is not a fallback.
"""
from __future__ import annotations

import re

from ..nlu.semantic_parser import SemanticParser
from ..schemas import SemanticFrame

_RULES = [
    (r"\b(next|skip|agla)\b", "media_next_track", {}),
    (r"\b(previous|pichla|back)\b", "media_previous_track", {}),
    (r"\b(pause|stop|band karo|rok)", "media_pause", {}),
    (r"\b(play|resume|chalao)\b", "media_play", {}),
    (r"\b(navigate|take me|rasta|directions?)\b", "navigation_search", {}),
    (r"\bac\b.*\b(off|band)\b", "climate_ac_power", {"power": "off"}),
    (r"\bac\b.*\b(on|chalu)\b", "climate_ac_power", {"power": "on"}),
    (r"\b(\d{2})\s*(degrees?|pe)", "climate_set_temperature", {}),
    (r"\bfan\b.*\b(up|increase|tez|badha)", "fan_adjust_speed", {"direction": "increase"}),
    (r"\bfan\b.*\b(down|reduce|lower|kam)", "fan_adjust_speed", {"direction": "decrease"}),
    (r"\b(volume|awaaz)\b.*\b(up|increase|badha|louder)", "volume_adjust", {"direction": "increase"}),
    (r"\b(volume|awaaz)\b.*\b(down|decrease|kam|lower)", "volume_adjust", {"direction": "decrease"}),
    (r"\b(warm|warmer|heat|garam|cold|thand)\b", "climate_adjust_temperature", {"direction": "increase"}),
    (r"\b(cool|cooler|hot|garmi)\b", "climate_adjust_temperature", {"direction": "decrease"}),
]


class KeywordBaselineParser(SemanticParser):
    name = "keyword-baseline"

    def parse(self, transcript: str, context_summary: str) -> SemanticFrame:
        t = transcript.lower()
        for pattern, intent, params in _RULES:
            m = re.search(pattern, t)
            if m:
                params = dict(params)
                if intent == "climate_set_temperature":
                    params["target_temperature_celsius"] = float(m.group(1))
                if intent == "navigation_search":
                    params["destination"] = re.sub(r".*\b(to|for)\b", "", transcript).strip(" .") or transcript
                return SemanticFrame(speech_act="direct_command", intent=intent, parameters=params)
        return SemanticFrame(speech_act="statement", intent="none")
