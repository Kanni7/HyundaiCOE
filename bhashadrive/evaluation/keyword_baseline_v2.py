"""Keyword-rule baseline for the V2 taxonomy (comparison only; never used at runtime)."""
from __future__ import annotations

import re

from ..nlu.slots import extract_numbers

_R = lambda p: re.compile(p, re.I)  # noqa: E731
RULES = [
    ("dialog_confirm", _R(r"^\W*(yes|yeah|yep|haan|han|ha|हाँ|हां|ok|okay|sure|theek hai|ठीक है|go ahead|correct)\b")),
    ("dialog_cancel", _R(r"^\W*(no|nope|nah|nahi|नहीं|cancel|rehne do|रहने दो|never mind|forget it|leave it)\b")),
    ("navigation_destination_request", _R(r"navigate|take me|directions?|route|le chalo|ले चलो|jaana hai|जाना है|rasta|रास्ता|nearest")),
    ("media_next", _R(r"\bnext\b|skip|agla|अगला|badlo|बदलो")),
    ("media_previous", _R(r"previous|pichla|पिछला|go back|last (song|track)|pehle wala")),
    ("media_pause", _R(r"pause|stop (the )?(music|song|playing)|roko|रोको|रोक दो")),
    ("media_play", _R(r"\bplay\b|resume|chalao|bajao|बजाओ|चलाओ|music (laga|start)")),
    ("volume_up", _R(r"(volume|awaaz|aawaz|आवाज़|आवाज|sound|music).*(up|badha|बढ़ा|tez|तेज़|loud|increase|high|full)|louder")),
    ("volume_down", _R(r"(volume|awaaz|aawaz|आवाज़|आवाज|sound|music).*(down|kam|कम|dheem|low|reduce|soft)|quieter")),
    ("climate_fan_speed_up", _R(r"(fan|blower|pankha|पंखा|hawa|हवा|air).*(up|tez|तेज़|badha|बढ़ा|increase|high|full|more)")),
    ("climate_fan_speed_down", _R(r"(fan|blower|pankha|पंखा|hawa|हवा|air).*(down|kam|कम|dheem|धीमा|slow|low|reduce|less)")),
    ("climate_off", _R(r"\b(ac|aircon|climate|एसी)\b.*\b(off|band|bandh|बंद|shut)\b|turn off the ac")),
    ("climate_on", _R(r"\b(ac|aircon|climate|एसी)\b.*\b(on|chalu|चालू|start)\b|turn on the ac")),
    ("climate_temperature_up", _R(r"warm|garam|गरम|heat|cold|thand|ठंड|sardi|सर्दी|freez|chilly")),
    ("climate_temperature_down", _R(r"cool|thand[ai]|ठंडा|hot|garmi|गर्मी|pasina|पसीना|sweat|boiling")),
]


def predict(text: str) -> str:
    if extract_numbers(text) and re.search(r"temp|ac|एसी|degree|डिग्री|climate|तापमान|taapmaan|cabin", text, re.I):
        return "climate_temperature_set"
    for intent, pat in RULES:
        if pat.search(text):
            return intent
    return "unsupported"
