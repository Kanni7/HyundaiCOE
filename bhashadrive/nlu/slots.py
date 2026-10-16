"""Constrained parameter extraction and safety cues.

These helpers never choose the intent. They only fill the small, typed parameter slots
of the intent the classifier predicted (temperature value, step size, destination text)
and flag explicit negation/meta-reference cues that the policy treats conservatively.
"""
from __future__ import annotations

import re

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

_NUM_WORDS = {
    # English
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "twenty one": 21, "twenty-one": 21, "twenty two": 22, "twenty-two": 22, "twenty three": 23, "twenty-three": 23,
    "twenty four": 24, "twenty-four": 24, "twenty five": 25, "twenty-five": 25, "twenty six": 26, "twenty-six": 26,
    "twenty seven": 27, "twenty-seven": 27, "twenty eight": 28, "twenty-eight": 28, "twenty nine": 29,
    "twenty-nine": 29, "thirty": 30, "thirty one": 31, "thirty two": 32, "ten": 10, "twelve": 12,
    # Romanized Hindi (common spellings)
    "pandrah": 15, "pandra": 15, "solah": 16, "sola": 16, "satrah": 17, "satra": 17, "atharah": 18, "athara": 18,
    "unnis": 19, "unees": 19, "bees": 20, "bis": 20, "ikkis": 21, "ikkees": 21, "baais": 22, "bais": 22,
    "baees": 22, "teis": 23, "tees": 30, "teyis": 23, "chaubis": 24, "chobis": 24, "chaubees": 24,
    "pachchis": 25, "pachis": 25, "pachees": 25, "chhabbis": 26, "chabbis": 26, "sattais": 27, "sattaees": 27,
    "atthais": 28, "athais": 28, "untis": 29, "unatees": 29, "ikattis": 31, "battis": 32,
    # Devanagari
    "पंद्रह": 15, "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19, "बीस": 20, "इक्कीस": 21, "बाईस": 22,
    "तेईस": 23, "चौबीस": 24, "पच्चीस": 25, "छब्बीस": 26, "सत्ताईस": 27, "अट्ठाईस": 28, "उनतीस": 29, "तीस": 30,
}
_WORD_RE = re.compile(r"(?<![\w-])(" + "|".join(sorted(map(re.escape, _NUM_WORDS), key=len, reverse=True)) + r")(?![\w-])",
                      re.IGNORECASE)
_DIGIT_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d)?)(?![\d])")
_AFTER_ANCHOR = re.compile(r"^\s*(?:degree|°|celsius|डिग्री|सेल्सियस|pe\b|par\b|पे|पर)", re.IGNORECASE)
_BEFORE_ANCHOR = re.compile(r"(?:\bto|\bat|\bon|\bko)\s*$", re.IGNORECASE)


def extract_numbers(text: str, with_end: bool = False) -> list:
    """All numeric mentions with their character offsets (digits, English, Hindi, Devanagari)."""
    t = text.translate(_DEV_DIGITS)
    out = [(float(m.group(1)), m.start(), m.end()) for m in _DIGIT_RE.finditer(t)]
    out += [(float(_NUM_WORDS[m.group(1).lower()]), m.start(), m.end()) for m in _WORD_RE.finditer(t)]
    out.sort(key=lambda x: x[1])
    return out if with_end else [(v, s) for v, s, _ in out]


def extract_temperature(text: str) -> float | None:
    """Target temperature for climate_temperature_set. Returns None if absent or ambiguous."""
    nums = extract_numbers(text, with_end=True)
    if not nums:
        return None
    if len({v for v, _, _ in nums}) == 1:
        return nums[0][0]
    # Several different numbers: prefer one directly anchored as a temperature, else give up (ask).
    t = text.translate(_DEV_DIGITS)
    anchored = [v for v, s, e in nums if _AFTER_ANCHOR.search(t[e:]) or _BEFORE_ANCHOR.search(t[:s])]
    return anchored[-1] if len(set(anchored)) == 1 else None


_SMALL = re.compile(r"\b(thoda|thodi|thoda sa|zara|jara|halka|halki|slightly|a (little )?bit|a little|a tad|"
                    r"little|bit)\b|थोड़ा|थोड़ी|थोडा|ज़रा|जरा|हल्का", re.IGNORECASE)
_LARGE = re.compile(r"\b(a lot|way|much|full|ekdum|max(imum)?|bahut zyada|kaafi|bohot|bahut)\b|बहुत|काफ़ी|काफी|पूरा",
                    re.IGNORECASE)


def extract_amount(text: str) -> str | None:
    if _SMALL.search(text):
        return "small"
    if _LARGE.search(text):
        return "large"
    return None


_DEST_PATTERNS = [
    re.compile(r"(?:navigate|navigation|directions?|route|take me|drive me|go|get me|head|way)\s+(?:to|towards)\s+(?:the\s+)?(?P<d>.+)", re.I),
    re.compile(r"(?:find|search for|search|look for|show me|locate)\s+(?:the\s+)?(?:nearest\s+|closest\s+|a\s+|an\s+)?(?P<d>.+)", re.I),
    re.compile(r"(?:nearest|closest)\s+(?P<d>.+)", re.I),
    re.compile(r"(?P<d>.+?)\s+(?:le chalo|lekar chalo|le chal|chalo|chalna hai|chaliye|jaana hai|jana hai|jaana|jana|"
               r"tak chalo|tak le chalo|ka rasta|ka raasta|ki taraf|ki direction|ka route|ke liye navigation)\b", re.I),
    re.compile(r"(?P<d>.+?)\s*(?:ले चलो|ले चलिए|चलो|चलना है|जाना है|का रास्ता|की तरफ|तक)", re.I),
    re.compile(r"(?:to|towards)\s+(?P<d>.+)", re.I),
]
_DEST_STRIP = re.compile(r"^(?:bhai|yaar|please|pls|zara|mujhe|hume|humein|hamein|मुझे|हमें|ok|okay|hey|car)[\s,]+|"
                         r"[\s,]+(?:please|pls|yaar|bhai|now|abhi|jaldi|asap|na)$|[.!?।]+$", re.I)
_DEST_STOP = {"", "there", "it", "home?", "somewhere", "wahan", "vahan", "waha", "kahin"}


def extract_destination(text: str) -> str | None:
    for pat in _DEST_PATTERNS:
        m = pat.search(text)
        if m:
            d = m.group("d").strip(" ,.")
            for _ in range(3):
                d = _DEST_STRIP.sub("", d).strip(" ,.")
            d = re.sub(r"\s+(?:ke paas|ke pass|nearby|near me|around here)$", "", d, flags=re.I).strip()
            if d.lower() not in _DEST_STOP and len(d) <= 120:
                return d
    return None


# Explicit negation of the requested action. "stop" and a bare "not"/"nahi" are deliberately
# absent: "stop the music" is a pause request and "not the temperature, the fan" is a correction.
_NEG_CUES = re.compile(
    r"\b(don'?t|do not|dont|never|no need to|shouldn'?t|should not|mustn'?t|must not|without)\b|"
    r"\b(mat|matt)\b|\bna\s+(karo|karna|karein|kare|kar)\b|\bnahi\s+(karna|karo|chahiye|chahta|chahti|karein)\b|"
    r"\bnahin\s+(karna|karo|chahiye)\b|\bzaroorat nahi\b|\bjarurat nahi\b|"
    r"मत|न\s+करें|न\s+करो|नहीं\s+करना|नहीं\s+करो|नहीं\s+चाहिए|ज़रूरत नहीं|जरूरत नहीं",
    re.IGNORECASE)


def negation_cue(text: str) -> str | None:
    m = _NEG_CUES.search(text)
    return m.group(0) if m else None


_QUOTED = re.compile(r"(?:^|\s)[\"“‘'][^\"”’']{2,}[\"”’'](?=\s|$|[,.?!।])")
_META = re.compile(r"\b(what does|what is|what's|whats|kya karta|kya hai|matlab|means?|meaning|delete|remove|forget|"
                   r"rename|change|update|correct|list|explain|batao|bhool|hata)\b|क्या करता|मतलब|हटा|भूल", re.I)


def quoted_reference(text: str) -> bool:
    """True when the utterance talks *about* a phrase (quoted or meta question) instead of using it."""
    return bool(_QUOTED.search(text) or _META.search(text))
