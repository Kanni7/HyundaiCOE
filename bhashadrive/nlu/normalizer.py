"""Automotive speech normalizer: phonetic correction and anti-hallucination.

Standardizes noisy ASR transcripts (from Whisper / local engines) before intent classification,
ensuring phonetic variations and ASR slips map to canonical commands while preserving scripts
and negation words.
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari digits to ASCII digits
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# Whisper hallucination boilerplate (common in internet-trained models during silence or background noise)
_HALLUCINATION_PATTERNS = [
    re.compile(r"\b(?:thank\s+you(?:\s+very\s+much|\s+for\s+watching)?|thanks\s+for\s+watching)\b[.!?,]*", re.I),
    re.compile(r"\b(?:subtitles?\s+by|please\s+subscribe|like\s+and\s+subscribe)\b[.!?,]*", re.I),
    re.compile(r"\b(?:bye\s+bye|goodbye)\b[.!?,]*", re.I),
    re.compile(r"^[.!?,\s]+$", re.I),
]

# Common ASR mishearings, slips, and spelling variations in Hinglish / Indic speech
# Maps variant pattern -> canonical replacement
_PHONETIC_REPLACEMENTS: list[tuple[re.Pattern, str]] = [
    # ASR slips for colloquial greeting "bhai"
    (re.compile(r"^(?:बाय|बाई)[\s,]+", re.I), "bhai, "),
    # ASR slips for "badha do" / "badhao" (increase)
    (re.compile(r"\b(?:butadu|batado|bata\s+do|bada\s+de|bada\s+do|badha\s+de|badhao(?:\s+karo|\s+kar\s+do)?|badhana)\b", re.I), "badha do"),
    # ASR slips for "kam karo" / "ghatao" (decrease)
    (re.compile(r"\b(?:kamti(?:\s+karo|\s+kar\s+do)?|ghata\s+do|ghatao(?:\s+karo|\s+kar\s+do)?|dheeme\s+karo|dheema\s+karo|halka\s+karo)\b", re.I), "kam karo"),
    # ASR slips for "thoda" (a little bit)
    (re.compile(r"\b(?:toda|thodi|thoda\s+sa|thodi\s+si|jara|zara\s+sa)\b", re.I), "thoda"),
    # ASR slips for "pe kar do" / "pe set kar do" (set to)
    (re.compile(r"\b(?:pika\s+do|pika\s+de|pe\s+kardo|par\s+kardo|par\s+kar\s+do|pe\s+set\s+kardo)\b", re.I), "pe kar do"),
    # ASR slips for "chalu" / "on" (turn on)
    (re.compile(r"\b(?:chalu\s+kar\s+do|chala\s+do|chalu\s+karo|start\s+kar\s+do|start\s+karo|on\s+kardo)\b", re.I), "on karo"),
    # ASR slips for "band" / "off" (turn off)
    (re.compile(r"\b(?:bandh\s+kar\s+do|bandh\s+karo|band\s+kardo|off\s+kardo|switch\s+off\s+kar\s+do)\b", re.I), "band karo"),
    # ASR slips for "awaaz" (sound/volume)
    (re.compile(r"\b(?:aawaz|aawaj|awaz|awaaj|avaz)\b", re.I), "awaaz"),
    # ASR slips for "gaana" (song/music)
    (re.compile(r"\b(?:gana|geet|gane)\b", re.I), "gaana"),
    # ASR slips for "pankha" / "fan"
    (re.compile(r"\b(?:pankhe|pankha)\b", re.I), "fan"),
    # ASR slips for "taapmaan" / "temperature"
    (re.compile(r"\b(?:taapmaan|tapman|temprature|tempreture)\b", re.I), "temperature"),
    # ASR slips for "agla" (next)
    (re.compile(r"\b(?:agle|agla\s+track|next\s+track)\b", re.I), "next"),
    # ASR slips for "pichla" (previous)
    (re.compile(r"\b(?:pichhla|pichle|pichla\s+track|previous\s+track)\b", re.I), "previous"),
    # Additional Whisper English-mode slips for Indic accents
    (re.compile(r"\ba\s+vast\b", re.I), "awaaz"),
    (re.compile(r"\bhodhi\b", re.I), "thodi"),
    (re.compile(r"\bdhimi\b", re.I), "dheemi"),
    (re.compile(r"\b(?:gaani|gami)\s*(?:se|sae)?\b", re.I), "garmi se"),
    (re.compile(r"\b(?:ni\s*kaal|neekal)\b", re.I), "nikal"),
    (re.compile(r"\bwarm-cut\b", re.I), "warm kar"),
    (re.compile(r"\byar\b", re.I), "yaar"),
    (re.compile(r"\b(?:ko\s+pos|kopos)\b", re.I), "pause"),
    (re.compile(r"\bpos\b", re.I), "pause"),
    (re.compile(r"\bvahi\b", re.I), "bhai"),
    (re.compile(r"\bshouru\b", re.I), "shuru"),
    (re.compile(r"\bdikhao\s*do\b", re.I), "badha do"),
    (re.compile(r"\bto\s*da\b", re.I), "thoda"),
    (re.compile(r"\bkaibin\b", re.I), "cabin"),
    (re.compile(r"\b(?:tanda|kanda)\b", re.I), "thanda"),
    (re.compile(r"\b(?:volin|voleum|valim)\b", re.I), "volume"),
    (re.compile(r"\b(?:ndal|thal|hand)\s+lag\s+rahi\b", re.I), "thand lag rahi"),
    (re.compile(r"\b(?:per\s*milagri\w*|milagriya)\b", re.I), "garmi lag rahi"),
    (re.compile(r"\b(?:bhootayal|bhootal)\b", re.I), "bahut zyada"),
]


_DEV_PHRASE_TRANSLITERATION: list[tuple[re.Pattern, str]] = [
    (re.compile(r"आवाज़\s*(?:बढ़ा\s*दो|बढ़ाओ|तेज़\s*करो)", re.I), "volume badha do"),
    (re.compile(r"आवाज़\s*(?:कम\s*करो|धीमी\s*करो|घटाओ)", re.I), "volume kam karo"),
    (re.compile(r"आवाज़|आवाज|वॉल्यूम", re.I), "volume"),
    (re.compile(r"बढ़ा\s*दो|बढ़ाओ|तेज़\s*करो", re.I), "badha do"),
    (re.compile(r"कम\s*करो|धीमी\s*करो|घटाओ", re.I), "kam karo"),
    (re.compile(r"एसी\s*(?:चालू|ऑन|शुरू)\s*(?:करो|कर\s*दो)?", re.I), "AC on karo"),
    (re.compile(r"एसी\s*(?:बंद|ऑफ)\s*(?:करो|कर\s*दो)?", re.I), "AC off karo"),
    (re.compile(r"एसी", re.I), "AC"),
    (re.compile(r"गाना\s*(?:चलाओ|बजाओ|लगाओ|शुरू\s*करो)", re.I), "play music"),
    (re.compile(r"गाना\s*(?:रोको|रोक\s*दो|बंद\s*करो)", re.I), "pause music"),
    (re.compile(r"गाना|संगीत", re.I), "music"),
    (re.compile(r"अगला\s*गाना|अगला", re.I), "next song"),
    (re.compile(r"पिछला\s*गाना|पिछला", re.I), "previous song"),
    (re.compile(r"पंखा\s*(?:तेज़\s*करो|बढ़ाओ)", re.I), "fan speed badha do"),
    (re.compile(r"पंखा\s*(?:धीमा\s*करो|कम\s*करो)", re.I), "fan speed kam karo"),
    (re.compile(r"पंखा|पंखे", re.I), "fan"),
    (re.compile(r"हवा", re.I), "hawa"),
    (re.compile(r"तापमान", re.I), "temperature"),
    (re.compile(r"डिग्री", re.I), "degree"),
    (re.compile(r"सेल्सियस", re.I), "celsius"),
    (re.compile(r"गर्मी", re.I), "garmi"),
    (re.compile(r"ठंड|सर्दी", re.I), "thand"),
    (re.compile(r"लग\s*रही\s*है", re.I), "lag rahi hai"),
    (re.compile(r"कनॉट\s*प्लेस", re.I), "Connaught Place"),
    (re.compile(r"इंडिया\s*गेट", re.I), "India Gate"),
    (re.compile(r"ले\s*चलो", re.I), "le chalo"),
    (re.compile(r"रास्ता\s*दिखाओ|रास्ता", re.I), "rasta dikhao"),
    (re.compile(r"हाँ|ठीक\s*है", re.I), "yes"),
    (re.compile(r"नहीं|रहने\s*दो", re.I), "no cancel"),
    (re.compile(r"भाई", re.I), "bhai"),
    (re.compile(r"थोड़ा|थोड़ी|ज़रा|जरा", re.I), "thoda"),
    (re.compile(r"बहुत|काफ़ी|काफी", re.I), "bahut"),
]

_DEV_CHAR_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "ा": "a", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "्": "", "़": "", "ँ": "n", "ः": "h",
}


def to_english(text: str) -> str:
    """Ensure any transcript or text is rendered in English (Latin alphabet)."""
    if not text:
        return ""
    # Quick check: does it contain any Devanagari character?
    if not any("\u0900" <= ch <= "\u097f" for ch in text):
        return text
    # Apply high-precision phrase replacements
    out = text
    for pat, rep in _DEV_PHRASE_TRANSLITERATION:
        out = pat.sub(rep, out)
    # Character-level transliteration for any residual Devanagari
    chars = [str(_DEV_CHAR_MAP.get(ch, ch)) for ch in out]
    return " ".join("".join(chars).split())


def remove_hallucinations(text: str) -> str:
    """Filter out known trailing subtitle and hallucination strings from ASR models."""
    cleaned = text
    for pat in _HALLUCINATION_PATTERNS:
        cleaned = pat.sub("", cleaned)
    return " ".join(cleaned.split())


def deduplicate_consecutive_words(text: str) -> str:
    """Collapse accidental stuttering/repeating words (e.g. 'ac ac on' -> 'ac on')."""
    words = text.split()
    if len(words) <= 1:
        return text
    deduped = [words[0]]
    for w in words[1:]:
        if w.lower() == deduped[-1].lower() and w.lower() not in {"bahut", "haan", "kam"}:
            continue
        deduped.append(w)
    return " ".join(deduped)


def normalize_phonetics(text: str) -> str:
    """Fix common ASR slips and standardize phonetic variants to canonical in-car keywords."""
    res = text
    for pat, rep in _PHONETIC_REPLACEMENTS:
        res = pat.sub(rep, res)
    return " ".join(res.split())


DOMAIN_ANCHORS: dict[str, list[str]] = {
    "temperature": [
        "temperature", "temprature", "tempreture", "tempretur", "taapmaan",
        "tapman", "tepreture", "tempercher", "temprecher", "tamprature",
        "temperator", "temperatue", "temparature", "temperater",
    ],
    "cabin": ["cabin", "kaibin", "caibin", "caben", "kaybin", "kabin"],
    "blower": ["blower", "bloer", "blour", "blowr", "bloor"],
    "volume": ["volume", "volin", "valim", "voleum", "volum", "bolume", "valume", "valum", "volyoom"],
    "climate": ["climate", "klimat", "claimeit"],
}

_STOP_WORDS_NO_SNAP = {
    "this", "that", "with", "from", "karo", "kard", "bhai", "yaar",
    "some", "have", "more", "less", "good", "need", "like", "will",
    "what", "when", "where", "which", "there", "their", "about",
    "please", "karna", "kardo", "chahiye", "lagao", "chalao", "badhao",
    "gaana", "gana", "geet", "song", "thand", "thanda", "garam", "garmi"
}


def snap_to_automotive_domain(text: str) -> str:
    """
    Rounds off / snaps noisy or phonetically distorted ASR words to canonical
    automotive vocabulary (e.g. 'tempretur' -> 'temperature', 'volin' -> 'volume').
    """
    if not text:
        return ""
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return text

    words = text.split()
    snapped = []
    for w in words:
        clean = w.lower().strip(".,!?;:\"'()[]{}")
        if len(clean) < 4 or clean in _STOP_WORDS_NO_SNAP or clean in DOMAIN_ANCHORS:
            snapped.append(w)
            continue
        matched = False
        for canonical, variants in DOMAIN_ANCHORS.items():
            for v in variants:
                if fuzz.ratio(clean, v) >= 90:
                    prefix = w[:len(w) - len(w.lstrip(".,!?;:\"'()[]{}"))]
                    suffix = w[len(w.rstrip(".,!?;:\"'()[]{}")):]
                    target = canonical
                    if w.isupper():
                        target = target.upper()
                    elif w and w[0].isupper():
                        target = target.capitalize()
                    snapped.append(f"{prefix}{target}{suffix}")
                    matched = True
                    break
            if matched:
                break
        if not matched:
            snapped.append(w)
    return " ".join(snapped)



def normalize_speech_transcript(text: str) -> str:
    """
    Standardize ASR transcripts:
    1. Unicode NFC normalization & control character cleanup
    2. Devanagari numerals to ASCII digits
    3. Removal of ASR hallucination boilerplate
    4. Deduplication of accidental stuttering words
    5. Phonetic slip normalization
    6. Automotive domain snapping (rounds off acoustic slips to nearest in-car keyword)
    """
    if not text:
        return ""
    # 1. Unicode NFC + whitespace cleanup
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cc" or ch in "\n\t")

    # 2. Devanagari digits
    text = text.translate(_DEV_DIGITS)

    # 3. Hallucination boilerplate removal
    text = remove_hallucinations(text)

    # 4. Word deduplication
    text = deduplicate_consecutive_words(text)

    # 5. Phonetic slip replacements
    text = normalize_phonetics(text)

    # 6. Automotive domain snapping
    text = snap_to_automotive_domain(text)

    return " ".join(text.split())

