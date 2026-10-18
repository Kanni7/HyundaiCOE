"""Tests verifying all 4 solutions:
1. Whisper domain priming and VAD anti-hallucination settings
2. Pre-encoder phonetic and spelling normalization (ASR slip fixes)
3. Devanagari script canonicalization across all vehicle actions
4. Elimination of fake guesses and resolution of the garmi/thand inversion
"""
import pytest

from bhashadrive.config import ASRConfig, DEFAULT_INCAR_PROMPT
from bhashadrive.nlu.normalizer import (
    deduplicate_consecutive_words,
    normalize_phonetics,
    normalize_speech_transcript,
    remove_hallucinations,
    to_english,
)
from bhashadrive.nlu.taxonomy import IntentV2
from bhashadrive.schemas_v2 import DecisionV2
from bhashadrive.tests.fakes_v2 import make_pipeline


# ---- Solution 1: Domain Priming & VAD anti-hallucination configuration ----
def test_solution1_asr_config_defaults():
    cfg = ASRConfig()
    assert "AC" in cfg.initial_prompt and "volume" in cfg.initial_prompt
    assert cfg.language in ("en", None)
    assert cfg.beam_size in (1, 5)  # Fast greedy decode or beam search
    assert cfg.vad_filter is True
    assert cfg.vad_parameters["min_silence_duration_ms"] >= 300
    assert cfg.vad_parameters["threshold"] in (0.35, 0.5)
    assert cfg.no_speech_threshold == 0.6


def test_solution1_hallucination_filtering():
    raw_subtitles = "thank you for watching"
    assert remove_hallucinations(raw_subtitles) == ""
    assert normalize_speech_transcript("subtitles by like and subscribe") == ""
    # Authentic command with trailing thank you
    cleaned = normalize_speech_transcript("AC on kar do thank you for watching")
    assert "thank you" not in cleaned and "on kar do" in cleaned


# ---- Solution 2: Phonetic & Spelling Normalization Layer ----
def test_solution2_phonetic_normalization():
    assert normalize_phonetics("volume butadu") == "volume badha do"
    assert normalize_phonetics("AC pika do") == "AC pe kar do"
    assert normalize_phonetics("toda warm kar do") == "thoda warm kar do"
    assert normalize_phonetics("awaaz kamti karo") == "awaaz kam karo"
    assert normalize_phonetics("aawaz badha de") == "awaaz badha do"
    assert deduplicate_consecutive_words("ac ac on kar do") == "ac on kar do"


def test_solution2_pipeline_handles_asr_slips():
    p = make_pipeline()
    # "volume butadu" normalized to "volume badha do" -> triggers volume_up
    r = p.process_text("volume butadu")
    assert r.decision == DecisionV2.EXECUTE
    assert r.action.intent == IntentV2.VOLUME_UP
    assert p.vehicle.state["volume"] == 12

    # "toda warm kar do" normalized to "thoda warm kar do" -> triggers climate_temperature_up
    r2 = p.process_text("toda warm kar do")
    assert r2.decision == DecisionV2.EXECUTE
    assert r2.action.intent == IntentV2.CLIMATE_TEMPERATURE_UP
    assert p.vehicle.state["cabin_temperature_c"] == 23


# ---- Solution 3: Multi-Script Devanagari Canonicalization ----
@pytest.mark.parametrize("dev_text,expected_intent", [
    ("आवाज़ बढ़ा दो", IntentV2.VOLUME_UP),
    ("आवाज़ कम करो", IntentV2.VOLUME_DOWN),
    ("एसी चालू करो", IntentV2.CLIMATE_ON),
    ("एसी बंद करो", IntentV2.CLIMATE_OFF),
    ("गाना चलाओ", IntentV2.MEDIA_PLAY),
    ("गाना रोको", IntentV2.MEDIA_PAUSE),
    ("अगला गाना", IntentV2.MEDIA_NEXT),
    ("पिछला गाना", IntentV2.MEDIA_PREVIOUS),
    ("पंखा तेज़ करो", IntentV2.CLIMATE_FAN_SPEED_UP),
    ("पंखा धीमा करो", IntentV2.CLIMATE_FAN_SPEED_DOWN),
])
def test_solution3_devanagari_commands_execute(dev_text, expected_intent):
    p = make_pipeline()
    r = p.process_text(dev_text)
    assert r.decision == DecisionV2.EXECUTE
    assert r.action.intent == expected_intent


def test_solution3_devanagari_navigation():
    p = make_pipeline()
    r = p.process_text("कनॉट प्लेस ले चलो")
    assert r.decision == DecisionV2.EXECUTE
    assert r.action.intent == IntentV2.NAVIGATION_DESTINATION_REQUEST
    assert "कनॉट प्लेस" in r.action.parameters.destination


def test_solution3_to_english_conversion():
    assert to_english("आवाज़ बढ़ा दो") == "volume badha do"
    assert to_english("एसी बंद करो") == "AC off karo"
    assert to_english("गाना चलाओ") == "play music"
    assert to_english("कनॉट प्लेस ले चलो") == "Connaught Place le chalo"
    assert to_english("Bhai, AC thoda warm kar do") == "Bhai, AC thoda warm kar do"


# ---- Solution 4: Fixing Inversion and Eliminating Fake Guesses ----
def test_solution4_garmi_thand_intent_polarity():
    p = make_pipeline()
    # "garmi lag rahi hai" means hot -> cabin cooler (climate_temperature_down)
    r = p.process_text("garmi lag rahi hai")
    assert r.intent == IntentV2.CLIMATE_TEMPERATURE_DOWN
    assert r.decision == DecisionV2.ASK_CONFIRMATION  # implicit complaint triggers confirmation
    assert r.action.parameters.direction == "cooler"

    # "thand lag rahi hai" means cold -> cabin warmer (climate_temperature_up)
    r2 = p.process_text("thand lag rahi hai")
    assert r2.intent == IntentV2.CLIMATE_TEMPERATURE_UP
    assert r2.decision == DecisionV2.ASK_CONFIRMATION
    assert r2.action.parameters.direction == "warmer"


@pytest.mark.parametrize("fake_phrase", [
    "open the sunroof",
    "khidki khol do",
    "roll down the window",
    "apply the brakes",
    "tell me a joke",
    "what is the capital of France",
    "apple banana orange",
    "random noise blah blah",
    "completely unrelated chit chat",
])
def test_solution4_eliminates_fake_guesses(fake_phrase):
    p = make_pipeline()
    r = p.process_text(fake_phrase)
    # Must never execute any vehicle action on out-of-domain or random words
    assert r.decision != DecisionV2.EXECUTE
    assert r.action is None
    assert p.vehicle.log == []
    assert r.decision in (DecisionV2.REJECT, DecisionV2.CLARIFY)
