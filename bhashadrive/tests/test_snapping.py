from bhashadrive.nlu.normalizer import snap_to_automotive_domain, normalize_speech_transcript
from bhashadrive.pipeline_v2 import build_v2_pipeline
from bhashadrive.schemas_v2 import DecisionV2


def test_snap_words():
    assert "temperature" in snap_to_automotive_domain("tempretur thoda badhao")
    assert "cabin" in snap_to_automotive_domain("kaibin thoda tanda karo")
    assert "blower" in snap_to_automotive_domain("bloer speed 3 pe karo")
    assert "volume" in snap_to_automotive_domain("volin 15 pe karo")


def test_domain_snapping_in_normalizer():
    # Verify distorted ASR inputs normalize to canonical automotive vocabulary
    assert "temperature" in normalize_speech_transcript("tempratur thoda badhao")
    assert "temperature" in normalize_speech_transcript("tempreture 22 pe karo")
    assert "volume" in normalize_speech_transcript("volin kam karo")
    assert "blower" in normalize_speech_transcript("bloer tez karo")
    # Verify valid non-domain words and case are preserved
    assert normalize_speech_transcript("Volume up") == "Volume up"
    assert normalize_speech_transcript("gaana band karo") == "gaana band karo"


def test_domain_ambiguity_clarification():
    pipe = build_v2_pipeline(with_asr=False)

    # 1. Temperature: both exact and distorted
    r1 = pipe.process_text("temperature")
    assert r1.decision == DecisionV2.CLARIFY
    assert "Did you want to set the temperature, or make the cabin warmer or cooler?" in r1.response_text

    r1_distorted = pipe.process_text("tempratur")
    assert r1_distorted.decision == DecisionV2.CLARIFY
    assert "Did you want to set the temperature, or make the cabin warmer or cooler?" in r1_distorted.response_text

    # 2. Fan / Blower:
    r2 = pipe.process_text("fan")
    assert r2.decision == DecisionV2.CLARIFY
    assert "Did you want to increase or decrease the fan speed?" in r2.response_text

    r2_distorted = pipe.process_text("bloer")
    assert r2_distorted.decision == DecisionV2.CLARIFY
    assert "Did you want to increase or decrease the fan speed?" in r2_distorted.response_text

    # 3. Volume:
    r3 = pipe.process_text("volume")
    assert r3.decision == DecisionV2.CLARIFY
    assert "Did you want to turn the volume up or down?" in r3.response_text

    r3_distorted = pipe.process_text("volin")
    assert r3_distorted.decision == DecisionV2.CLARIFY
    assert "Did you want to turn the volume up or down?" in r3_distorted.response_text

    # 4. Music:
    r4 = pipe.process_text("music")
    assert r4.decision == DecisionV2.CLARIFY
    assert "Did you want to play, pause, or change the song?" in r4.response_text

    # 5. AC:
    r5 = pipe.process_text("ac")
    assert r5.decision == DecisionV2.CLARIFY
    assert "Did you want to turn the AC on or off, or change the temperature?" in r5.response_text

