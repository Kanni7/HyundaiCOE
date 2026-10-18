"""Comprehensive battery test for spoken automotive in-car commands in Hindi, English, and Hinglish."""
import re
import pytest

from bhashadrive.tests.fakes_v2 import DEFAULT_RULES

TEST_CASES = [
    # Climate on
    ('ac chala do', 'climate_on'),
    ('ac chalu karo', 'climate_on'),
    ('ac on kar do', 'climate_on'),
    ('turn on the ac', 'climate_on'),
    ('air conditioner chalu karo', 'climate_on'),
    ('climate control on', 'climate_on'),
    ('thandi hawa chalao', 'climate_on'),
    
    # Climate off
    ('ac band kar do', 'climate_off'),
    ('ac off karo', 'climate_off'),
    ('turn off the ac', 'climate_off'),
    ('switch off climate', 'climate_off'),
    ('aircon band karo', 'climate_off'),

    # Temp up / warmer / cold complaints
    ('ac thoda warm kar do', 'climate_temperature_up'),
    ('make it warmer', 'climate_temperature_up'),
    ('thand lag rahi hai', 'climate_temperature_up'),
    ('bohot sardi hai', 'climate_temperature_up'),
    ('heater chala do', 'climate_temperature_up'),
    ('gadi garam karo', 'climate_temperature_up'),
    ('temperature badha do', 'climate_temperature_up'),

    # Temp down / cooler / heat complaints
    ('ac thanda karo', 'climate_temperature_down'),
    ('make it cooler', 'climate_temperature_down'),
    ('garmi lag rahi hai', 'climate_temperature_down'),
    ('gami se jaan ni kaal rahi hai', 'climate_temperature_down'),
    ('bohot garmi hai yaar', 'climate_temperature_down'),
    ('cooling badha do', 'climate_temperature_down'),
    ('temperature kam karo', 'climate_temperature_down'),

    # Temp set
    ('temperature 22 pe set karo', 'climate_temperature_set'),
    ('set temperature to 21 degrees', 'climate_temperature_set'),
    ('ac 20 pe kar do', 'climate_temperature_set'),

    # Fan up
    ('fan speed badha do', 'climate_fan_speed_up'),
    ('fan tez karo', 'climate_fan_speed_up'),
    ('blower tez karo', 'climate_fan_speed_up'),
    ('hawa badhao', 'climate_fan_speed_up'),
    ('pankha tez kar do', 'climate_fan_speed_up'),
    ('turn up the fan', 'climate_fan_speed_up'),

    # Fan down
    ('fan speed kam karo', 'climate_fan_speed_down'),
    ('fan dheema karo', 'climate_fan_speed_down'),
    ('fan slow karo', 'climate_fan_speed_down'),
    ('blower kam karo', 'climate_fan_speed_down'),
    ('hawa dheemi kar do', 'climate_fan_speed_down'),
    ('pankha dheema kar do', 'climate_fan_speed_down'),
    ('turn down the fan', 'climate_fan_speed_down'),

    # Volume up
    ('volume badha do', 'volume_up'),
    ('awaaz tez karo', 'volume_up'),
    ('sound increase karo', 'volume_up'),
    ('louder please', 'volume_up'),
    ('turn up the volume', 'volume_up'),

    # Volume down
    ('volume kam karo', 'volume_down'),
    ('awaaz dheemi karo', 'volume_down'),
    ('sound kam karo', 'volume_down'),
    ('quieter', 'volume_down'),
    ('mute karo', 'volume_down'),

    # Media play
    ('gaana chalao', 'media_play'),
    ('gaana lagao', 'media_play'),
    ('gana shuru karo', 'media_play'),
    ('music play karo', 'media_play'),
    ('play some music', 'media_play'),
    ('song chala do', 'media_play'),
    ('kuch bajao', 'media_play'),

    # Media pause
    ('gaana band karo', 'media_pause'),
    ('gaana roko', 'media_pause'),
    ('music pause karo', 'media_pause'),
    ('pause the song', 'media_pause'),
    ('chup ho jao', 'media_pause'),
    ('stop the music', 'media_pause'),

    # Media next / prev
    ('next song', 'media_next'),
    ('agla gaana', 'media_next'),
    ('skip this song', 'media_next'),
    ('previous song', 'media_previous'),
    ('pichla gaana', 'media_previous'),

    # Navigation
    ('navigate to Connaught Place', 'navigation_destination_request'),
    ('rasta dikhao', 'navigation_destination_request'),
    ('take me to airport', 'navigation_destination_request'),
    ('delhi le chalo', 'navigation_destination_request'),

    # Safety rejection
    ('open the sunroof', 'unsupported'),
    ('khidki kholo', 'unsupported'),
    ('order a pizza', 'unsupported'),
    ('what is the cricket score', 'unsupported'),

    # Real spoken session variations
    ('AC thoda tanda, kati koi.', 'climate_temperature_down'),
    ('Kaibin thoda warm kar do,', 'climate_temperature_up'),
    ('Kaibin thoda kanda karo, do.', 'climate_temperature_down'),
    ('Chut, fagrin, potte, garmi lag rahi hai.', 'climate_temperature_down'),
    ('Per milagriya faqtaj.', 'climate_temperature_down'),
    ('ndal lag rahi hai, bhootayal dha ma chodha.', 'climate_temperature_up'),
    ('volin badha do, volume badha do, volume kam karo,', 'volume_up'),
]


@pytest.mark.parametrize("text,expected", TEST_CASES)
def test_command_battery(text, expected):
    from bhashadrive.nlu.normalizer import normalize_speech_transcript, to_english
    norm = to_english(normalize_speech_transcript(text))
    matched = "unsupported"
    for pat_str, it, _ in DEFAULT_RULES:
        if re.search(pat_str, norm, re.I):
            matched = it
            break
    assert matched == expected, f"Failed for '{text}' (normalized '{norm}'): got '{matched}', expected '{expected}'"
