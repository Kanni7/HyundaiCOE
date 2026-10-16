"""Versioned V2 intent taxonomy and per-intent parameter schemas.

The classifier predicts one of `IntentV2`. Only intents listed in `PARAM_SCHEMA` map to a
vehicle action; every action is validated against its schema and the configured limits
before it can reach the vehicle adapter.
"""
from __future__ import annotations

from enum import Enum

TAXONOMY_VERSION = "2.0.0"


class IntentV2(str, Enum):
    CLIMATE_TEMPERATURE_UP = "climate_temperature_up"
    CLIMATE_TEMPERATURE_DOWN = "climate_temperature_down"
    CLIMATE_TEMPERATURE_SET = "climate_temperature_set"
    CLIMATE_FAN_SPEED_UP = "climate_fan_speed_up"
    CLIMATE_FAN_SPEED_DOWN = "climate_fan_speed_down"
    CLIMATE_ON = "climate_on"
    CLIMATE_OFF = "climate_off"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    MEDIA_PLAY = "media_play"
    MEDIA_PAUSE = "media_pause"
    MEDIA_NEXT = "media_next"
    MEDIA_PREVIOUS = "media_previous"
    NAVIGATION_DESTINATION_REQUEST = "navigation_destination_request"
    DIALOG_CONFIRM = "dialog_confirm"  # "yes", "haan kar do" — only meaningful with a pending proposal
    DIALOG_CANCEL = "dialog_cancel"  # "no", "rehne do", "cancel"
    UNSUPPORTED = "unsupported"  # out of domain or not permitted (brakes, windows, chit-chat)
    AMBIGUOUS = "ambiguous"  # in-domain but the target/action cannot be determined ("change it")


# Allowed parameters per executable intent: name -> required?
PARAM_SCHEMA: dict[IntentV2, dict[str, bool]] = {
    IntentV2.CLIMATE_TEMPERATURE_UP: {"direction": True, "amount": False},
    IntentV2.CLIMATE_TEMPERATURE_DOWN: {"direction": True, "amount": False},
    IntentV2.CLIMATE_TEMPERATURE_SET: {"target_temperature_celsius": True},
    IntentV2.CLIMATE_FAN_SPEED_UP: {"amount": False},
    IntentV2.CLIMATE_FAN_SPEED_DOWN: {"amount": False},
    IntentV2.CLIMATE_ON: {},
    IntentV2.CLIMATE_OFF: {},
    IntentV2.VOLUME_UP: {"amount": False},
    IntentV2.VOLUME_DOWN: {"amount": False},
    IntentV2.MEDIA_PLAY: {},
    IntentV2.MEDIA_PAUSE: {},
    IntentV2.MEDIA_NEXT: {},
    IntentV2.MEDIA_PREVIOUS: {},
    IntentV2.NAVIGATION_DESTINATION_REQUEST: {"destination": True},
}
EXECUTABLE_INTENTS = frozenset(PARAM_SCHEMA)
DIALOG_INTENTS = frozenset({IntentV2.DIALOG_CONFIRM, IntentV2.DIALOG_CANCEL})
NON_ACTION_INTENTS = frozenset({IntentV2.UNSUPPORTED, IntentV2.AMBIGUOUS}) | DIALOG_INTENTS
# Temperature direction is fixed by the intent, so it is filled in rather than extracted.
FIXED_DIRECTION = {IntentV2.CLIMATE_TEMPERATURE_UP: "warmer", IntentV2.CLIMATE_TEMPERATURE_DOWN: "cooler"}

DESCRIPTIONS = {
    IntentV2.CLIMATE_TEMPERATURE_UP: "make the cabin warmer",
    IntentV2.CLIMATE_TEMPERATURE_DOWN: "make the cabin cooler",
    IntentV2.CLIMATE_TEMPERATURE_SET: "set the cabin temperature",
    IntentV2.CLIMATE_FAN_SPEED_UP: "increase the fan speed",
    IntentV2.CLIMATE_FAN_SPEED_DOWN: "decrease the fan speed",
    IntentV2.CLIMATE_ON: "turn the climate/AC on",
    IntentV2.CLIMATE_OFF: "turn the climate/AC off",
    IntentV2.VOLUME_UP: "turn the volume up",
    IntentV2.VOLUME_DOWN: "turn the volume down",
    IntentV2.MEDIA_PLAY: "play media",
    IntentV2.MEDIA_PAUSE: "pause media",
    IntentV2.MEDIA_NEXT: "skip to the next track",
    IntentV2.MEDIA_PREVIOUS: "go back to the previous track",
    IntentV2.NAVIGATION_DESTINATION_REQUEST: "search navigation for a destination",
}
