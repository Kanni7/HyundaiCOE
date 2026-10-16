"""Deterministic parameter validation against configurable vehicle limits."""
from __future__ import annotations

import re

from ..config import VehicleLimits
from ..schemas import Command, Intent

# Destination text is passed to a navigation search, never executed; still strip control/markup chars.
_UNSAFE_DEST = re.compile(r"[<>{}`$;|\\\x00-\x1f]")


class ValidationIssue(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def validate_command(cmd: Command, limits: VehicleLimits) -> Command:
    """Return the command if valid; raise ValidationIssue otherwise."""
    p = cmd.parameters
    match cmd.intent:
        case Intent.CLIMATE_ADJUST_TEMPERATURE | Intent.FAN_ADJUST_SPEED | Intent.VOLUME_ADJUST:
            if p.direction is None:
                raise ValidationIssue("missing_parameter", "direction is required")
        case Intent.CLIMATE_SET_TEMPERATURE:
            t = p.target_temperature_celsius
            if t is None:
                raise ValidationIssue("missing_parameter", "target temperature is required")
            if not (limits.temp_min_c <= t <= limits.temp_max_c):
                raise ValidationIssue(
                    "invalid_parameter",
                    f"{t:g}°C is outside the allowed range {limits.temp_min_c:g}-{limits.temp_max_c:g}°C",
                )
        case Intent.CLIMATE_AC_POWER:
            if p.power is None:
                raise ValidationIssue("missing_parameter", "on/off is required")
        case Intent.NAVIGATION_SEARCH:
            d = p.destination
            if not d:
                raise ValidationIssue("missing_parameter", "destination is required")
            if len(d) > limits.max_destination_chars or _UNSAFE_DEST.search(d):
                raise ValidationIssue("invalid_parameter", "destination text is not acceptable")
        case Intent.MEDIA_PLAY | Intent.MEDIA_PAUSE | Intent.MEDIA_NEXT_TRACK | Intent.MEDIA_PREVIOUS_TRACK:
            pass
        case _:
            raise ValidationIssue("unsupported_action", f"{cmd.intent.value} is not an executable action")
    return cmd

