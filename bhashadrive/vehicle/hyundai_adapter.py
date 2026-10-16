"""Integration boundary for a Hyundai infotainment platform.

THIS IS NOT CONNECTED TO ANY VEHICLE. No Hyundai API is known to or assumed by
this project. Each method documents the manufacturer-authorised service that
Hyundai would need to provide. Every method raises NotImplementedError until a
real, authorised interface (with its own access control) is supplied.

Design rules for a real implementation:
- Call only OEM-authorised infotainment/body services (e.g. a climate HMI
  service, media session service, navigation search service) through the
  platform's IPC (for example SOME/IP, Android Automotive VHAL/CarPropertyManager,
  or an OEM middleware), never raw CAN frames.
- Respect the platform's permission model; the voice app gets the minimum
  permissions for the whitelisted functions only.
- Never expose steering, braking, propulsion or other safety-critical functions.
- Read back state after each request and report the actual outcome.
"""
from __future__ import annotations

from ..schemas import ActionResult
from .base import VehicleAdapter

_REQUIRED = {
    "adjust_temperature": "Climate service: relative set-point change per zone, with min/max limits and current set-point read-back.",
    "set_temperature": "Climate service: absolute set-point (°C) per zone, validated by the HVAC controller.",
    "set_ac_power": "Climate service: AC/compressor on-off request, subject to HVAC controller arbitration.",
    "adjust_fan": "Climate service: blower level step up/down with level read-back.",
    "adjust_volume": "Audio management service: media-stream volume step, respecting the audio-focus/arbitration policy.",
    "media_play": "Media session service: play on the active source.",
    "media_pause": "Media session service: pause on the active source.",
    "media_next": "Media session service: next track.",
    "media_previous": "Media session service: previous track.",
    "request_navigation_search": "Navigation service: POI/address search request; route guidance remains owned by the navigation app.",
    "get_state": "State read-back for all of the above (set-point, blower level, volume, media state).",
}


class HyundaiVehicleAdapter(VehicleAdapter):
    name = "hyundai-boundary"

    @staticmethod
    def required_interfaces() -> dict[str, str]:
        return dict(_REQUIRED)

    def _missing(self, method: str):
        raise NotImplementedError(
            f"HyundaiVehicleAdapter.{method}: requires an OEM-authorised interface — {_REQUIRED[method]} "
            "No such interface is available to this prototype."
        )

    def adjust_temperature(self, direction, step_c) -> ActionResult: self._missing("adjust_temperature")
    def set_temperature(self, celsius) -> ActionResult: self._missing("set_temperature")
    def set_ac_power(self, on) -> ActionResult: self._missing("set_ac_power")
    def adjust_fan(self, direction, step) -> ActionResult: self._missing("adjust_fan")
    def adjust_volume(self, direction, step) -> ActionResult: self._missing("adjust_volume")
    def media_play(self) -> ActionResult: self._missing("media_play")
    def media_pause(self) -> ActionResult: self._missing("media_pause")
    def media_next(self) -> ActionResult: self._missing("media_next")
    def media_previous(self) -> ActionResult: self._missing("media_previous")
    def request_navigation_search(self, destination) -> ActionResult: self._missing("request_navigation_search")
    def get_state(self) -> dict: self._missing("get_state")
