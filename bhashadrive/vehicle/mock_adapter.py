"""In-memory vehicle simulator for development and tests."""
from __future__ import annotations

from ..config import VehicleLimits
from ..schemas import ActionResult, Direction, Intent
from .base import VehicleAdapter


class MockVehicleAdapter(VehicleAdapter):
    name = "mock"

    def __init__(self, limits: VehicleLimits | None = None):
        self.limits = limits or VehicleLimits()
        self.reset()

    def reset(self) -> None:
        self.state = {
            "cabin_temperature_c": 22.0,
            "ac_on": True,
            "fan_speed": 3,
            "volume": 10,
            "media_playing": False,
            "track_index": 0,
            "navigation_request": None,
        }
        self.log: list[dict] = []

    def _ok(self, intent: Intent, msg: str) -> ActionResult:
        self.log.append({"intent": intent.value, "message": msg})
        return ActionResult(success=True, intent=intent, message=msg, state=dict(self.state))

    def _fail(self, intent: Intent, msg: str) -> ActionResult:
        return ActionResult(success=False, intent=intent, message=msg, state=dict(self.state))

    @staticmethod
    def _sign(direction: Direction) -> int:
        return 1 if direction == Direction.INCREASE else -1

    def adjust_temperature(self, direction, step_c):
        lim, cur = self.limits, self.state["cabin_temperature_c"]
        new = min(lim.temp_max_c, max(lim.temp_min_c, cur + self._sign(direction) * step_c))
        if new == cur:
            return self._fail(Intent.CLIMATE_ADJUST_TEMPERATURE, f"Temperature already at limit ({cur:g}°C)")
        self.state["cabin_temperature_c"] = new
        return self._ok(Intent.CLIMATE_ADJUST_TEMPERATURE, f"Cabin temperature {cur:g}°C -> {new:g}°C")

    def set_temperature(self, celsius):
        lim = self.limits
        if not (lim.temp_min_c <= celsius <= lim.temp_max_c):
            return self._fail(Intent.CLIMATE_SET_TEMPERATURE, "Temperature out of range")
        self.state["cabin_temperature_c"] = float(celsius)
        return self._ok(Intent.CLIMATE_SET_TEMPERATURE, f"Cabin temperature set to {celsius:g}°C")

    def set_ac_power(self, on):
        self.state["ac_on"] = bool(on)
        return self._ok(Intent.CLIMATE_AC_POWER, f"AC turned {'on' if on else 'off'}")

    def adjust_fan(self, direction, step):
        lim, cur = self.limits, self.state["fan_speed"]
        new = min(lim.fan_max, max(lim.fan_min, cur + self._sign(direction) * step))
        if new == cur:
            return self._fail(Intent.FAN_ADJUST_SPEED, f"Fan speed already at limit ({cur})")
        self.state["fan_speed"] = new
        return self._ok(Intent.FAN_ADJUST_SPEED, f"Fan speed {cur} -> {new}")

    def adjust_volume(self, direction, step):
        lim, cur = self.limits, self.state["volume"]
        new = min(lim.volume_max, max(lim.volume_min, cur + self._sign(direction) * step))
        if new == cur:
            return self._fail(Intent.VOLUME_ADJUST, f"Volume already at limit ({cur})")
        self.state["volume"] = new
        return self._ok(Intent.VOLUME_ADJUST, f"Volume {cur} -> {new}")

    def media_play(self):
        self.state["media_playing"] = True
        return self._ok(Intent.MEDIA_PLAY, "Media playing")

    def media_pause(self):
        self.state["media_playing"] = False
        return self._ok(Intent.MEDIA_PAUSE, "Media paused")

    def media_next(self):
        self.state["track_index"] += 1
        return self._ok(Intent.MEDIA_NEXT_TRACK, f"Skipped to track {self.state['track_index']}")

    def media_previous(self):
        self.state["track_index"] = max(0, self.state["track_index"] - 1)
        return self._ok(Intent.MEDIA_PREVIOUS_TRACK, f"Back to track {self.state['track_index']}")

    def request_navigation_search(self, destination):
        # Simulator only records the request. No route is calculated.
        self.state["navigation_request"] = {"query": destination, "status": "search_requested_simulated"}
        return self._ok(Intent.NAVIGATION_SEARCH, f"Navigation search request queued for '{destination}' (simulated, no route computed)")

    def get_state(self):
        return dict(self.state)
