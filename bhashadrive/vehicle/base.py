"""Vehicle integration boundary. The intent engine only talks to this interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import ActionResult, Command, Direction, Intent


class VehicleAdapter(ABC):
    """One method per whitelisted action. Implementations return structured results."""

    name = "abstract"

    @abstractmethod
    def adjust_temperature(self, direction: Direction, step_c: float) -> ActionResult: ...
    @abstractmethod
    def set_temperature(self, celsius: float) -> ActionResult: ...
    @abstractmethod
    def set_ac_power(self, on: bool) -> ActionResult: ...
    @abstractmethod
    def adjust_fan(self, direction: Direction, step: int) -> ActionResult: ...
    @abstractmethod
    def adjust_volume(self, direction: Direction, step: int) -> ActionResult: ...
    @abstractmethod
    def media_play(self) -> ActionResult: ...
    @abstractmethod
    def media_pause(self) -> ActionResult: ...
    @abstractmethod
    def media_next(self) -> ActionResult: ...
    @abstractmethod
    def media_previous(self) -> ActionResult: ...
    @abstractmethod
    def request_navigation_search(self, destination: str) -> ActionResult: ...
    @abstractmethod
    def get_state(self) -> dict: ...


def dispatch(adapter: VehicleAdapter, cmd: Command, limits) -> ActionResult:
    """Map a validated Command onto the explicit adapter allowlist. No dynamic dispatch by name."""
    p = cmd.parameters
    amount = p.amount.value if p.amount else "small"
    match cmd.intent:
        case Intent.CLIMATE_ADJUST_TEMPERATURE:
            return adapter.adjust_temperature(p.direction, limits.temp_steps[amount])
        case Intent.CLIMATE_SET_TEMPERATURE:
            return adapter.set_temperature(p.target_temperature_celsius)
        case Intent.CLIMATE_AC_POWER:
            return adapter.set_ac_power(p.power == "on")
        case Intent.FAN_ADJUST_SPEED:
            return adapter.adjust_fan(p.direction, limits.fan_steps[amount])
        case Intent.VOLUME_ADJUST:
            return adapter.adjust_volume(p.direction, limits.volume_steps[amount])
        case Intent.MEDIA_PLAY:
            return adapter.media_play()
        case Intent.MEDIA_PAUSE:
            return adapter.media_pause()
        case Intent.MEDIA_NEXT_TRACK:
            return adapter.media_next()
        case Intent.MEDIA_PREVIOUS_TRACK:
            return adapter.media_previous()
        case Intent.NAVIGATION_SEARCH:
            return adapter.request_navigation_search(p.destination)
    return ActionResult(success=False, intent=cmd.intent, message="Action not in allowlist")


def dispatch_v2(adapter: VehicleAdapter, action, limits) -> ActionResult:
    """Map a validated V2 `VehicleAction` onto the same explicit adapter allowlist."""
    from ..nlu.taxonomy import IntentV2 as I

    p = action.parameters
    amount = p.amount or "small"
    match action.intent:
        case I.CLIMATE_TEMPERATURE_UP:
            return adapter.adjust_temperature(Direction.INCREASE, limits.temp_steps[amount])
        case I.CLIMATE_TEMPERATURE_DOWN:
            return adapter.adjust_temperature(Direction.DECREASE, limits.temp_steps[amount])
        case I.CLIMATE_TEMPERATURE_SET:
            return adapter.set_temperature(p.target_temperature_celsius)
        case I.CLIMATE_FAN_SPEED_UP:
            return adapter.adjust_fan(Direction.INCREASE, limits.fan_steps[amount])
        case I.CLIMATE_FAN_SPEED_DOWN:
            return adapter.adjust_fan(Direction.DECREASE, limits.fan_steps[amount])
        case I.CLIMATE_ON:
            return adapter.set_ac_power(True)
        case I.CLIMATE_OFF:
            return adapter.set_ac_power(False)
        case I.VOLUME_UP:
            return adapter.adjust_volume(Direction.INCREASE, limits.volume_steps[amount])
        case I.VOLUME_DOWN:
            return adapter.adjust_volume(Direction.DECREASE, limits.volume_steps[amount])
        case I.MEDIA_PLAY:
            return adapter.media_play()
        case I.MEDIA_PAUSE:
            return adapter.media_pause()
        case I.MEDIA_NEXT:
            return adapter.media_next()
        case I.MEDIA_PREVIOUS:
            return adapter.media_previous()
        case I.NAVIGATION_DESTINATION_REQUEST:
            return adapter.request_navigation_search(p.destination)
    return ActionResult(success=False, intent=Intent.UNSUPPORTED, message="Action not in allowlist")
