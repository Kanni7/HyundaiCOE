import pytest

from bhashadrive.config import VehicleLimits
from bhashadrive.schemas import Command, Direction
from bhashadrive.vehicle.base import dispatch
from bhashadrive.vehicle.hyundai_adapter import HyundaiVehicleAdapter
from bhashadrive.vehicle.mock_adapter import MockVehicleAdapter

L = VehicleLimits()


def test_temperature_steps_and_clamps():
    v = MockVehicleAdapter(L)
    assert dispatch(v, Command(intent="climate_adjust_temperature", parameters={"direction": "increase", "amount": "large"}), L).success
    assert v.state["cabin_temperature_c"] == 25.0
    for _ in range(10):
        dispatch(v, Command(intent="climate_adjust_temperature", parameters={"direction": "increase"}), L)
    assert v.state["cabin_temperature_c"] == L.temp_max_c
    r = dispatch(v, Command(intent="climate_adjust_temperature", parameters={"direction": "increase"}), L)
    assert not r.success  # already at limit


def test_volume_fan_media_nav():
    v = MockVehicleAdapter(L)
    dispatch(v, Command(intent="volume_adjust", parameters={"direction": "decrease", "amount": "small"}), L)
    assert v.state["volume"] == 8
    dispatch(v, Command(intent="fan_adjust_speed", parameters={"direction": "decrease"}), L)
    assert v.state["fan_speed"] == 2
    dispatch(v, Command(intent="media_play"), L)
    assert v.state["media_playing"]
    dispatch(v, Command(intent="media_pause"), L)
    assert not v.state["media_playing"]
    dispatch(v, Command(intent="media_next_track"), L)
    dispatch(v, Command(intent="media_previous_track"), L)
    assert v.state["track_index"] == 0
    r = dispatch(v, Command(intent="navigation_search", parameters={"destination": "India Gate"}), L)
    assert r.success and "simulated" in r.message and v.state["navigation_request"]["query"] == "India Gate"


def test_set_temperature_out_of_range_fails_in_adapter_too():
    assert not MockVehicleAdapter(L).set_temperature(50).success


def test_non_actionable_intent_not_dispatched():
    r = dispatch(MockVehicleAdapter(L), Command(intent="unsupported"), L)
    assert not r.success


def test_hyundai_adapter_is_an_unimplemented_boundary():
    h = HyundaiVehicleAdapter()
    with pytest.raises(NotImplementedError, match="OEM-authorised"):
        h.adjust_volume(Direction.INCREASE, 2)
    assert set(h.required_interfaces()) >= {"adjust_temperature", "request_navigation_search", "get_state"}
