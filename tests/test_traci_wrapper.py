"""Unit + integration tests for simulation/traci_wrapper.

The integration tests need a real SUMO install and a built network
(simulation/net/intersection.net.xml + .rou.xml). They're skipped, not
failed, when those aren't present, so `pytest` stays green on machines
(including CI) that haven't installed SUMO yet - matching Stage 1's
"skeleton first" goal.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from simulation.traci_wrapper import SumoNotFoundError, TraciSession, find_sumo_binary
from tests.conftest import SUMOCFG, requires_network, requires_sumo


class TestTraciSessionUnit:
    """Tests that don't need SUMO at all - just the wrapper's own logic."""

    def test_missing_sumocfg_raises_before_touching_sumo(self, tmp_path: Path) -> None:
        session = TraciSession(sumocfg_path=tmp_path / "does_not_exist.sumocfg")
        with pytest.raises(FileNotFoundError):
            session._build_command()

    def test_step_before_start_raises(self) -> None:
        session = TraciSession(sumocfg_path=SUMOCFG)
        with pytest.raises(RuntimeError):
            session.step()

    def test_double_start_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        session = TraciSession(sumocfg_path=SUMOCFG)
        session._connected = True
        with pytest.raises(RuntimeError):
            session.start()

    def test_close_before_start_is_a_safe_noop(self) -> None:
        session = TraciSession(sumocfg_path=SUMOCFG)
        session.close()  # must not raise
        assert not session.is_connected

    def test_find_sumo_binary_raises_clear_error_when_nothing_configured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda name: None)
        monkeypatch.delenv("SUMO_HOME", raising=False)
        with pytest.raises(SumoNotFoundError):
            find_sumo_binary(use_gui=False)


@requires_sumo
@requires_network
class TestTraciSessionIntegration:
    """End-to-end smoke tests against a real SUMO process."""

    def test_context_manager_steps_and_closes_cleanly(self) -> None:
        with TraciSession(sumocfg_path=SUMOCFG, seed=1) as sim:
            assert sim.is_connected
            new_count = sim.step(10)
            assert new_count == 10
        assert not sim.is_connected

    def test_vehicles_spawn_from_configured_demand(self) -> None:
        with TraciSession(sumocfg_path=SUMOCFG, seed=1) as sim:
            seen_any_vehicle = False
            for _ in range(300):
                sim.step()
                if sim.traci.vehicle.getIDList():
                    seen_any_vehicle = True
                    break
            assert seen_any_vehicle, "No vehicles appeared in 300 steps - check demand config."

    def test_traffic_light_program_is_queryable(self) -> None:
        with TraciSession(sumocfg_path=SUMOCFG, seed=1) as sim:
            state = sim.traci.trafficlight.getRedYellowGreenState("C")
            assert isinstance(state, str)
            assert len(state) > 0
            assert set(state.upper()) <= set("RYG")

    def test_exception_inside_context_still_closes_session(self) -> None:
        session = TraciSession(sumocfg_path=SUMOCFG, seed=1)
        with pytest.raises(ValueError):
            with session:
                session.step()
                raise ValueError("boom")
        assert not session.is_connected
