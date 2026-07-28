from common.constants import Direction
from common.constants import PhaseAction as PhaseActionEnum
from common.schemas.control import PhaseAction, PhaseState, ReasoningLog


def test_phase_action_accepts_canonical_phase() -> None:
    action = PhaseAction(target_phase=PhaseActionEnum.EAST_WEST, timestamp_s=1.0)

    assert action.target_phase == PhaseActionEnum.EAST_WEST
    assert action.target_directions == (Direction.EAST, Direction.WEST)


def test_phase_action_accepts_legacy_direction() -> None:
    action = PhaseAction(target_direction="NORTH", timestamp_s=1.0)

    assert action.target_phase == PhaseActionEnum.NORTH_SOUTH
    assert action.target_directions == (Direction.NORTH, Direction.SOUTH)


def test_phase_state_carries_two_way_phase_and_legacy_first_direction() -> None:
    state = PhaseState(
        active_phase=PhaseActionEnum.NORTH_SOUTH,
        active_directions=[Direction.NORTH, Direction.SOUTH],
        active_direction=Direction.NORTH,
        timestamp_s=1.0,
    )

    assert state.active_phase == PhaseActionEnum.NORTH_SOUTH
    assert state.active_directions == [Direction.NORTH, Direction.SOUTH]
    assert state.active_direction == Direction.NORTH


def test_reasoning_log_uses_phase_q_values() -> None:
    log = ReasoningLog(
        timestamp_s=1.0,
        q_values={"EAST_WEST": 1.0, "NORTH_SOUTH": 2.0},
        chosen_action=PhaseActionEnum.NORTH_SOUTH,
    )

    assert log.chosen_action == PhaseActionEnum.NORTH_SOUTH
