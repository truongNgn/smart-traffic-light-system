"""Data contracts for intersection control and AI reasoning."""

from typing import Dict, Optional
from pydantic import BaseModel, Field
from common.constants import Direction

class PhaseAction(BaseModel):
    """Command sent by the AI Agent to the executor."""
    target_direction: Direction = Field(description="The direction that should get the green light.")
    timestamp_s: float = Field(description="Timestamp of the decision.")

class PhaseState(BaseModel):
    """The current physical state of the intersection."""
    active_direction: Optional[Direction] = Field(description="Which direction currently has right of way. None if all-red.")
    is_yellow: bool = Field(default=False, description="True if the active direction is currently showing yellow.")
    is_all_red: bool = Field(default=False, description="True if all lights are currently red (safety buffer).")
    timestamp_s: float = Field(description="Timestamp when this state became active.")

class ReasoningLog(BaseModel):
    """Detailed log of the AI's decision process for the dashboard."""
    timestamp_s: float = Field(description="Timestamp of the decision.")
    q_values: Dict[str, float] = Field(description="Q-values for each possible action (e.g., {'EAST': 12.5, ...}).")
    chosen_action: Direction = Field(description="The action selected by the policy.")
    exploration: bool = Field(default=False, description="True if the action was chosen randomly (epsilon-greedy).")
