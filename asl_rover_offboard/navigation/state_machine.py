# hydro_mpc/navigation/state_machine.py
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass

class NavState(Enum):
    IDLE = auto()
    MISSION = auto()


@dataclass
class NavEvents:
    have_odom: bool
    auto_start: bool
    trajectory_fresh: bool
    at_destination: bool
    halt_condition: bool

class NavStateMachine:
    """Pure transition logic; no ROS, no planning side effects."""
    def __init__(self):
        self.state = NavState.IDLE

    def reset(self, state: NavState = NavState.IDLE):
        self.state = state

    def step(self, ev: NavEvents) -> NavState:
        s = self.state
        if s == NavState.IDLE:
            if ev.have_odom and ev.auto_start and ev.trajectory_fresh:
                self.state = NavState.MISSION

        elif s == NavState.MISSION:
            if ev.at_destination or ev.halt_condition:
                self.state = NavState.IDLE 

        return self.state
