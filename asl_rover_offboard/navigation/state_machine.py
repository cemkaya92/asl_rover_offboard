# hydro_mpc/navigation/state_machine.py
from __future__ import annotations
from enum import IntEnum
from dataclasses import dataclass

class NavState(IntEnum):
    UNKNOWN        = 0
    IDLE           = 1
    MISSION        = 2
    AVOIDANCE      = 3
    MANUAL         = 4


@dataclass
class NavEvents:
    have_odom: bool
    auto_start: bool
    trajectory_fresh: bool
    start_requested: bool
    mission_valid: bool
    at_destination: bool
    halt_condition: bool
    manual_requested: bool = False
    offboard_ok: bool = False
    in_avoidance: bool = False

class NavStateMachine:
    """Pure transition logic; no ROS, no planning side effects."""
    def __init__(self):
        self.state = NavState.IDLE

    def reset(self, state: NavState = NavState.IDLE):
        self.state = state

    def step(self, ev: NavEvents) -> NavState:
        
        # Global halt: go to HOLD (air) unless EMERGENCY forces landing
        if ev.halt_condition:
            self.state = NavState.IDLE
            return self.state
        
        if ev.manual_requested:
            self.state = NavState.MANUAL
            return self.state
        
        s = self.state
        if s == NavState.IDLE:
            if ev.have_odom and ev.trajectory_fresh and ev.offboard_ok and (ev.auto_start or ev.start_requested) and ev.mission_valid:
                self.state = NavState.MISSION

        elif s == NavState.MISSION:
            if ev.at_destination or (not ev.trajectory_fresh) or (not ev.mission_valid):
                self.state = NavState.IDLE 
            elif (not ev.offboard_ok) or (ev.manual_requested):
                self.state = NavState.IDLE
            # elif ev.in_avoidance:
            #     self.state = NavState.AVOIDANCE

        # elif s == NavState.AVOIDANCE:
        #     if (not ev.in_avoidance) and ev.trajectory_fresh and ev.mission_valid:
        #         self.state = NavState.MISSION

        elif s == NavState.MANUAL:
            # We only reach here when manual_requested just turned False
            if ev.offboard_ok:
                self.state = NavState.IDLE

        return self.state
