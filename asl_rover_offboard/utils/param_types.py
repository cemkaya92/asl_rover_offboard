# param_types.py

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union, List

Repeat = Literal["none", "loop", "pingpong"]
MissionType = Literal[
    "line_to", "straight", "arc", "rounded_rectangle", "racetrack_capsule"
]

@dataclass
class VehicleParams:
    mass: float                         # Kg
    base_width: float                   # m
    wheel_radius: float                 # m
    inertia: Tuple[float, float, float] # (Ix, Iy, Iz) Kg.m^2
    max_linear_speed: float             # m/s
    max_angular_speed: float            # deg/s
    omega_to_pwm_coefficient: Tuple[float, float, float] # (x_2, x_1, x_0)
    PWM_MIN: float
    PWM_MAX: float
    input_scaling: float
    zero_position_armed: float

@dataclass
class ControlParams:
    frequency: float
    N: int
    v_max: float
    w_max: float
    Q: List[float]  # length should be NX (e.g., 12)
    R: List[float]  # length should be NU (e.g., 4)
    R_delta: List[float]  # length should be NU (e.g., 4)
    Qf_factor: float

# -------- MISSION RELATED DATA CLASSES ---------
# ---------------- Shared blocks ----------------
@dataclass
class StartPose:
    use_current: bool = True
    x: float = 0.0
    y: float = 0.0
    psi: float = 0.0  # radians

@dataclass
class Common:
    repeat: Repeat = "none"
    start: StartPose = StartPose()
    speed: Optional[float] = None  # m/s (may be unused by some types)

# ---------------- Variants ----------------
@dataclass
class LineTo:
    type: Literal["line_to"] = "line_to"
    common: Common = Common()
    goal_xypsi: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    duration: float = 0.0  # seconds

@dataclass
class Straight:
    type: Literal["straight"] = "straight"
    common: Common = Common()
    segment_distance: float = 0.0  # m, 0 => unbounded

@dataclass
class Arc:
    type: Literal["arc"] = "arc"
    common: Common = Common()
    radius: float = 1.0
    angle: Optional[float] = None   # radians; provide angle OR yaw_rate
    yaw_rate: Optional[float] = None # rad/s
    cw: bool = True

@dataclass
class RoundedRectangle:
    type: Literal["rounded_rectangle"] = "rounded_rectangle"
    common: Common = Common()
    width: float = 1.0
    height: float = 1.0
    corner_radius: float = 0.1
    cw: bool = True

@dataclass
class RacetrackCapsule:
    type: Literal["racetrack_capsule"] = "racetrack_capsule"
    common: Common = Common()
    straight_length: float = 1.0
    radius: float = 0.5
    cw: bool = True

Mission = Union[LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule]
