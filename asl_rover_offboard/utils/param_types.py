# param_types.py

from dataclasses import dataclass
from typing import Tuple, List

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
    Qf_factor: float
