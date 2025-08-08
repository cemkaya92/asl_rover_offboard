#!/usr/bin/env python3

import numpy as np
import casadi as ca


# ==================== CasADi MPC Model ====================
def build_casadi_model(Ts: float):
    """
    State: x=[x, y, yaw]
    Input: u=[v, w]
    """
    xs = ca.SX.sym('x', 3)
    us = ca.SX.sym('u', 2)

    xdot = ca.vertcat(
        us[0]*ca.cos(xs[2]),
        us[0]*ca.sin(xs[2]),
        us[1]
    )
    f = ca.Function('f', [xs, us], [xdot])

    h = Ts
    k1 = f(xs, us)
    k2 = f(xs + h/2*k1, us)
    k3 = f(xs + h/2*k2, us)
    k4 = f(xs + h*k3,   us)
    xn = xs + h/6*(k1 + 2*k2 + 2*k3 + k4)

    F = ca.Function('F', [xs, us], [xn])  # discrete-time RK4 step
    return f, F


