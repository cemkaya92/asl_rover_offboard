#!/usr/bin/env python3

import numpy as np
from casadi import SX, vertcat, horzcat, Function, diag, sin, cos, tan


# ==================== CasADi MPC Model ====================
def build_casadi_model(params):
    """
    Builds the CasADi symbolic model of the rover.
    The model now has 3 states and 2 control inputs (Wheel Speeds).
    """
    HALF_WIDTH = 0.5*params.base_width # 2L is the total width of the vehicle base
    WHEEL_RADIUS = params.wheel_radius

    # === States: 3 states ===
    # North - East - Down Coordinate Frame Convention is Used
    # pose: x, y, yaw
    x_pose = SX.sym('x_pose', 3)
    x = vertcat(x_pose)

    # === Controls: 2 inputs ===
    # wl: Left wheel speed
    # wr: Right wheel speed 
    u = SX.sym('u', 2)

    # Extract states for clarity
    yaw = x_pose[2]

    # === Dynamics Equations ===
    # 1. Rotation matrix from body to world frame
    cψ, sψ = cos(yaw), sin(yaw)
    R = vertcat(
        horzcat(cψ, - sψ, 0),
        horzcat(sψ,   cψ, 0),
        horzcat( 0,    0, 1)
    )


    # 2. Translational Kinematics (in world frame)
    x_vel = vertcat(0.5*WHEEL_RADIUS*(u[0]+u[1])*cψ,
                    0.5*WHEEL_RADIUS*(u[0]+u[1])*sψ,
                    0.5*WHEEL_RADIUS*(u[0]-u[1])/HALF_WIDTH)

    pose_dot = x_vel


    # Full state derivative vector
    xdot = vertcat(pose_dot)

    # CasADi function
    f = Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f, x.size1(), u.size1()

