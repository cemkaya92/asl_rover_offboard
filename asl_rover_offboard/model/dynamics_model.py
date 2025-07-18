#!/usr/bin/env python3

import numpy as np
from casadi import SX, vertcat, horzcat, Function, diag, sin, cos, tan


# ==================== CasADi MPC Model ====================
def build_casadi_model(params):
    """
    Builds the CasADi symbolic model of the rover.
    The model now has 6 states and 2 control inputs (Wheel Speeds).
    """
    MASS = params.mass
    ARM_LENGTH = params.arm_length
    IX, IY, IZ = params.inertia
    GRAV = params.gravity

    # === States: 6 states ===
    # North - East - Down Coordinate Frame Convention is Used
    # pos: x, y
    # vel: vx, vy
    # rpy: yaw
    # omega: r (body-frame angular velocities)
    x_pos = SX.sym('x_pos', 2)
    x_vel = SX.sym('x_vel', 2)
    x_rpy = SX.sym('x_rpy', 1)  
    x_omega = SX.sym('x_omega', 1)
    x = vertcat(x_pos, x_vel, x_rpy, x_omega)

    # === Controls: 2 inputs ===
    # wl: Left wheel speed
    # wr: Right wheel speed 
    u = SX.sym('u', 2)

    # Extract states for clarity
    yaw = x_rpy
    r = x_omega

    # === Dynamics Equations ===
    # 1. Rotation matrix from body to world frame
    cψ, sψ = cos(yaw), sin(yaw)
    R = vertcat(
        horzcat(cψ, - sψ, 0),
        horzcat(sψ,   cψ, 0),
        horzcat( 0,    0, 1)
    )


    # 2. Translational Dynamics (in world frame)
    # Acceleration = gravity + rotated body thrust
    f_thrust = vertcat(0, 0, -u[0]) # Total thrust is along body z-axis
    accel = vertcat(0, 0, GRAV) + (R @ f_thrust) / MASS
    pos_dot = x_vel
    vel_dot = accel

    # 3. Rotational Dynamics (in body frame)
    # Euler's equations of motion
    omega_dot = vertcat(
        (u[1] - (IZ - IY) * q * r) / IX,
        (u[2] - (IX - IZ) * p * r) / IY,
        (u[3] - (IY - IX) * p * q) / IZ
    )

    # 4. Attitude Kinematics
    # Transformation from body rates (p,q,r) to Euler angle rates (d(rpy)/dt)
    # Using tan(pitch) can lead to singularity at +/- 90 degrees
    W_rpy = vertcat(
        horzcat(1, sφ*tan(pitch), cφ*tan(pitch)),
        horzcat(0, cφ, -sφ),
        horzcat(0, sφ/cθ, cφ/cθ)
    )
    rpy_dot = W_rpy @ x_omega

    # Full state derivative vector
    xdot = vertcat(pos_dot, vel_dot, rpy_dot, omega_dot)

    # CasADi function
    f = Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])
    return f, x.size1(), u.size1()

