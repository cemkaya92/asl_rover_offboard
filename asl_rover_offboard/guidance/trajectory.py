#!/usr/bin/env python3

import numpy as np

# ==================== Trajectory Generation ====================

def eval_traj(t,pos):
    """
    Generates an infinity 
    """
    # === Define Trajectory Parameters ===
    X_RADIUS = 5.0
    Y_RADIUS = 15.0  
    OMEGA = 10.0 * np.pi / 180.0     # angular speed rad/s

    # === Initialize the current position ===
    if not hasattr(eval_traj, "pos0"):
        eval_traj.pos0 = pos  # Store first t as offset
    

    return figure8_trajectory(t,X_RADIUS,Y_RADIUS,OMEGA)


def figure8_trajectory(t, a=1.0, b=1.0, omega=1.0):
    """
    Returns position and velocity for a figure-8 trajectory on the XY plane.
    
    Parameters:
        t (float or array): Time
        a (float): Amplitude in X direction
        b (float): Amplitude in Y direction
        omega (float): Angular frequency

    Returns:
        pos, vel: arrays of x and y positions and velocities
    """
    x = eval_traj.pos0[0] + a * np.sin(omega * t)
    y = eval_traj.pos0[1] + b * np.sin(2 * omega * t)

    x_dot = a * omega * np.cos(omega * t)
    y_dot = 2 * b * omega * np.cos(2 * omega * t)

    return np.array([x,y]), np.array([x_dot,y_dot])

def elliptical_trajectory(t, a=1.0, b=1.0, omega=1.0):
    """
    Returns position and velocity for a elliptical trajectory on the XY plane.
    
    Parameters:
        t (float or array): Time
        a (float): Amplitude in X direction
        b (float): Amplitude in Y direction
        omega (float): Angular frequency

    Returns:
        pos, vel: arrays of x and y positions and velocities
    """
    x = eval_traj.pos0[0] + a * np.cos(omega * t)
    y = eval_traj.pos0[1] + b * np.sin(omega * t)

    x_dot = - a * omega * np.sin(omega * t)
    y_dot =   b * omega * np.cos(omega * t)

    return np.array([x,y]), np.array([x_dot,y_dot])