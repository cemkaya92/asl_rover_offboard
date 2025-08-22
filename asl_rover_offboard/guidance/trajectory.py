#!/usr/bin/env python3

import numpy as np

# ==================== Trajectory Generation ====================

def eval_traj(t,states):
    """
    Generates an infinity 
    """
    # === Define Trajectory Parameters ===
    X_RADIUS = 5.0
    Y_RADIUS = 5.0  
    OMEGA = 5.0 * np.pi / 180.0     # angular speed rad/s

    # === Initialize the current position ===
    if not hasattr(eval_traj, "states"):
        eval_traj.states = states  # Store first t as offset
    

    return straight_line_trajectory(t,1.0)


def figure8_trajectory(t, a=1.0, b=1.0, omega=1.0):
    """
    Returns position and velocity for a figure-8 trajectory on the XY plane.
    
    Parameters:
        t (float or array): Time
        a (float): Amplitude in X direction
        b (float): Amplitude in Y direction
        omega (float): Angular frequency

    Returns:
        pos, vel, acc: arrays of x and y positions, velocities, and accelerations
    """
    x = eval_traj.states[0] + a * np.sin(omega * t)
    y = eval_traj.states[1] + b * np.sin(2 * omega * t)

    x_dot = a * omega * np.cos(omega * t)
    y_dot = 2 * b * omega * np.cos(2 * omega * t)

    x_ddot = - a * omega * omega * np.sin(omega * t)
    y_ddot = - 4 * b * omega * omega * np.sin(2 * omega * t)

    return np.array([x,y]), np.array([x_dot,y_dot]), np.array([x_ddot,y_ddot])

def elliptical_trajectory(t, a=1.0, b=1.0, omega=1.0):
    """
    Returns position and velocity for a elliptical trajectory on the XY plane.
    
    Parameters:
        t (float or array): Time
        a (float): Amplitude in X direction
        b (float): Amplitude in Y direction
        omega (float): Angular frequency

    Returns:
        pos, vel, acc: arrays of x and y positions, velocities, and accelerations
    """
    x = eval_traj.states[0] + a * np.cos(omega * t)
    y = eval_traj.states[1] + b * np.sin(omega * t)

    x_dot = - a * omega * np.sin(omega * t)
    y_dot =   b * omega * np.cos(omega * t)

    x_ddot = - a * omega * omega * np.cos(omega * t)
    y_ddot = - b * omega * omega * np.sin(omega * t)

    return np.array([x,y]), np.array([x_dot,y_dot]), np.array([x_ddot,y_ddot])


def straight_line_trajectory(t, vel=1.0):
    """
    Returns position and velocity for a straight line trajectory on the XY plane.
    
    Parameters:
        t (float or array): Time
        vel (float): Linear speed in body x axis

    Returns:
        pos, vel, acc: arrays of x and y positions, velocities, and accelerations
    """
    yaw = 0.0
    x = eval_traj.states[0] + vel * np.cos(yaw) * t
    y = eval_traj.states[1] + vel * np.sin(yaw) * t

    x_dot = vel * np.cos(yaw)
    y_dot = vel * np.sin(yaw)

    x_ddot = 0.0
    y_ddot = 0.0

    return np.array([x,y]), np.array([x_dot,y_dot]), np.array([x_ddot,y_ddot])