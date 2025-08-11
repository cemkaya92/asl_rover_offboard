#!/usr/bin/env python3


import numpy as np
from casadi import SX, vertcat, diag, nlpsol
import casadi as ca

from asl_rover_offboard.model.dynamics_model import build_casadi_model


# ==================== MPC Solver Class ====================
class MPCSolver:
    def __init__(self, mpc_params, vehicle_params, debug=False):
        self.N = mpc_params.N
        self.Ts = 1.0 / mpc_params.frequency
        self.Q = np.diag(mpc_params.Q)
        self.R = np.diag(mpc_params.R)
        self.Qf_factor = mpc_params.Qf_factor
        self.debug = debug

        self.v_min, self.v_max = 0.0, mpc_params.v_max
        self.w_min, self.w_max = -mpc_params.w_max, mpc_params.w_max


        # === Load CasADi dynamic model with UAV params ===
        _, F = build_casadi_model(self.Ts)
        self.F = F

        # === Define optimization variables ===
        opti = ca.Opti()
        X = opti.variable(3, self.N+1)
        U = opti.variable(2, self.N)

        x_param    = opti.parameter(3, 1)       # current state
        xref_param = opti.parameter(3, self.N+1)     # horizon reference
        O_param    = opti.parameter(2, self.N+1)     # obstacle center over horizon
        R_safe     = opti.parameter(1, 1)       # safety radius

        opti.subject_to(X[:,0] == x_param)


        Qc  = ca.MX(self.Q)
        Rc  = ca.MX(self.R)
        Qf  = ca.MX(self.Qf_factor*self.Q)
        J   = 0

        for k in range(self.N):
            opti.subject_to(X[:, k+1] == F(X[:, k], U[:, k]))
            opti.subject_to(opti.bounded(self.v_min, U[0, k], self.v_max))
            opti.subject_to(opti.bounded(self.w_min, U[1, k], self.w_max))

            ex = X[:, k] - xref_param[:, k]
            # wrap yaw error
            ex[2] = ca.fmod(ex[2] + np.pi, 2*np.pi) - np.pi
            J += ca.mtimes([ex.T, Qc, ex]) + ca.mtimes([U[:, k].T, Rc, U[:, k]])

            # obstacle avoidance: (x_{k+1} - ox)^2 + (y_{k+1} - oy)^2 >= R_safe^2
            dx = X[0, k+1] - O_param[0, k+1]
            dy = X[1, k+1] - O_param[1, k+1]
            opti.subject_to(dx*dx + dy*dy >= R_safe[0]*R_safe[0])

        exN = X[:, self.N] - xref_param[:, self.N]
        exN[2] = ca.fmod(exN[2] + np.pi, 2*np.pi) - np.pi
        J += ca.mtimes([exN.T, Qf, exN])

        opti.minimize(J)
        opti.solver('ipopt', {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 120,
        })

        self.opti = opti
        self.X, self.U = X, U
        self.x_param, self.xref_param = x_param, xref_param
        self.O_param, self.R_safe = O_param, R_safe

        # rolling warm-start buffers
        self.X_prev = None
        self.U_prev = None

    def _dummy_obstacle(self):
        # Hide obstacle far away
        return np.array([[1e5]*(self.N+1), [1e5]*(self.N+1)])

    def set_initial_guess(self, x0):
        if self.X_prev is None:
            X0 = np.tile(x0.reshape(3,1), (1, self.N+1))
            U0 = np.zeros((2, self.N))
            self.X_prev, self.U_prev = X0, U0

    def solve(self, x_now: np.ndarray,
              xref_h: np.ndarray,
              obs_center_h: np.ndarray | None,
              r_safe: float):
        """
        x_now: (3,)
        xref_h: (3, N+1)
        obs_center_h: (2, N+1) or None
        """
        self.set_initial_guess(x_now)

        if obs_center_h is None:
            O_h = self._dummy_obstacle()
            r_val = 0.0
        else:
            O_h = obs_center_h
            r_val = float(r_safe)

        self.opti.set_value(self.x_param,   x_now.reshape(3,1))
        self.opti.set_value(self.xref_param, xref_h)
        self.opti.set_value(self.O_param,    O_h)
        self.opti.set_value(self.R_safe,     np.array([[r_val]]))

        # warm start (shifted)
        Xs = np.hstack([self.X_prev[:, 1:], self.X_prev[:, -1][:, None]])
        Us = np.hstack([self.U_prev[:, 1:], self.U_prev[:, -1][:, None]])
        self.opti.set_initial(self.X, Xs)
        self.opti.set_initial(self.U, Us)

        try:
            sol = self.opti.solve()
            U_sol = sol.value(self.U) # (2, N)
            X_sol = sol.value(self.X) # (3, N+1)
            self.U_prev = U_sol
            self.X_prev = X_sol
            u0 = U_sol[:, 0].copy()
            return True, u0, X_sol, U_sol
        except RuntimeError:
            # fall back to previous
            if self.U_prev is None:
                u0 = np.zeros(2)
            else:
                u0 = self.U_prev[:, 0].copy()
            return False, u0, self.X_prev, self.U_prev