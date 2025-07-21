#!/usr/bin/env python3


import numpy as np
from casadi import SX, vertcat, diag, nlpsol
import casadi as ca

from asl_rover_offboard.model.dynamics_model import build_casadi_model


# ==================== MPC Solver Class ====================
class MPCSolver:
    def __init__(self, mpc_params, vehicle_params, debug=False):
        self.N = mpc_params.N
        self.Tf = mpc_params.horizon
        self.dt = mpc_params.horizon / mpc_params.N
        self.debug = debug

        self.Qv = 1.0
        self.Qomega = 0.5

        # === Load CasADi dynamic model with UAV params ===
        self.f_model, self.NX, self.NU = build_casadi_model(vehicle_params)

        # === Define optimization variables ===
        X = SX.sym('X', self.NX, self.N + 1) # States over the horizon
        U = SX.sym('U', self.NU, self.N)     # Controls over the horizon
        P = SX.sym('P', 2 * self.NX + 2)    # Parameters (initial state x0, reference xref)  +2 for v_ref and omega_ref

        # Extract initial state and reference from parameters
        x0_param = P[0:self.NX]
        xref_param = P[self.NX:]

        # Cost Function (penalizes state error and control effort)
        # Tuning these matrices is crucial for performance
        Q = diag(mpc_params.Q)  
                        # Pose error [x, y, psi]
        R = diag(mpc_params.R) # Control effort (v, omega)
        cost = 0
        
        # Dynamics Constraints
        g = []
        g.append(X[:, 0] - x0_param) # Initial state constraint

        for k in range(self.N):

            # Position error (x, y)
            pos_err = X[0:2, k] - xref_param[0:2]

            # Heading error (wrapped)
            psi_err = self.angle_diff(X[2, k], xref_param[2])

            # Compute cost
            cost += pos_err.T @ Q[0:2,0:2] @ pos_err
            cost += psi_err * Q[2,2] * psi_err

            # Add reference velocity cost
            cost += self.Qv * (U[0, k] - xref_param[3])**2
            cost += self.Qomega * (U[1, k] - xref_param[4])**2

            #cost += U[:,k].T @ R @ U[:,k]  
            
            # Add dynamics constraint (Explicit Euler integration)
            x_next_pred = X[:,k] + self.dt * self.f_model(X[:,k], U[:,k])
            g.append(X[:,k+1] - x_next_pred)
        
        # Final state cost
        cost += (X[:, self.N] - xref_param[0:3]).T @ Q @ (X[:, self.N] - xref_param[0:3])

        # NLP problem setup
        g = vertcat(*g)
        OPT_VARS = vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp = {'x': OPT_VARS, 'f': cost, 'g': g, 'p': P}
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': False,
            'ipopt.tol': 1e-4,
            'ipopt.max_iter': 100
        }

        self.solver = nlpsol('solver', 'ipopt', nlp, opts)
        
        # Bounds for constraints (all dynamics constraints are equality)
        self.lbg = np.zeros(g.shape)
        self.ubg = np.zeros(g.shape)
        self.x_guess = np.zeros(OPT_VARS.shape)

    def solve(self, x0, xref):
        p = np.concatenate([x0, xref])
        sol = self.solver(x0=self.x_guess, p=p, lbg=self.lbg, ubg=self.ubg)
        w_opt = sol['x'].full().flatten()

        # Warm start for next iteration
        self.x_guess = w_opt

        # Extract the first optimal control action
        u0 = w_opt[self.NX * (self.N + 1) : self.NX * (self.N + 1) + self.NU]

        if self.debug:
            print(f"\n[MPC] Solving for state x0: {np.round(x0, 2)}")
            print(f"[MPC] Reference xref: {np.round(xref, 2)}")
            print(f"[MPC] Control output: {np.round(u0, 3)}")

        return u0

    def angle_diff(self,a, b):
        """
        CasADi-compatible wrapped angle difference in [-pi, pi]
        """
        return ca.atan2(ca.sin(a - b), ca.cos(a - b))