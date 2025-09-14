#!/usr/bin/env python3
import math
import numpy as np
from typing import Optional, Tuple, Callable, List, Dict

# ==================== Trajectory Generation ====================
class TrajectoryGenerator:
    """
    State layout (np.ndarray shape (9,)):
        p   = [x, y, psi]
        v   = [xd, yd, psid]
        a   = [xdd, ydd, psidd]
    Units: meters, radians, seconds (SI)
    """
    def __init__(self, v_max: float = 1.0, omega_max: float = 1.0):
        self._v_max = float(v_max)
        self._omega_max = float(omega_max)

        self._generated: bool = False
        self._duration: Optional[float] = None
        self._coeffs = None                 # for minimum-jerk: list of 3 arrays [a0..a5]
        self._ref_func: Optional[Callable[[float], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None
        self._traj_name: Optional[str] = None
        self._repeat_mode = "none"     # "none", "loop", "pingpong"
        self._post_behavior = "hold"   # currently used for non-repeating polynomials


    # -------------------- Public API --------------------

    def generate_circular_trajectory(self, state_current, speed, yaw_rate,
                                    duration=None, repeat="loop"):
        """
        Constant-twist arc (includes straight line when yaw_rate≈0).
        - If duration is None and |yaw_rate|>0, default is one full circle: T = 2π/|yaw_rate|.
        - If duration is None and |yaw_rate|≈0, default T = 5.0 s.
        """
        p0, v0, a0 = state_current[0:3].astype(float), state_current[3:6].astype(float), state_current[6:9].astype(float)
        x0, y0, psi0 = p0
        v = float(np.clip(speed, -self._v_max, self._v_max))
        w = float(np.clip(yaw_rate, -self._omega_max, self._omega_max))

        # Default duration
        if duration is None:
            T = (2.0*np.pi/abs(w)) if abs(w) > 1e-6 else np.inf  # circle loops; straight is unbounded
        else:
            T = float(duration)

        # Build a reference function f(t)->(p,v,a)
        if abs(w) < 1e-6:
            # Straight line with heading psi0
            cpsi, spsi = np.cos(psi0), np.sin(psi0)

            def _ref(t: float):
                x = x0 + v * t * cpsi
                y = y0 + v * t * spsi
                psi = psi0
                xd = v * cpsi
                yd = v * spsi
                psid = 0.0
                xdd = 0.0
                ydd = 0.0
                psidd = 0.0
                return np.array([x, y, psi]), np.array([xd, yd, psid]), np.array([xdd, ydd, psidd])
        else:
            R = v / w
            def _ref(t: float):
                psi = psi0 + w * t
                # Integrated kinematics
                x = x0 + R * (np.sin(psi) - np.sin(psi0))
                y = y0 - R * (np.cos(psi) - np.cos(psi0))
                xd = v * np.cos(psi)
                yd = v * np.sin(psi)
                psid = w
                xdd = -v * w * np.sin(psi)
                ydd =  v * w * np.cos(psi)
                psidd = 0.0
                return np.array([x, y, psi]), np.array([xd, yd, psid]), np.array([xdd, ydd, psidd])

        self._duration = T
        self._repeat_mode = repeat  # "loop" by default for circles
        self._post_behavior = "hold"
        self._ref_func = _ref
        self._coeffs = None
        self._traj_name = "arc" if abs(w) > 1e-6 else "straight (constant heading)"
        self._generated = True

    def generate_straight_trajectory(self, state_current, distance=None, speed=None,
                                 duration=None, heading=None, repeat="none"):
        """
        Straight segment at constant heading and constant linear speed.
        You can specify either (distance & speed) or (duration & speed).
        If heading is None, uses current psi0.
        """
        p0 = state_current[0:3].astype(float)
        x0, y0, psi0 = p0
        psi = psi0 if heading is None else float(heading)

        if speed is None:
            speed = min(self._v_max, 0.5 * self._v_max)
        v = float(np.clip(speed, -self._v_max, self._v_max))

        if duration is None:
            if distance is None:
                distance = 5.0  # default 5 m segment
            T = abs(distance / v) if abs(v) > 1e-9 else 0.0
        else:
            T = float(duration)

        cpsi, spsi = np.cos(psi), np.sin(psi)

        def _ref(t: float):
            x = x0 + v * t * cpsi
            y = y0 + v * t * spsi
            xd = v * cpsi
            yd = v * spsi
            xdd = 0.0
            ydd = 0.0
            return np.array([x, y, psi]), np.array([xd, yd, 0.0]), np.array([xdd, ydd, 0.0])

        self._duration = T  # finite if distance/duration provided
        self._repeat_mode = repeat      # set "loop" to repeat the segment forever
        self._post_behavior = "hold"
        self._ref_func = _ref
        self._coeffs = None
        self._traj_name = "straight (segment)"
        self._generated = True

    def generate_minimum_jerk_line_to(self, state_current, target_pose, duration,
                                  repeat="none", post_behavior="hold"):
        """
        5th-order (quintic) min-jerk from current (p, v, a) to target (p, v, a) over T.
        target_pose: shape (3,) for [x_f, y_f, psi_f] (final v=a assumed 0),
                     OR shape (9,) to specify full end (p, v, a).
        """
        p0 = state_current[0:3].astype(float)
        v0 = state_current[3:6].astype(float)
        a0 = state_current[6:9].astype(float)

        if target_pose.size == 3:
            pf = target_pose.astype(float)
            # Wrap yaw to the nearest equivalent
            pf[2] = p0[2] + self._wrap_to_pi(pf[2] - p0[2])
            vf = np.zeros(3)
            af = np.zeros(3)
        elif target_pose.size == 9:
            pf = target_pose[0:3].astype(float)
            vf = target_pose[3:6].astype(float)
            af = target_pose[6:9].astype(float)
            pf[2] = p0[2] + self._wrap_to_pi(pf[2] - p0[2])
        else:
            raise ValueError("target_pose must be shape (3,) or (9,)")

        T = float(duration)
        coeffs = []
        for i in range(3):
            coeffs.append(self._plan_mj(p0[i], v0[i], a0[i], pf[i], vf[i], af[i], T))

        self._coeffs = coeffs
        self._duration = float(duration)
        self._repeat_mode = repeat          # "loop" or "pingpong" for indefinite
        self._post_behavior = post_behavior # "hold" after T if repeat=='none'
        self._ref_func = None
        self._traj_name = "minimum-jerk line-to"
        self._generated = True


    def generate_piecewise_track(
        self,
        state_start: np.ndarray,
        segments: List[Dict],
        repeat: str = "loop",
        name: str = "piecewise track",
    ):
        """
        Build a single reference (p,v,a) by concatenating straight and arc segments.
        Each segment is a dict with:
          - type: 'straight' or 'arc'
          - For 'straight': {'type':'straight', 'length': L, 'speed': v}
              (moves forward L meters at constant heading)
          - For 'arc': {'type':'arc', 'radius': R, 'angle': theta, 'speed': v}
              (turns by signed theta radians at constant speed and curvature)
              theta<0 => CCW (left); theta>0 => CW (right)
        """
        assert state_start.shape == (9,), "state_start must be (9,) [p v a]"
        p0 = state_start[0:3].astype(float)  # x, y, psi
        v0 = state_start[3:6].astype(float)
        a0 = state_start[6:9].astype(float)

        # Build closures for each segment, tracking start pose and duration
        seg_funcs: List[Tuple[float, callable]] = []
        # xk, yk, psik = p0.tolist(), p0.tolist(), p0.tolist()  # not used; just a scratch note
        xk, yk, psik = p0[0], p0[1], p0[2]

        for seg in segments:
            stype = seg["type"].lower()
            if stype == "straight":
                L = float(seg["length"])
                v = float(np.clip(seg["speed"], -self._v_max, self._v_max))
                if abs(v) < 1e-9:
                    raise ValueError("straight segment speed must be nonzero")
                T = abs(L / v)

                cpsi, spsi = np.cos(psik), np.sin(psik)

                # closure evaluates this segment from its own local time τ∈[0,T]
                def make_straight(x0, y0, psi0, v, T):
                    cpsi0, spsi0 = np.cos(psi0), np.sin(psi0)
                    def _ref_tau(tau: float):
                        # clamp for hold-at-end behavior inside segment
                        tt = min(max(0.0, tau), T)
                        x = x0 + v * tt * cpsi0
                        y = y0 + v * tt * spsi0
                        psi = psi0
                        xd = v * cpsi0
                        yd = v * spsi0
                        psid = 0.0
                        xdd = 0.0
                        ydd = 0.0
                        psidd = 0.0
                        return (np.array([x, y, psi]),
                                np.array([xd, yd, psid]),
                                np.array([xdd, ydd, psidd]))
                    return _ref_tau

                seg_funcs.append((T, make_straight(xk, yk, psik, v, T)))

                # advance the "cursor" pose to next segment start
                xk += L * cpsi
                yk += L * spsi
                # heading unchanged for straight

            elif stype == "arc":
                R = float(seg["radius"])
                theta = float(seg["angle"])
                v = float(np.clip(seg["speed"], -self._v_max, self._v_max))
                if R <= 0.0:
                    raise ValueError("arc radius must be > 0")
                if abs(v) < 1e-9:
                    raise ValueError("arc segment speed must be nonzero")

                w = v / R  # signed with v; direction controlled by theta's sign via duration
                # We want to cover |theta| radians: duration T = |theta| / |w|
                T = abs(theta) / abs(w)

                def make_arc(x0, y0, psi0, v, w, T, theta_sign):
                    # Use your existing constant-twist integration
                    def _ref_tau(tau: float):
                        tt = min(max(0.0, tau), T)
                        psi = psi0 + np.sign(theta_sign) * abs(w) * tt
                        Rloc = v / (np.sign(theta_sign) * abs(w))
                        x = x0 + Rloc * (np.sin(psi) - np.sin(psi0))
                        y = y0 - Rloc * (np.cos(psi) - np.cos(psi0))
                        xd = v * np.cos(psi)
                        yd = v * np.sin(psi)
                        psid = np.sign(theta_sign) * abs(w)
                        xdd = -v * psid * np.sin(psi)
                        ydd =  v * psid * np.cos(psi)
                        psidd = 0.0
                        return (np.array([x, y, psi]),
                                np.array([xd, yd, psid]),
                                np.array([xdd, ydd, psidd]))
                    return _ref_tau

                seg_funcs.append((T, make_arc(xk, yk, psik, abs(v), abs(w), T, theta_sign=np.sign(theta))))

                # advance pose to end of arc
                psik = psik + theta
                # position update using the same integration at t=T:
                psi_end = psik
                # Use local R with signed turn matching theta
                w_signed = np.sign(theta) * abs(w)
                Rloc = abs(v) / abs(w) * np.sign(theta)  # signed radius for CW/CCW
                # Better: compute with plain formulas using start psi and theta
                # End-point delta:
                xk = xk + (abs(v)/abs(w)) * (np.sin(psik) - np.sin(psik - theta))
                yk = yk - (abs(v)/abs(w)) * (np.cos(psik) - np.cos(psik - theta))

            else:
                raise ValueError(f"unknown segment type: {stype}")

        # Total period
        Ttot = sum(Ti for Ti, _ in seg_funcs)

        # master ref that selects the active segment
        def _ref_master(t: float):
            if Ttot <= 0.0:
                # degenerate
                return np.array([xk, yk, psik]), np.zeros(3), np.zeros(3)

            # handle repetition
            tt, ended = self._time_map(t)  # will mod if repeat='loop'
            # when repeat='none', _time_map clamps at T; we still need total duration
            if self._repeat_mode == "none":
                tt = min(tt, Ttot)

            # walk the segments
            acc = 0.0
            for Ti, f in seg_funcs:
                if tt <= acc + Ti or np.isclose(tt, acc + Ti):
                    return f(tt - acc)
                acc += Ti
            # if we fell through due to float rounding, return end of last segment
            return seg_funcs[-1][1](seg_funcs[-1][0])

        # Register as current trajectory
        self._duration = Ttot
        self._repeat_mode = repeat
        self._post_behavior = "hold"
        self._ref_func = _ref_master
        self._coeffs = None
        self._traj_name = name
        self._generated = True

    def generate_rounded_rectangle(
        self,
        state_start: np.ndarray,
        width: float,
        height: float,
        corner_radius: float,
        speed: float,
        cw: bool = True,
        repeat: str = "loop",
    ):
        """
        Rectangle with arc (fillet) corners.
        - width: total in X of the centerline rectangle
        - height: total in Y of the centerline rectangle
        - corner_radius: fillet radius (applied at all 4 corners)
        Requirement: width > 2r, height > 2r
        Start heading = state_start.psi; first segment is a straight.
        """
        W = float(width); H = float(height); r = float(corner_radius)
        if W <= 2*r or H <= 2*r:
            raise ValueError("width and height must be > 2*corner_radius")

        Lx = W - 2.0*r
        Ly = H - 2.0*r
        turn = (+1.0 if cw else -1.0)

        segs = [
            {'type':'straight', 'length': Lx, 'speed': speed},
            {'type':'arc',      'radius': r,  'angle':  turn*math.pi/2, 'speed': speed},
            {'type':'straight', 'length': Ly, 'speed': speed},
            {'type':'arc',      'radius': r,  'angle':  turn*math.pi/2, 'speed': speed},
            {'type':'straight', 'length': Lx, 'speed': speed},
            {'type':'arc',      'radius': r,  'angle':  turn*math.pi/2, 'speed': speed},
            {'type':'straight', 'length': Ly, 'speed': speed},
            {'type':'arc',      'radius': r,  'angle':  turn*math.pi/2, 'speed': speed},
        ]
        self.generate_piecewise_track(state_start, segs, repeat=repeat, name="rounded-rectangle")

    def generate_racetrack_capsule(
        self,
        state_start: np.ndarray,
        straight_length: float,
        radius: float,
        speed: float,
        cw: bool = True,
        repeat: str = "loop",
    ):
        """
        Two parallel straights of length L connected by semicircles of radius r.
        Start heading = state_start.psi; first segment is a straight.
        """
        L = float(straight_length); r = float(radius)
        if L < 0.0 or r <= 0.0:
            raise ValueError("straight_length must be >= 0 and radius > 0")

        turn = (+1.0 if cw else -1.0)
        segs = [
            {'type':'straight', 'length': L, 'speed': speed},
            {'type':'arc',      'radius': r, 'angle':  turn*math.pi, 'speed': speed},
            {'type':'straight', 'length': L, 'speed': speed},
            {'type':'arc',      'radius': r, 'angle':  turn*math.pi, 'speed': speed},
        ]
        self.generate_piecewise_track(state_start, segs, repeat=repeat, name="capsule-oval")    

    def get_ref_at_time(self, t: float):
        """
        Returns (p, v, a) for clamped time in [0, duration].
        If no trajectory has been generated yet → (None, None, None).
        """
        if not self._generated:
            return None, None, None

        tc, ended = self._time_map(t)

        if self._ref_func is not None:
            # closed-form primitive (arc/straight/etc.)
            # For finite T with repeat='none' and t>T, we "hold" by evaluating at T.
            if ended and np.isfinite(self._duration):
                return self._ref_func(self._duration)
            return self._ref_func(tc)

        # Polynomial (minimum-jerk) path
        if ended and self._repeat_mode == "none":
            if self._post_behavior == "hold":
                tc = self._duration  # hold at final pose
            # (If you later want 'extrapolate', implement here.)
        p = np.array([self._mj_eval(self._coeffs[i], tc)[0] for i in range(3)])
        v = np.array([self._mj_eval(self._coeffs[i], tc)[1] for i in range(3)])
        a = np.array([self._mj_eval(self._coeffs[i], tc)[2] for i in range(3)])
        return p, v, a

    # -------------------- Private Helpers --------------------

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        """Map angle to (-pi, pi]."""
        a = (angle + np.pi) % (2.0 * np.pi) - np.pi
        # ensure +pi maps to +pi (not -pi)
        if a <= -np.pi:
            a += 2.0 * np.pi
        return a

    @staticmethod
    def _plan_mj(p0, v0, a0, pf, vf, af, T) -> np.ndarray:
        """
        Quintic coefficients a0..a5 such that:
            p(0)=p0, p'(0)=v0, p''(0)=a0
            p(T)=pf, p'(T)=vf, p''(T)=af
        """
        T1 = T
        T2 = T1 * T1
        T3 = T2 * T1
        T4 = T3 * T1
        T5 = T4 * T1

        A0 = p0
        A1 = v0
        A2 = 0.5 * a0

        # RHS residuals at T after accounting for A0..A2
        c0 = pf - (A0 + A1*T1 + A2*T2)
        c1 = vf - (A1 + 2.0*A2*T1)
        c2 = af - (2.0*A2)

        # Solve for A3..A5
        M = np.array([
            [   T3,     T4,      T5],
            [ 3*T2,   4*T3,    5*T4],
            [ 6*T1,  12*T2,   20*T3]
        ], dtype=float)
        b = np.array([c0, c1, c2], dtype=float)
        A3, A4, A5 = np.linalg.solve(M, b)

        return np.array([A0, A1, A2, A3, A4, A5], dtype=float)

    @staticmethod
    def _mj_eval(coeffs: np.ndarray, t: float):
        """
        Evaluate quintic at time t: returns (pos, vel, acc).
        coeffs = [a0..a5]
        """
        a0, a1, a2, a3, a4, a5 = coeffs.tolist()
        tt = t
        tt2 = tt*tt
        tt3 = tt2*tt
        tt4 = tt3*tt
        tt5 = tt4*tt

        p = a0 + a1*tt + a2*tt2 + a3*tt3 + a4*tt4 + a5*tt5
        v = a1 + 2*a2*tt + 3*a3*tt2 + 4*a4*tt3 + 5*a5*tt4
        a = 2*a2 + 6*a3*tt + 12*a4*tt2 + 20*a5*tt3
        return p, v, a

    
    def _time_map(self, t: float):
        """Map external time t to an internal time in [0, T] or unbounded."""
        t = max(0.0, float(t))
        T = self._duration

        if T is None or np.isinf(T):
            # Unbounded trajectory (no clamping / repeating).
            return t, False  # ended=False

        if self._repeat_mode == "none":
            return min(t, T), (t >= T)

        if T <= 0.0:
            return 0.0, False

        if self._repeat_mode == "loop":
            return t % T, False

        if self._repeat_mode == "pingpong":
            cycle = 2.0 * T
            tau = t % cycle
            return (tau if tau <= T else (2.0 * T - tau)), False

        # Fallback
        return min(t, T), (t >= T)

    
