# asl_rover_offboard/navigation/trajectory_manager.py
from __future__ import annotations
import math
import numpy as np
import rclpy

from px4_msgs.msg import TrajectorySetpoint6dof as TrajMsg
from custom_offboard_msgs.msg import TrajectoryPlan


from asl_rover_offboard.guidance.trajectory_generator import TrajectoryGenerator

class TrajectoryManager:
    """
    Handles segment planning (min-jerk), loiter reference, and publishing.
    Does NOT decide states — only provides references / planning.
    """
    def __init__(
        self, 
        v_max: float, 
        omega_max: float, 
        default_T: float
    ):
        self.gen = TrajectoryGenerator(
            v_max=v_max, 
            omega_max=omega_max
        )

        self.default_T = float(default_T)
        self.plan_active = False
        self._plan_start_time = 0.0
        self._plan_type: str | None = None
        self._plan_meta: dict = {}   # stash params needed to reconstruct


    # ---------- planning ----------
    def plan_min_jerk(
        self, 
        t_now: float, 
        p0: np.ndarray, 
        v0: np.ndarray,
        p1: np.ndarray, 
        v1: np.ndarray, 
        T: float | None = None,
        repeat: str = "none") -> None:
        p0 = np.asarray(p0,float).reshape(3,)
        v0 = np.asarray(v0,float).reshape(3,)
        p1 = np.asarray(p1,float).reshape(3,)
        # hold yaw = psi0 estimated from v0 (fallback 0)
        psi0 = float(np.arctan2(v0[1], v0[0])) if np.linalg.norm(v0[:2]) > 1e-3 else 0.0
        state0 = np.array([p0[0], p0[1], psi0, v0[0], v0[1], 0.0, 0.0, 0.0, 0.0])

        # end yaw = psi0 (change if you want a different final yaw)
        goal_pose = np.array([p1[0], p1[1], psi0])
        T = float(T if T is not None else self.default_T)

        self.gen.generate_minimum_jerk_line_to(state0, goal_pose, duration=T,
                                            repeat=repeat, post_behavior="hold")
        self.plan_active = True
        self._plan_start_time = float(t_now)

        self._plan_type = "min_jerk"
        self._plan_meta = {
            "repeat": repeat,
            "duration": T,
            "state0": state0.copy(),
            # NOTE: access generator coeffs for export (same process/package)
            "coeffs": [c.copy() for c in self.gen._coeffs],  # 3 arrays, each 6 coeffs
        }

    def plan_arc_by_angle(
        self,
        t_now: float,
        p0: np.ndarray,
        heading: float,
        radius: float,
        angle: float,
        speed: float,
        repeat: str = "loop",
    ) -> None:
        """
        Arc defined by geometry: radius r and turn angle 'angle' (rad).
        Positive angle = CW, negative = CCW. Travels at (clamped) 'speed'.
        Computes yaw_rate = sign(angle) * (speed / r) and duration = |angle| / |yaw_rate|.
        If omega_max requires clamping yaw_rate, duration is adjusted accordingly.
        """

        p0 = np.asarray(p0, float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])

        # Clamp speed to the generator's limits (to keep math consistent with what will run)
        v_max = float(self.gen._v_max)
        w_max = float(self.gen._omega_max)
        v = float(np.clip(speed, -v_max, v_max))

        if radius <= 0.0:
            raise ValueError("plan_arc_by_angle: radius must be > 0")

        # Desired yaw_rate from geometry (sign from angle)
        w_des = np.sign(angle) * (abs(v) / float(radius))

        # Enforce omega limit; if we clamp yaw rate, recompute duration
        w = float(np.clip(w_des, -w_max, w_max))
        # Duration needed to sweep the requested angle with the actual yaw rate
        T = (abs(angle) / max(abs(w), 1e-9))

        # Generate
        self.gen.generate_circular_trajectory(
            state0, speed=v, yaw_rate=w, duration=T, repeat=repeat
        )

        self.plan_active = True
        self._plan_start_time = float(t_now)

        self._plan_type = "arc"
        self._plan_meta = {
            "repeat": repeat,
            "duration": float(T),
            "state0": state0.copy(),
            "speed": float(v),
            "yaw_rate": float(w),
            # optional bookkeeping if you want to export geometry later:
            "radius": float(radius),
            "angle": float(angle),
            "heading": float(heading),
        }


    def plan_arc_by_rate(
        self,
        t_now: float,
        p0: np.ndarray,
        heading: float,
        radius: float,            # kept for API symmetry (not strictly required here)
        yaw_rate: float,
        speed: float,
        duration: float | None = None,
        repeat: str = "loop",
    ) -> None:
        """
        Arc defined by kinematics: constant yaw_rate and speed.
        If 'duration' is None, it defaults to a full circle per generator behavior.
        Note: provided 'radius' is not enforced unless your (speed, yaw_rate) satisfy v = |w| * r.
        """

        p0 = np.asarray(p0, float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])

        v_max = float(self.gen._v_max)
        w_max = float(self.gen._omega_max)
        v = float(np.clip(speed, -v_max, v_max))
        w = float(np.clip(yaw_rate, -w_max, w_max))

        # If you prefer to exactly enforce the requested 'radius', you could
        # overwrite v := |w| * radius (and clamp), but we keep 'speed' authoritative.

        self.gen.generate_circular_trajectory(
            state0, speed=v, yaw_rate=w, duration=duration, repeat=repeat
        )

        self.plan_active = True
        self._plan_start_time = float(t_now)

        self._plan_type = "arc"
        self._plan_meta = {
            "repeat": repeat,
            "duration": (float("inf") if duration is None else float(duration)),
            "state0": state0.copy(),
            "speed": float(v),
            "yaw_rate": float(w),
            # optional notes:
            "heading": float(heading),
            "radius_hint": float(radius),
        }

    def plan_straight(self, t_now: float, p0: np.ndarray, heading: float,
                    speed: float, distance: float | None = None,
                    duration: float | None = None, repeat: str = "none") -> None:
        p0 = np.asarray(p0,float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])
        self.gen.generate_straight_trajectory(state0, distance=distance, speed=speed,
                                            duration=duration, heading=heading, repeat=repeat)
        self.plan_active = True
        self._plan_start_time = float(t_now)

        self._plan_type = "straight"
        self._plan_meta = {
            "repeat": repeat,
            "duration": float(self.gen._duration),
            "state0": state0.copy(),
            "speed": float(speed),
            "heading": float(heading),
            "distance": 0.0 if distance is None else float(distance),
        }

    def plan_rounded_rectangle(
        self,
        t_now: float,
        p0: np.ndarray,
        heading: float,
        width: float,
        height: float,
        corner_radius: float,
        speed: float,
        cw: bool = True,
        repeat: str = "loop",
    ) -> None:
        """
        Builds a rectangle with identical fillets at all 4 corners.
        Start pose = (p0, heading). First segment is a straight along 'heading'.
        Requires: width > 2*corner_radius and height > 2*corner_radius.
        """
        p0 = np.asarray(p0, float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])

        # relies on TrajectoryGenerator.generate_rounded_rectangle(...)
        self.gen.generate_rounded_rectangle(
            state_start=state0,
            width=float(width),
            height=float(height),
            corner_radius=float(corner_radius),
            speed=float(speed),
            cw=bool(cw),
            repeat=repeat,
        )

        self.plan_active = True
        self._plan_start_time = float(t_now)
        self._plan_type = "rounded_rectangle"
        self._plan_meta = {
            "repeat": repeat,
            "duration": float(self.gen._duration),
            "state0": state0.copy(),
            "width": float(width),
            "height": float(height),
            "corner_radius": float(corner_radius),
            "speed": float(speed),
            "cw": bool(cw),
        }


    def plan_racetrack_capsule(
        self,
        t_now: float,
        p0: np.ndarray,
        heading: float,
        straight_length: float,
        radius: float,
        speed: float,
        cw: bool = True,
        repeat: str = "loop",
    ) -> None:
        """
        Two parallel straights of length L connected by semicircles of radius r.
        Start pose = (p0, heading). First segment is a straight along 'heading'.
        """
        p0 = np.asarray(p0, float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])

        # relies on TrajectoryGenerator.generate_racetrack_capsule(...)
        self.gen.generate_racetrack_capsule(
            state_start=state0,
            straight_length=float(straight_length),
            radius=float(radius),
            speed=float(speed),
            cw=bool(cw),
            repeat=repeat,
        )

        self.plan_active = True
        self._plan_start_time = float(t_now)
        self._plan_type = "racetrack_capsule"
        self._plan_meta = {
            "repeat": repeat,
            "duration": float(self.gen._duration),
            "state0": state0.copy(),
            "straight_length": float(straight_length),
            "radius": float(radius),
            "speed": float(speed),
            "cw": bool(cw),
        }

    # ---------- references ----------
    def get_plan_ref(self, t_now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.plan_active:
            return None, None, None
        tau = float(t_now) - float(self._plan_start_time)  # elapsed since plan start
        p, v, a = self.gen.get_ref_at_time(tau)
        if p is None or v is None or a is None:
            self.plan_active = False
            return None, None, None
        return np.asarray(p,float), np.asarray(v,float), np.asarray(a,float)


    # ---------- publish ----------
    @staticmethod
    def publish_traj(pub, clock_now_us: int, p: np.ndarray, v: np.ndarray, a: np.ndarray,
                     yaw: float | None = None) -> None:
        p = np.asarray(p,float).reshape(3,)
        v = np.asarray(v,float).reshape(3,)
        a = np.asarray(a,float).reshape(3,)
        msg = TrajMsg()
        if hasattr(msg, "timestamp"):
            msg.timestamp = int(clock_now_us)

        # attempt common fields
        def set_triplet(obj, names, vec):
            ok = True
            for n, val in zip(names, vec):
                if hasattr(obj, n):
                    setattr(obj, n, float(val))
                else:
                    ok = False
            return ok

        # Try array fields: position / velocity / acceleration
        for field, vec in (("position", p), ("velocity", v), ("acceleration", a)):
            if hasattr(msg, field):
                arr = getattr(msg, field)
                try:
                    arr[:] = np.asarray(vec, float).tolist()
                    continue
                except Exception:
                    pass
            # fallbacks
            if field == "position":
                set_triplet(msg, ("x","y","z"), vec)
            elif field == "velocity":
                set_triplet(msg, ("vx","vy","vz"), vec)
            else:
                set_triplet(msg, ("ax","ay","az"), vec)

        if yaw is not None:
            if hasattr(msg, "yaw"):
                msg.yaw = float(yaw)
            elif hasattr(msg, "heading"):
                msg.heading = float(yaw)

        pub.publish(msg)


    def to_plan_msg(self, t0_us: int) -> TrajectoryPlan:
        """Serialize current plan so another node can reproduce/evaluate it."""
        if not self.plan_active or self._plan_type is None:
            raise RuntimeError("No active plan to export")

        pm = TrajectoryPlan()
        pm.header.stamp = rclpy.clock.Clock().now().to_msg()
        pm.type = self._plan_type
        pm.t0_us = float(t0_us)

        meta = self._plan_meta or {}
        pm.state0   = meta.get("state0", np.zeros(9)).tolist()
        pm.duration = float(meta.get("duration", 0.0))
        pm.repeat   = str(meta.get("repeat", "none"))
        pm.speed    = float(meta.get("speed", 0.0))
        pm.heading  = float(meta.get("heading", 0.0))
        pm.yaw_rate = float(meta.get("yaw_rate", 0.0))
        pm.distance = float(meta.get("distance", 0.0))

        # min-jerk keeps 18 coeffs; others carry params as coeffs
        if pm.type == "min_jerk":
            C = self._coeffs if self._coeffs is not None else np.empty((3,6))
            pm.coeffs = C.reshape(-1).tolist()
        elif pm.type == "rounded_rectangle":
            # Layout: [W, H, r, turn_sign, heading, speed, 0..]
            W = float(meta["width"])
            H = float(meta["height"])
            r = float(meta["corner_radius"])
            turn = 1.0 if meta.get("cw", True) else -1.0
            pm.coeffs = self._pad18([W, H, r, turn, pm.heading, pm.speed])
        elif pm.type == "racetrack_capsule":
            # Layout: [L, r, turn_sign, heading, speed, 0..]
            L = float(meta["straight_length"])
            r = float(meta["radius"])
            turn = 1.0 if meta.get("cw", True) else -1.0
            pm.coeffs = self._pad18([L, r, turn, pm.heading, pm.speed])
        else:
            # For other non-min-jerk types that don’t need params
            pm.coeffs = self._pad18([])

        return pm
    
    
    def _pad18(self, values):
        vals = [float(v) for v in values]
        if len(vals) > 18:
            vals = vals[:18]
        return vals + [0.0] * (18 - len(vals))