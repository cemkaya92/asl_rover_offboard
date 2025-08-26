# asl_rover_offboard/navigation/trajectory_manager.py
from __future__ import annotations
import math
import numpy as np
import rclpy

from px4_msgs.msg import TrajectorySetpoint6dof as TrajMsg
from asl_rover_offboard_msgs.msg import TrajectoryPlan


from asl_rover_offboard.guidance.trajectory_generator import TrajectoryGenerator

class TrajectoryManager:
    """
    Handles segment planning (min-jerk), loiter reference, and publishing.
    Does NOT decide states — only provides references / planning.
    """
    def __init__(self, v_max: float, omega_max: float, default_T: float):
        self.gen = TrajectoryGenerator(v_max=v_max, omega_max=omega_max)
        self.default_T = float(default_T)

        self.plan_active = False
        self._plan_start_time = 0.0

        self._plan_type: str | None = None
        self._plan_meta: dict = {}   # stash params needed to reconstruct


    # ---------- planning ----------
    def plan_min_jerk(self, t_now: float, p0: np.ndarray, v0: np.ndarray,
                    p1: np.ndarray, v1: np.ndarray, T: float | None = None,
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

    def plan_arc(self, t_now: float, p0: np.ndarray, heading: float,
                speed: float, yaw_rate: float,
                duration: float | None = None, repeat: str = "loop") -> None:
        p0 = np.asarray(p0,float).reshape(3,)
        state0 = np.array([p0[0], p0[1], float(heading), 0, 0, 0, 0, 0, 0])
        self.gen.generate_circular_trajectory(state0, speed=speed, yaw_rate=yaw_rate,
                                            duration=duration, repeat=repeat)
        self.plan_active = True
        self._plan_start_time = float(t_now)

        self._plan_type = "arc"
        self._plan_meta = {
            "repeat": repeat,
            "duration": duration,
            "state0": state0.copy(),
            "speed": float(speed),
            "yaw_rate": float(yaw_rate),
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
            "duration": self._duration,
            "state0": state0.copy(),
            "speed": float(speed),
            "heading": float(heading),
            "distance": 0.0 if distance is None else float(distance),
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
        pm.t0_us = float(int(t0_us))
        pm.duration = float(self._plan_meta.get("duration", 0.0))
        pm.repeat = str(self._plan_meta.get("repeat", "none"))

        st0 = self._plan_meta.get("state0", np.zeros(9))
        pm.state0 = np.asarray(st0, float).tolist()

        # coeffs (min_jerk)
        coeffs = self._plan_meta.get("coeffs", None)
        if coeffs is not None:
            C = np.vstack([coeffs[0], coeffs[1], coeffs[2]]).reshape(-1)  # (18,)
            pm.coeffs = C.tolist()
        else:
            pm.coeffs = [float("nan")] * 18

        # primitives
        pm.speed = float(self._plan_meta.get("speed", 0.0))
        pm.yaw_rate = float(self._plan_meta.get("yaw_rate", 0.0))
        pm.heading = float(self._plan_meta.get("heading", 0.0))
        pm.distance = float(self._plan_meta.get("distance", 0.0))
        return pm