#!/usr/bin/env python3
# file: trajectory_visualizer_node.py
from __future__ import annotations
import os, math, numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from ament_index_python.packages import get_package_share_directory

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from px4_msgs.msg import VehicleOdometry

from custom_offboard_msgs.msg import TrajectoryPlan
from asl_rover_offboard.utils.param_loader import ParamLoader

class TrajectoryVisualizer(Node):
    """
    Subscribes to TrajectoryPlan (latched), re-evaluates it using your same
    equations, and publishes:
      - nav_msgs/Path on <viz_path_topic> for RViz XY visualization
      - visualization_msgs/Marker line strip on <viz_marker_topic>
    """
    def __init__(self):
        super().__init__("trajectory_visualizer")

        # ---------- Parameters ----------
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')
        self.declare_parameter('world_frame',      'map')
        self.declare_parameter('viz_path_topic',   'trajectory/xy_path')
        self.declare_parameter('viz_marker_topic', 'trajectory/xy_marker')
        self.declare_parameter('samples',          600)     # number of samples along curve
        self.declare_parameter('max_seconds',      60.0)    # cap for unbounded paths (e.g., loop)
        self.declare_parameter('resample_dt',      0.05)    # used if duration is tiny/0
        self.declare_parameter('vehicle_frame', 'base_link')
        self.declare_parameter('viz_rate_hz', 10.0)
        self.declare_parameter('flip_y', True)

        package_dir = get_package_share_directory('asl_rover_offboard')
        sitl_yaml_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_yaml_file)
        world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        self.vehicle_frame = self.get_parameter('vehicle_frame').get_parameter_value().string_value
        self.viz_rate_hz = float(self.get_parameter('viz_rate_hz').get_parameter_value().double_value)
        self.flip_y = bool(self.get_parameter('flip_y').get_parameter_value().bool_value)

        # Load SITL topics to discover the TrajectoryPlan channel
        sitl_yaml = ParamLoader(sitl_yaml_path)
        trajectory_plan_topic = sitl_yaml.get_topic("trajectory_plan_topic")
        odom_topic = sitl_yaml.get_topic("odometry_topic")

        self.samples      = int(self.get_parameter('samples').get_parameter_value().integer_value)
        self.max_seconds  = float(self.get_parameter('max_seconds').get_parameter_value().double_value)
        self.resample_dt  = float(self.get_parameter('resample_dt').get_parameter_value().double_value)
        self.world_frame  = world_frame

        self._have_odom = False
        self._pos = np.zeros(3, dtype=float)
        self._psi = 0.0
        self._plan_xypsi_world = None

        self.vehicle_frame = world_frame

        # ---------- QoS ----------
        plan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # latch last plan
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ---------- IO ----------
        self.sub_plan = self.create_subscription(
            TrajectoryPlan, trajectory_plan_topic, self._on_plan, plan_qos
        )
        self.sub_odom = self.create_subscription(
            VehicleOdometry, odom_topic, self._on_odom, qos_sensor
        )
                
        self.pub_path   = self.create_publisher(Path,   self.get_parameter('viz_path_topic').get_parameter_value().string_value, 10)
        self.pub_marker = self.create_publisher(Marker, self.get_parameter('viz_marker_topic').get_parameter_value().string_value, 10)

        # timer to continuously re-publish in vehicle frame using latest odom
        self._viz_timer = self.create_timer(
            1.0 / max(self.viz_rate_hz, 1e-3),
            self._viz_tick
        )

        self.plan = None   # latest TrajectoryPlan
        self.get_logger().info(f"TrajectoryVisualizer ready. Subscribed to '{trajectory_plan_topic}'.")

    # ---------- Callbacks ----------
    def _on_plan(self, msg: TrajectoryPlan):
        self.plan = msg
        # self.get_logger().info(f"[Viz] TrajectoryPlan received: type={msg.type}, duration={msg.duration:.3f}, repeat={msg.repeat}")
        # sample ONCE in WORLD frame (your existing sampler that returns Nx3)
        self._plan_xypsi_world = self._sample_plan_xypsi(self.plan)

        if self._plan_xypsi_world is None or self._plan_xypsi_world.shape[0] == 0:
            self.get_logger().warn("[Viz] Plan produced no samples.")
            return

        # publish one snapshot right away
        self._publish_path(self._plan_xypsi_world)   # transforms to vehicle frame inside
        self._publish_marker(self._plan_xypsi_world)


    def _on_odom(self, msg: VehicleOdometry):
        # PX4 gives q = [w, x, y, z]; convert to yaw
        # (same convention you use in mpc_controller)
        from scipy.spatial.transform import Rotation as R
        q = [msg.q[1], msg.q[2], msg.q[3], msg.q[0]]  # x,y,z,w
        eul_zyx = R.from_quat(q).as_euler('ZYX', degrees=False)
        self._psi = float(eul_zyx[0])
        self._pos = np.array(msg.position, dtype=float)
        self._have_odom = True

    def _viz_tick(self):
        # re-publish using the latest odom → path stays anchored to the robot
        if (self._plan_xypsi_world is None) or (not self._have_odom):
            return
        self._publish_path(self._plan_xypsi_world)   # will use current self._pos/self._psi
        self._publish_marker(self._plan_xypsi_world)

    # ---------- Core: Evaluator (mirrors your controller/manager math) ----------
    def _sample_plan_xypsi(self, plan: TrajectoryPlan):
        """
        Returns Nx2 array of [x,y] samples along the plan, using the exact
        primitives your stack already uses.
        """
        # Deserialize meta
        _type = (plan.type or "").lower()
        state0 = np.asarray(plan.state0, dtype=float).reshape(9,) if len(plan.state0) >= 9 else np.zeros(9)
        x0, y0, psi0 = state0[0], state0[1], state0[2]

        duration = float(plan.duration) if np.isfinite(plan.duration) and plan.duration > 0.0 else None
        repeat   = plan.repeat or "none"

        # Note: TrajectoryManager/Generator use:
        #  - min_jerk: coeffs is (3,6)
        #  - arc/straight: speed, yaw_rate, heading
        #  - rounded_rectangle/racetrack_capsule: params array
        coeffs = None
        if len(plan.coeffs) == 18 and _type == "min_jerk":
            coeffs = np.asarray(plan.coeffs, float).reshape(3, 6)

        speed    = float(plan.speed)
        yaw_rate = float(plan.yaw_rate)
        heading  = float(plan.heading)
        params   = np.asarray(plan.coeffs if _type not in ("min_jerk",) else plan.params, float)
        if params.size == 0 and hasattr(plan, "params"):
            params = np.asarray(plan.params, float)

        # Decide total time to draw
        Tdraw = None
        if duration is not None:
            Tdraw = duration
        else:
            # For loops / unbounded, choose a sensible window
            if _type == "arc" and abs(yaw_rate) > 1e-6:
                Tdraw = min(self.max_seconds, (2.0 * math.pi) / abs(yaw_rate))
            elif _type in ("straight", "rounded_rectangle", "racetrack_capsule"):
                Tdraw = self.max_seconds
            else:
                Tdraw = self.max_seconds

        N = max(2, self.samples)
        ts = np.linspace(0.0, Tdraw if Tdraw > 0 else self.resample_dt, N)

        xs, ys, psis = [], [], []
        for tau in ts:
            p, v, a = self._eval_one(_type, state0, coeffs, speed, yaw_rate, heading, params, tau)
            if p is None:
                continue
            xs.append(float(p[0])); ys.append(float(p[1])); psis.append(float(p[2]))
        return np.column_stack([xs, ys, psis]) if xs else None

    def _eval_one(self, typ, st0, C, speed, yaw_rate, heading, P, tau):
        """
        Mirror of your evaluators:
          - min_jerk  (quintic per axis)
          - arc       (constant-twist, straight when |w|≈0)
          - straight  (const heading/speed)
          - rounded_rectangle / racetrack_capsule via piecewise straight+arc
        """
        x0, y0, psi0 = st0[0], st0[1], st0[2]

        if typ == "min_jerk" and C is not None:
            def mj_eval(c, t):
                a0,a1,a2,a3,a4,a5 = c
                t2,t3,t4,t5 = t*t, t*t*t, t*t*t*t, t*t*t*t*t
                p = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
                v = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4
                a = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3
                return p, v, a
            px,_,_ = mj_eval(C[0], tau)
            py,_,_ = mj_eval(C[1], tau)
            ppsi,_,_ = mj_eval(C[2], tau)
            return np.array([px, py, ppsi]), None, None

        elif typ == "arc":
            v = float(speed); w = float(yaw_rate)
            if abs(w) < 1e-6:
                cpsi, spsi = math.cos(psi0), math.sin(psi0)
                x = x0 + v * tau * cpsi
                y = y0 + v * tau * spsi
                psi = psi0
                return np.array([x,y,psi]), None, None
            R = v / w
            psi = psi0 + w * tau
            x = x0 + R * (math.sin(psi) - math.sin(psi0))
            y = y0 - R * (math.cos(psi) - math.cos(psi0))
            return np.array([x,y,psi]), None, None

        elif typ == "straight":
            v = float(speed); psi = float(heading) if not np.isnan(heading) else float(psi0)
            c,s = math.cos(psi), math.sin(psi)
            x = x0 + v * tau * c
            y = y0 + v * tau * s
            return np.array([x,y,psi]), None, None

        elif typ in ("rounded_rectangle", "racetrack_capsule"):
            # Build the same piecewise track your generator makes
            # rounded_rectangle params: [width, height, corner_radius, cw?]
            # racetrack_capsule params: [straight_length, radius, cw?]
            segs = []
            if typ == "rounded_rectangle":
                if P.size < 3:
                    return None, None, None
                W, H, r = float(P[0]), float(P[1]), float(P[2])
                turn = (+1.0 if (P.size >= 4 and float(P[3]) > 0) else -1.0)
                v = float(speed)
                Lx, Ly = W - 2*r, H - 2*r
                segs = [
                    ("straight", Lx, 0.0, v),
                    ("arc",      r,  turn*math.pi/2, v),
                    ("straight", Ly, 0.0, v),
                    ("arc",      r,  turn*math.pi/2, v),
                    ("straight", Lx, 0.0, v),
                    ("arc",      r,  turn*math.pi/2, v),
                    ("straight", Ly, 0.0, v),
                    ("arc",      r,  turn*math.pi/2, v),
                ]
            else:  # racetrack_capsule
                if P.size < 2:
                    return None, None, None
                L, r = float(P[0]), float(P[1])
                turn = (+1.0 if (P.size >= 3 and float(P[2]) > 0) else -1.0)
                v = float(speed)
                segs = [
                    ("straight", L, 0.0, v),
                    ("arc",      r,  turn*math.pi, v),
                    ("straight", L, 0.0, v),
                    ("arc",      r,  turn*math.pi, v),
                ]

            # Walk segments until reaching tau
            xk, yk, psik = x0, y0, psi0
            accT = 0.0
            for kind, p1, p2, v in segs:
                if kind == "straight":
                    L = p1; T = abs(L / max(abs(v), 1e-9))
                    if tau <= accT + T:
                        tt = tau - accT
                        cpsi, spsi = math.cos(psik), math.sin(psik)
                        x = xk + v * tt * cpsi
                        y = yk + v * tt * spsi
                        return np.array([x, y, psik]), None, None
                    # advance
                    xk += L * math.cos(psik)
                    yk += L * math.sin(psik)
                    accT += T
                else:  # arc
                    r = p1; theta = p2
                    w = (v / r) * (1.0 if theta >= 0 else -1.0)
                    T = abs(theta) / max(abs(w), 1e-9)
                    if tau <= accT + T:
                        tt = tau - accT
                        psi = psik + (math.copysign(1.0, theta) * abs(w) * tt)
                        Rloc = v / (math.copysign(1.0, theta) * abs(w))
                        x = xk + Rloc * (math.sin(psi) - math.sin(psik))
                        y = yk - Rloc * (math.cos(psi) - math.cos(psik))
                        return np.array([x, y, psi]), None, None
                    # advance to end of arc
                    psi_end = psik + theta
                    xk = xk + (v/abs(w)) * (math.sin(psi_end) - math.sin(psik)) * math.copysign(1.0, theta)
                    yk = yk - (v/abs(w)) * (math.cos(psi_end) - math.cos(psik)) * math.copysign(1.0, theta)
                    psik = psi_end
                    accT += T

            # Fell through due to float; return end of last segment
            return np.array([xk, yk, psik]), None, None

        else:
            return None, None, None

    # ---------- Publishers ----------
    def _publish_path(self, xypsi_world: np.ndarray):
        """
        Publish the path in the VEHICLE frame (origin at robot).
        Input is Nx3 in WORLD frame; we transform to vehicle frame here.
        """
        if not self._have_odom:
            self.get_logger().warn("[Viz] No odometry yet; skipping path publish.")
            return

        xypsi = self._world_to_vehicle_xypsi(xypsi_world)

        msg = Path()
        msg.header.frame_id = self.vehicle_frame     # <-- vehicle frame (e.g., base_link)
        msg.header.stamp = self.get_clock().now().to_msg()

        for i in range(xypsi.shape[0]):
            x = float(xypsi[i, 0])
            y = float(xypsi[i, 1])
            psi = float(xypsi[i, 2])

            ps = PoseStamped()
            ps.header.frame_id = self.vehicle_frame
            ps.header.stamp = msg.header.stamp

            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0

            # yaw -> quaternion (about +Z)
            half = 0.5 * psi
            ps.pose.orientation.x = 0.0
            ps.pose.orientation.y = 0.0
            ps.pose.orientation.z = math.sin(half)
            ps.pose.orientation.w = math.cos(half)

            msg.poses.append(ps)

        self.pub_path.publish(msg)
        # self.get_logger().info(
        #     f"[Viz] Published VEHICLE-frame Path(+ψ) with {len(msg.poses)} poses on {self.pub_path.topic}"
        # )

    def _publish_marker(self, xypsi_world: np.ndarray):
        """
        Publish a line strip in the VEHICLE frame for RViz XY overlay.
        """
        if not self._have_odom:
            return
        xypsi = self._world_to_vehicle_xypsi(xypsi_world)

        m = Marker()
        m.header.frame_id = self.vehicle_frame       # <-- vehicle frame (e.g., base_link)
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "traj_xy_vehicle"
        m.id = 1
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.03
        m.color.a = 1.0; m.color.r = 0.0; m.color.g = 1.0; m.color.b = 0.0
        m.pose.orientation.w = 1.0

        from geometry_msgs.msg import Point
        for i in range(xypsi.shape[0]):
            p = Point()
            p.x = float(xypsi[i, 0])
            p.y = float(xypsi[i, 1])
            p.z = 0.0
            m.points.append(p)

        self.pub_marker.publish(m)


    @staticmethod
    def _wrap(a: float) -> float:
        # [-pi, pi]
        return (a + math.pi) % (2.0*math.pi) - math.pi

    def _world_to_vehicle_xypsi(self, xypsi_world: np.ndarray) -> np.ndarray:
        """
        Transform an Nx3 array [x_w, y_w, psi_w] from world -> vehicle frame
        using current odometry (position, yaw). Result is [x_v, y_v, psi_v].
        """
        if xypsi_world is None or xypsi_world.shape[1] < 3:
            return xypsi_world

        dx = xypsi_world[:, 0] - self._pos[0]
        dy = xypsi_world[:, 1] - self._pos[1]
        c, s = math.cos(self._psi), math.sin(self._psi)

        # R_world_from_vehicle = [[c,-s],[s,c]]
        # vehicle = R^T * (world - pos) = [[c,s],[-s,c]] @ [dx, dy]
        xv =  c*dx + s*dy
        yv = -s*dx + c*dy

        psi_rel = (xypsi_world[:, 2] - self._psi)
        psi_rel = np.vectorize(self._wrap)(psi_rel)

        if self.flip_y:
            yv = -yv

        return np.column_stack([xv, yv, psi_rel])

def main():
    rclpy.init()
    node = TrajectoryVisualizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
