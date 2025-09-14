import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np


from std_msgs.msg import Float32MultiArray, UInt8
from px4_msgs.msg import VehicleOdometry, TimesyncStatus, TrajectorySetpoint6dof
from obstacle_detector.msg import Obstacles
from custom_offboard_msgs.msg import TrajectoryPlan

from asl_rover_offboard.utils.mpc_solver import MPCSolver

from asl_rover_offboard.utils.param_loader import ParamLoader

from ament_index_python.packages import get_package_share_directory
import os

import signal

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        
        self.declare_parameter('vehicle_param_file', 'asl_rover_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('controller_param_file', 'mpc_controller_asl_rover.yaml')
        self.declare_parameter('world_frame', 'map')  # or 'odom'
        self.declare_parameter('mpc_trajectory_topic', '/trajectory') 

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        controller_param_file = self.get_parameter('controller_param_file').get_parameter_value().string_value
        self.world_frame = self.get_parameter('world_frame').get_parameter_value().string_value
        mpc_trajectory_topic = self.get_parameter('mpc_trajectory_topic').get_parameter_value().string_value

        package_dir = get_package_share_directory('asl_rover_offboard')
        
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)
        controller_yaml_path = os.path.join(package_dir, 'config', 'controller', controller_param_file)


        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)
        controller_yaml = ParamLoader(controller_yaml_path)

        # Topic names
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        timesync_topic = sitl_yaml.get_topic("status_topic")
        control_cmd_topic = sitl_yaml.get_topic("control_command_topic")
        trajectory_sub_topic = sitl_yaml.get_topic("command_traj_topic")
        trajectory_plan_topic = sitl_yaml.get_topic("trajectory_plan_topic")
        obstacle_topic = sitl_yaml.get_topic("obstacle_topic")
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")

        # Controller parameters
        self.control_params = controller_yaml.get_control_params()
        # Vehicle parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()



        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        obs_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        plan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- ask for the stored last sample
        )

        # Sub
        self.sub_odom = self.create_subscription(
            VehicleOdometry, 
            odom_topic, 
            self._odom_callback, qos_profile)
        
        self.sub_sync = self.create_subscription(
            TimesyncStatus, 
            timesync_topic, 
            self._sync_callback, qos_profile)
        
        self.sub_commanded_trajectory = self.create_subscription(
            TrajectorySetpoint6dof, 
            trajectory_sub_topic, 
            self._trajectory_callback, 10)
        
        self.sub_obs = self.create_subscription(
            Obstacles, 
            obstacle_topic, 
            self._obstacles_callback, obs_qos)
        
        self.sub_plan = self.create_subscription(
            TrajectoryPlan,
            trajectory_plan_topic,
            self._on_trajectory_plan, plan_qos)
        
        self.sub_nav_state = self.create_subscription(
            UInt8, 
            nav_state_topic, 
            self._on_nav_state, plan_qos)


        # Pub
        self.motor_cmd_pub = self.create_publisher(
            Float32MultiArray, 
            control_cmd_topic, 
            10)
        
        self.trajectory_pub = self.create_publisher(
            Path, 
            mpc_trajectory_topic, 
            10)
                        

        # State variables
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp = 0
        self._odom_ready = False
        self._trajectory_ready = False

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.zeros(4)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3) # Angular velocity in body frame


        self.plan_type = None
        self.plan_t0_us = None
        self.plan_data = {}     # dict: coeffs/state0/speed/yaw_rate/heading/duration/repeat/distance
 
        self.nav_state = 1  # IDLE by default

        # --- obstacle state ---
        self.obstacles_world = []     # list of {'center': np.array([x,y]), 'radius': float}
        self.obs_center = None        # np.array([x,y]) of nearest obstacle (world frame)
        self.obs_radius = None        # float

        # MPC predicted path 
        self.last_pred_X = None     # np.ndarray, shape (NX, N+1), world frame
        self.last_ref_X  = None     # np.ndarray, shape (NX, N+1), world frame

        # Trajectory 
        self._x_ref = None

        self.u_prev = np.array([0.0, 0.0])


        self.Ts = 1.0 / self.control_params.frequency

        self.mpc = MPCSolver(self.control_params, self.vehicle_params, debug=False)
        self.timer_mpc = self.create_timer(self.Ts, self._control_loop)  

        self.get_logger().info("Controller Node initialized.")


    # ---------- callbacks ----------
    def _odom_callback(self, msg):
        self.px4_timestamp = msg.timestamp
        t_now = self._timing(msg.timestamp)
        self.t_sim = t_now

        # Position
        self.pos = np.array(msg.position)

        # Attitude (Quaternion to Euler)
        self.q[0],self.q[1],self.q[2],self.q[3] = msg.q
        self.rpy[2],self.rpy[1],self.rpy[0] = self._quat_to_eul(self.q)
        
        #rot_mat = eul2rotm_py(self.rpy)
        
        # World-frame Velocity (transform from body frame)
        self.vel = np.array(msg.velocity)

        # Body-frame Angular Velocity
        self.omega_body = np.array(msg.angular_velocity)

        #self.get_logger().info(f"self.rpy: {self.rpy}")

        if not self._odom_ready:
            self.get_logger().warn("First odometry message is received...")
            self._odom_ready = True
            return
        
        self._odom_ready = True

    def _trajectory_callback(self, msg):
        return
    
    def _on_nav_state(self, msg: UInt8):
        self.nav_state = int(msg.data)
        if self.nav_state != 2:          # 2 = MISSION
            self._trajectory_ready = False
            self.u_prev = np.array([0.0, 0.0])
    
    def _obstacles_callback(self, msg: Obstacles):
        obs_list = []
    
        # Circles (preferred if available)
        if hasattr(msg, 'circles'):
            for c in msg.circles:
                cx, cy = float(c.center.x), float(-c.center.y)
                r = float(getattr(c, 'radius', 0.0))
                center = np.array([cx, cy], dtype=float)
                world = self._local_to_world(center)
                obs_list.append({'center': world, 'radius': r})

        # Segments (approximate each as a circle at the midpoint, radius ≈ half length)
        if hasattr(msg, 'segments'):
            for s in msg.segments:
                x1, y1 = float(s.first_point.x), float(-s.first_point.y)
                x2, y2 = float(s.last_point.x),  float(-s.last_point.y)
                mid = np.array([(x1 + x2)/2.0, (y1 + y2)/2.0], dtype=float)
                length = np.linalg.norm( np.array([x2 - x1, y2 - y1]) )
                r = 0.5 * length
                world = self._local_to_world(mid) 
                obs_list.append({'center': world, 'radius': r})

        self.obstacles_world = obs_list

        #self.get_logger().info(f"obs_list= {obs_list}")

        if obs_list:
            # Pick the nearest obstacle to the current robot position
            dists = [np.linalg.norm(o['center'] - self.pos[:2]) for o in obs_list]
            i = int(np.argmin(dists))
            self.obs_center = obs_list[i]['center']
            self.obs_radius = obs_list[i]['radius']
            # self.get_logger().debug(f"Nearest obstacle @ {self.obs_center}, r={self.obs_radius:.2f}")
        else:
            self.obs_center = None
            self.obs_radius = None
    
    def _on_trajectory_plan(self, msg: TrajectoryPlan):
        self.plan_type = msg.type
        self.plan_t0_us = int(msg.t0_us)

        raw = np.asarray(msg.coeffs, dtype=float)
        coeffs_mat, params = None, None

        _type = (msg.type or "").lower()
        if _type == "min_jerk":
            if raw.size == 18:
                coeffs_mat = raw.reshape(3, 6)
            elif raw.size == 0 or np.all(np.isnan(raw)):
                coeffs_mat = None
            else:
                self.get_logger().warn(f"TrajectoryPlan.coeffs has size {raw.size}, expected 18 for min_jerk; ignoring.")
                coeffs_mat = None
        else:
            # For other plan types, treat coeffs as generic parameters
            params = raw  # may be empty
                
        self.plan_data = {
            "duration": float(msg.duration),
            "repeat": msg.repeat,
            "state0": np.array(msg.state0, dtype=float),
            "coeffs": coeffs_mat,    # only used for min_jerk
            "params": params,        # used by new shapes
            "speed": float(msg.speed),
            "yaw_rate": float(msg.yaw_rate),
            "heading": float(msg.heading),
            "distance": float(msg.distance),
        }
        self._trajectory_ready = True
        self.get_logger().info(f"TrajectoryPlan received: type={self.plan_type}, data={self.plan_data}")

    
    def _sync_callback(self, msg):
        self.px4_timestamp = msg.timestamp

    

    # ---------- main loop ----------
    def _control_loop(self):
        """
        Runs at a slower rate, solving the optimization problem.
        """
        if not self._odom_ready:
            self.get_logger().warn("Waiting for odometry...")
            return
        
        # Construct the 3-dimensional state vector
        # v_curr = np.linalg.norm(self.vel[:2])  # magnitude of [vx, vy]
        _x0 = np.array([self.pos[0], self.pos[1], self.rpy[2]])

        if not self._trajectory_ready and self.nav_state != 2:
            # publish hold/zero command and bail
            msg = Float32MultiArray(); 
            msg.data = [0.0, 0.0]
            self.motor_cmd_pub.publish(msg)
            return
        
        xref_h, obs_h = self._build_horizon(_x0)
        ok, _u_mpc, _X_opt, _ = self.mpc.solve(_x0, self.u_prev, xref_h, obs_h, r_safe=1.0)
        #ok, u0, _, _ = self.mpc.solve(x0, xref_h, obs_h, r_safe=2.0)

        # self.get_logger().info(f"ok: {ok} | x0= {_x0} | u0= {_u_mpc} | xref= {xref_h} ")
        # _u_mpc = np.array([1.0, 0.0])
        self.get_logger().info(f"obs center: {obs_h} ")
      
        self.u_prev = _u_mpc

        #X_opt = xref_h
        self.last_ref_X  = xref_h

        # Current pose in world
        p0_w = _x0[0:2]           # shape (2,)
        psi0 = float(_x0[2])

        # Vectorized world->body transform for all horizon points
        p_w = _X_opt[0:2, :]      # (2, N+1)
        dp_w = p_w - p0_w.reshape(2, 1)    # translate so body origin is at (0,0)

        c, s = np.cos(psi0), np.sin(psi0)
        R_w2b = np.array([[ c,  s],
                        [-s,  c]])       # rotates world coords into body frame

        p_b = R_w2b @ dp_w                 # (2, N+1)

        # Build body-frame copy
        X_body = _X_opt.copy()
        X_body[0:2, :] = p_b

        # Yaw relative to body (wrap to [-pi, pi])
        def wrap(a):
            return (a + np.pi) % (2*np.pi) - np.pi

        X_body[2, :] = wrap(_X_opt[2, :] - psi0)

        # Optional: transform velocity components if your state stores world vx,vy
        # If your state has speed v along heading already, no change needed.

        self.last_pred_X = X_body


        #self.get_logger().info(f"u0= {u0} | {self.control_params.frequency}")


        #u0 = np.array([0.0,0.0])

        #self.get_logger().info(f"x_ref= {pos_ref[0]} | diff= {pos_ref[0] - x0[0]}")
        #self.get_logger().info(f"y_ref= {pos_ref[1]} | diff= {pos_ref[1] - x0[1]}")
        # self.get_logger().info(f"v_ref= {p_ref} | diff= {np.array(v_ref) - self.vel}")

        # self.get_logger().info(f"roll= {self.rpy[0]*180/np.pi} | diff= {(0.0 - self.rpy[0])*180/np.pi}")
        # self.get_logger().info(f"pitch= {self.rpy[1]*180/np.pi} | diff= {(0.0 - self.rpy[1])*180/np.pi}")
        #self.get_logger().info(f"yaw_cmd= {psi_ref*180/np.pi} | diff= {(psi_ref - self.rpy[2])*180/np.pi}")
        

        # Publish
        msg = Float32MultiArray()
        msg.data = _u_mpc.tolist()
        self.motor_cmd_pub.publish(msg)

   
        self._publish_trajectory()

        #self.get_logger().info(f"t={self.t_sim:.2f} | pos={self.pos.round(2)} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"t={self.t_sim:.2f} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"x={x0}")
        # self.get_logger().info(f"x_ref={x_ref}")
        # self.get_logger().info(f"x_dif={x_ref-x0}")



    # ---------- helpers ----------
    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0
    
    def _reset_t0(self, t_now):
        self.t0 = t_now

    def _quat_to_eul(self, q_xyzw):
        # PX4: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        _r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
        return _r.as_euler('ZYX', degrees=False)

    def _local_to_world(self, xy: np.ndarray) -> np.ndarray:
        """Transform (x,y) from robot frame -> world using current pose/yaw."""
        yaw = float(self.rpy[2])
        c, s = np.cos(yaw), np.sin(yaw)
        xw = self.pos[0] + c*xy[0] - s*xy[1]
        yw = self.pos[1] + s*xy[0] + c*xy[1]
        return np.array([xw, yw], dtype=float)

    def angle_diff(self, a, b):
        """
        Compute the wrapped difference between two angles in radians.
        Both a and b can be scalars or numpy arrays.
        Returns value in [-pi, pi]
        """
        return (a - b + np.pi) % (2 * np.pi) - np.pi
    
    def _build_horizon(self,x0):
        """
        Build xref_h (3, N+1) from the analytic trajectory and an optional obs_h (2, N+1).
        Uses current sim time self.t_sim and controller frequency from params.
        """
        
        N  = self.mpc.N

        # Sample trajectory over horizon
        t0_abs = self.px4_timestamp * 1e-6
        t_samples_abs = t0_abs + self.Ts * np.arange(N + 1)
        psi_prev = float(x0[2])          # anchor / fallback
        v_eps = 1e-3

        xs, ys, psis_raw = [], [], []
        for tk_abs in t_samples_abs:
            out = self._eval_plan_abs(tk_abs)
            if out[0] is None:
                # no plan yet → hold
                pos_ref = self.pos.copy()
                vel_ref = np.zeros(3)
            else:
                pos_ref, vel_ref, _ = out

            vx, vy = vel_ref[0], vel_ref[1]
            # heading from velocity; hold previous if almost stopped
            if vx*vx + vy*vy > v_eps*v_eps:
                psi_ref = np.arctan2(vy, vx)
                psi_prev = psi_ref
            else:
                psi_ref = psi_prev

            xs.append(pos_ref[0])
            ys.append(pos_ref[1])
            psis_raw.append(psi_ref)

        
        # unwrap relative to current yaw so it's continuous across ±π
        psi0 = float(x0[2])
        psis_raw = np.asarray(psis_raw, dtype=float)
        psis = np.unwrap(np.r_[psi0, psis_raw])[1:]   # keep first value aligned to psi0

        xref_h = np.vstack([np.array(xs), np.array(ys), np.array(psis)])  # (3, N+1)

        #self.get_logger().info(f"psi0= {psi0} | psis_raw= {psis_raw}")

        #self.get_logger().info(f"Ts= {self.Ts} | {self.control_params.frequency}")

        # --- obstacle horizon (optional) ---
        obs_h = None
        if self.obs_center is not None:
            dist = np.linalg.norm(xref_h[:2, 0] - self.obs_center)  # distance from current ref start
            # keep it simple: only enable if close enough
            if dist <= 8.0:
                obs_h = np.tile(self.obs_center.reshape(2, 1), (1, N + 1))

        return xref_h, obs_h
    
    # --- local evaluator for one time sample (absolute PX4 time, in seconds) ---
    def _eval_plan_abs(self, t_abs_s: float):
        if not self._trajectory_ready or self.plan_t0_us is None:
            return None, None, None
        tau = max(0.0, t_abs_s - (self.plan_t0_us * 1e-6))

        _type = (self.plan_type or "").lower()
        meta = self.plan_data
        st0 = meta["state0"]  # [x,y,psi, xd,yd,psid, xdd,ydd,psidd]
        x0, y0, psi0 = st0[0], st0[1], st0[2]

        if _type == "min_jerk" and meta["coeffs"] is not None:
            C = meta["coeffs"]  # (3,6)
            def mj_eval(c, t):
                a0,a1,a2,a3,a4,a5 = c
                t2,t3,t4,t5 = t*t, t*t*t, t*t*t*t, t*t*t*t*t
                p = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
                v = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4
                a = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3
                return p, v, a
            px,vx,ax = mj_eval(C[0], tau)
            py,vy,ay = mj_eval(C[1], tau)
            ppsi,vpsi,apsi = mj_eval(C[2], tau)
            p = np.array([px, py, ppsi]); v = np.array([vx, vy, vpsi]); a = np.array([ax, ay, apsi])
            return p, v, a

        elif _type == "arc":
            v = meta["speed"]; w = meta["yaw_rate"]
            if abs(w) < 1e-6:
                # straight as arc limit
                c,s = np.cos(psi0), np.sin(psi0)
                x = x0 + v*tau*c; y = y0 + v*tau*s; psi = psi0
                xd,yd,psid = v*c, v*s, 0.0
                xdd,ydd,psidd = 0.0, 0.0, 0.0
                return np.array([x,y,psi]), np.array([xd,yd,psid]), np.array([xdd,ydd,psidd])
            R = v / w
            psi = psi0 + w*tau
            s0,c0 = np.sin(psi0), np.cos(psi0)
            s,c = np.sin(psi), np.cos(psi)
            x = x0 + R*(s - s0)
            y = y0 - R*(c - c0)
            xd,yd,psid = v*c, v*s, w
            xdd,ydd,psidd = -v*w*s, v*w*c, 0.0
            return np.array([x,y,psi]), np.array([xd,yd,psid]), np.array([xdd,ydd,psidd])

        elif _type == "straight":
            v = meta["speed"]; psi = meta["heading"]
            c,s = np.cos(psi), np.sin(psi)
            x = x0 + v*tau*c; y = y0 + v*tau*s
            return np.array([x,y,psi]), np.array([v*c, v*s, 0.0]), np.array([0.0,0.0,0.0])

        elif _type == "rounded_rectangle":
            # params = [width, height, corner_radius, turn_sign(optional)]
            P = meta.get("params", None)
            if P is None or P.size < 3:
                return None, None, None
            W, H, r = float(P[0]), float(P[1]), float(P[2])
            turn = float(P[3]) if P.size >= 4 else +1.0  # -1=CCW, +1=CW
            v = float(meta["speed"])
            if W <= 2*r or H <= 2*r or r <= 0.0 or abs(v) < 1e-9:
                return None, None, None

            Lx, Ly = W - 2*r, H - 2*r
            # segment specs: (type, length_or_radius, angle_if_arc)
            segs = [
                ("straight", Lx, 0.0),
                ("arc",      r,  turn*np.pi/2),
                ("straight", Ly, 0.0),
                ("arc",      r,  turn*np.pi/2),
                ("straight", Lx, 0.0),
                ("arc",      r,  turn*np.pi/2),
                ("straight", Ly, 0.0),
                ("arc",      r,  turn*np.pi/2),
            ]
            return self._eval_piecewise(segs, v, x0, y0, psi0, tau, meta.get("repeat", "loop"))

        elif _type == "racetrack_capsule":
            # params = [straight_length, radius, turn_sign(optional)]
            P = meta.get("params", None)
            if P is None or P.size < 2:
                return None, None, None
            L = float(P[0]); r = float(P[1])
            turn = float(P[2]) if P.size >= 3 else +1.0
            v = float(meta["speed"])
            if r <= 0.0 or L < 0.0 or abs(v) < 1e-9:
                return None, None, None

            segs = [
                ("straight", L, 0.0),
                ("arc",      r, turn*np.pi),
                ("straight", L, 0.0),
                ("arc",      r, turn*np.pi),
            ]
            return self._eval_piecewise(segs, v, x0, y0, psi0, tau, meta.get("repeat", "loop"))


        # fallback
        return None, None, None
    
    def _eval_piecewise(self, segs, v, x0, y0, psi0, tau, repeat_mode="loop"):
        """
        segs: list of tuples
        - ("straight", L, 0.0)
        - ("arc",      r, theta)  # theta signed; >0 CCW, <0 CW
        Constant speed v (>0 assumed along forward heading).
        """
        if abs(v) < 1e-9:
            p = np.array([x0, y0, psi0]); z = np.zeros(3)
            return p, z, z

        # Precompute durations and cumulative times
        T_list = []
        for kind, a, b in segs:
            if kind == "straight":
                T_list.append(abs(a) / abs(v))
            else:  # arc
                r = float(a); theta = float(b)
                if r <= 0.0 or abs(theta) < 1e-9:
                    T_list.append(0.0)
                else:
                    w = abs(v) / r  # yaw rate magnitude
                    T_list.append(abs(theta) / w)

        T_tot = sum(T_list)
        if T_tot <= 0.0:
            p = np.array([x0, y0, psi0]); z = np.zeros(3)
            return p, z, z

        # repeat/hold behavior
        if isinstance(repeat_mode, str) and repeat_mode.lower() == "loop":
            t_eff = tau % T_tot
        else:
            t_eff = min(max(0.0, tau), T_tot)

        # Walk segments to find the active one and evaluate
        xk, yk, psik = float(x0), float(y0), float(psi0)
        acc = 0.0
        for (kind, a, b), T in zip(segs, T_list):
            if t_eff <= acc + T or np.isclose(t_eff, acc + T):
                dt = t_eff - acc
                if kind == "straight":
                    c, s = np.cos(psik), np.sin(psik)
                    x = xk + v*dt*c
                    y = yk + v*dt*s
                    psi = psik
                    xd, yd, psid = v*c, v*s, 0.0
                    xdd, ydd, psidd = 0.0, 0.0, 0.0
                    return np.array([x, y, psi]), np.array([xd, yd, psid]), np.array([xdd, ydd, psidd])
                else:  # arc
                    r, theta = float(a), float(b)
                    # signed yaw rate to hit theta in time T
                    w = np.sign(theta) * (abs(v) / r)
                    psi = psik + w*dt
                    s0, c0 = np.sin(psik), np.cos(psik)
                    s1, c1 = np.sin(psi),  np.cos(psi)
                    Rloc = v / w  # signed radius: v/w carries sign of w
                    x = xk + Rloc * (s1 - s0)
                    y = yk - Rloc * (c1 - c0)
                    xd, yd, psid = v*c1, v*s1, w
                    xdd, ydd, psidd = -v*w*s1, v*w*c1, 0.0
                    return np.array([x, y, psi]), np.array([xd, yd, psid]), np.array([xdd, ydd, psidd])
            # advance pose to the end of this segment and continue
            if T > 0.0:
                if kind == "straight":
                    c, s = np.cos(psik), np.sin(psik)
                    xk += v*T*c
                    yk += v*T*s
                    # psi unchanged
                else:
                    r, theta = float(a), float(b)
                    w = np.sign(theta) * (abs(v) / r)
                    psi_end = psik + w*T
                    s0, c0 = np.sin(psik), np.cos(psik)
                    s1, c1 = np.sin(psi_end), np.cos(psi_end)
                    Rloc = v / w
                    xk += Rloc * (s1 - s0)
                    yk -= Rloc * (c1 - c0)
                    psik = psi_end
            acc += T

        # Numerical fall-through: return end of last segment
        p = np.array([xk, yk, psik]); z = np.zeros(3)
        return p, z, z

    def _publish_trajectory(self):
        # Decide what to draw: prefer predicted trajectory, else reference
        X = self.last_pred_X if self.last_pred_X is not None else self.last_ref_X
        if X is None:
            return

        # Expect state layout: [x, y, psi, ...] (adjust indices if different)
        x_idx, y_idx, yaw_idx = 0, 1, 2
        xs = X[x_idx, :]
        ys = -X[y_idx, :]
        yaws = -X[yaw_idx, :] if X.shape[0] > yaw_idx else np.zeros_like(xs)

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.world_frame

        poses = []
        for x, y, yaw in zip(xs, ys, yaws):
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x = float(x)
            p.pose.position.y = float(y)
            p.pose.position.z = 0.0

            # yaw -> quaternion (Z-up, ENU)
            cy = np.cos(yaw * 0.5)
            sy = np.sin(yaw * 0.5)
            p.pose.orientation.x = 0.0
            p.pose.orientation.y = 0.0
            p.pose.orientation.z = sy
            p.pose.orientation.w = cy

            poses.append(p)

        msg.poses = poses
        self.trajectory_pub.publish(msg)



def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()

    def signal_handler(sig, frame):
        node.get_logger().info("Shutdown signal received. Cleaning up...")
        # node.logger.plot_logs()
        node.destroy_node()
        rclpy.shutdown()

    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
