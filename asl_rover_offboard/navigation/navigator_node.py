# asl_rover_offboard/navigation/navigator_node.py
from __future__ import annotations
import os, math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from typing import Optional

from px4_msgs.msg import VehicleOdometry, VehicleStatus, VehicleCommandAck
from std_srvs.srv import Trigger
from std_msgs.msg import UInt8

from custom_offboard_msgs.msg import TrajectoryPlan
from ament_index_python.packages import get_package_share_directory

from asl_rover_offboard.utils.param_loader import ParamLoader
from asl_rover_offboard.navigation.state_machine import NavState, NavStateMachine, NavEvents
from asl_rover_offboard.guidance.trajectory_manager import TrajectoryManager, TrajMsg

from asl_rover_offboard.utils.param_types import (
    LineTo, Straight, Arc, RoundedRectangle, RacetrackCapsule
)

from asl_rover_offboard.utils.helper_functions import quat_to_eul

class NavigatorNode(Node):
    def __init__(self):
        super().__init__("navigator")

        # ------- params -------
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')
        self.declare_parameter('mission_param_file', 'mission.yaml')
        self.declare_parameter('command_traj_topic', '/navigator/trajectory_setpoint')
        self.declare_parameter('control_frequency', 50.0)
        self.declare_parameter('auto_start', False)

        package_dir = get_package_share_directory('asl_rover_offboard')

        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        mission_param_file = self.get_parameter('mission_param_file').get_parameter_value().string_value

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        mission_yaml_path = os.path.join(package_dir, 'config', 'mission', mission_param_file)

        sitl_yaml = ParamLoader(sitl_yaml_path)
        mission_yaml = ParamLoader(mission_yaml_path)

        self.freq = float(self.get_parameter('control_frequency').get_parameter_value().double_value)
        self.Ts = 1.0 / self.freq
        self.auto_start = bool(self.get_parameter('auto_start').get_parameter_value().bool_value)
        
        # SITL topics
        traj_topic = sitl_yaml.get_topic("command_traj_topic")
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        status_topic = sitl_yaml.get_topic("status_topic")
        command_ack_topic = sitl_yaml.get_topic("vehicle_command_ack_topic")
        trajectory_plan_topic = sitl_yaml.get_topic("trajectory_plan_topic")
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")

        
        # Mission config
        self.mission = mission_yaml.get_mission()
        ok, msg = mission_yaml.validate_mission()
        self.mission_valid = bool(ok)
        if not ok:
            self.get_logger().warn(f"[Navigator] Mission invalid: {msg}")
        else:
            self.get_logger().info(f"[Navigator] Mission valid: {msg}")

        # ------- components -------
        self.sm = NavStateMachine()
        self.tm = TrajectoryManager(v_max= 1.0, omega_max=1.0, default_T=10000.0)


        # ------- state -------
        self.t0 = None
        self.t_sim = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3)


        self.got_odom = False
        self.nav_offboard = False
        self.armed = False
        self.last_ack_ok = False
        self.start_requested = False  # set by service to allow IDLE->MISSION

        self.last_stamp_us: int | None = None
        
        # mission bookkeeping
        self.plan_created = False
        self.trajectory_fresh = False
        self.at_destination = False
        self.halt_condition = False

        # obstacle avaidance state
        self.in_avoidance = False

        # offboard states
        self.allow_offboard_output = True        # master gate for Offboard setpoints
        self.in_offboard_last = None             # for logging transitions (optional)
        self.suppress_plan_output = False
        self.allow_offboard_output = True

        # manual states
        self.manual_requested = False
        self._manual_true_ticks = 0
        self._manual_false_ticks = 0


        # ------- IO -------
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        plan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- latch last message
        )

        traj_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST, 
            depth=1
        )

        self.create_subscription(VehicleOdometry, odom_topic, self._odom_cb, qos)
        self.create_subscription(VehicleStatus, status_topic, self._on_status, qos_sensor)
        self.create_subscription(VehicleCommandAck, command_ack_topic, self._on_ack, 10)


        # publisher lives in TrajectoryManager.publish_traj(); 
        self.traj_pub = self.create_publisher(TrajMsg, traj_topic, traj_qos)
        self.plan_pub = self.create_publisher(TrajectoryPlan, trajectory_plan_topic, plan_qos)
        self.nav_state_pub = self.create_publisher(UInt8, nav_state_topic, plan_qos)
        
        # Services
        self.start_srv = self.create_service(Trigger, 'navigator/start_mission', self._srv_start_mission)
        self.halt_srv  = self.create_service(Trigger, 'navigator/halt_mission',  self._srv_halt_mission)

        # timers
        self.timer = self.create_timer(self.Ts, self._tick)

        # initial publishes
        self._publish_nav_state()

        # Constructor logs
        self.get_logger().info("NavigatorNode ready.")

    # ---------- callbacks ----------
    def _odom_cb(self, msg: VehicleOdometry):
        self.t_sim = self._clock_to_t(msg.timestamp)
        self.last_stamp_us = int(msg.timestamp)
        self.pos = np.array(msg.position, float)
        self.vel = np.array(msg.velocity, float)
        self.rpy[0],self.rpy[1],self.rpy[2] = quat_to_eul(msg.q)
        #self.get_logger().info(f"self.rpy: {self.rpy}")

        # Body-frame Angular Velocity
        self.omega_body = np.array(msg.angular_velocity, float)

        if not self.got_odom:
            self.got_odom = True
            self.get_logger().info("First odometry received.")


    def _on_status(self, msg: VehicleStatus):
        self.nav_offboard = (msg.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.armed = (msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        raw_manual = (not self.nav_offboard) and self._is_manual_nav(msg.nav_state)

        if raw_manual:
            self._manual_true_ticks += 1; self._manual_false_ticks = 0
        else:
            self._manual_false_ticks += 1; self._manual_true_ticks = 0
        # require stability for 2 consecutive status messages (~60 ms @ 50 Hz)
        self.manual_requested = (self._manual_true_ticks >= 3)

        # self.get_logger().info(f"manual_requested: {self.manual_requested}")

        if self.nav_offboard != self.in_offboard_last:
            self.get_logger().info(f"[Navigator] FCU mode changed: {'OFFBOARD' if self.nav_offboard else 'NOT-OFFBOARD'}")
            self.in_offboard_last = self.nav_offboard

    def _on_ack(self, ack: VehicleCommandAck):
        # VEHICLE_RESULT_ACCEPTED = 0
        VEHICLE_RESULT_ACCEPTED = 0
        if ack.result == VEHICLE_RESULT_ACCEPTED:
            self.last_ack_ok = True


    # ---------- main loop ----------
    def _tick(self):

        if not self.got_odom:
            self._publish_nav_state()
            return


        # events        
        ev = NavEvents(
            have_odom=bool(self.got_odom),
            auto_start=bool(self.auto_start),
            trajectory_fresh=bool(self.trajectory_fresh),
            at_destination=bool(self.at_destination),
            start_requested=bool(self.start_requested),
            halt_condition=bool(self.halt_condition),
            mission_valid=bool(self.mission_valid),
            manual_requested=bool(self.manual_requested),
            offboard_ok=bool(self.nav_offboard), 
            in_avoidance=bool(self.in_avoidance),
        )
        prev = self.sm.state
        new = self.sm.step(ev)


        # plan on transitions
        if new != prev:
            self._on_state_change(prev, new)




    def _on_state_change(self, prev, state):

        self.get_logger().info(f"State: {prev.name} -> {state.name}")
        self.target_plan_active = False

        if state in (NavState.MISSION, NavState.IDLE, NavState.AVOIDANCE):
            self.allow_offboard_output = True 
            self.suppress_plan_output = False


        if state == NavState.IDLE:
            self._clear_active_plan()
            self.allow_offboard_output = False
            self.suppress_plan_output = True   # tells _plan_or_hold to hover
            self.halt_condition = False 
            
        elif state == NavState.MISSION:
            self._clear_active_plan()
            self._plan_mission()
            self.start_requested = False # clear the flag once you are in
            self.plan_created = True
            self.trajectory_fresh = True
            self.halt_condition = False 

        elif state == NavState.MANUAL:
            self._clear_active_plan()
            self.allow_offboard_output = False
            self.suppress_plan_output = True
            self.halt_condition = False

        self._publish_nav_state()

    # -------------------- planning --------------------
    def _plan_mission(self):
        """Plan according to mission params. Called once when odom first arrives."""
        m = self.mission

        # choose start pose
        if m.common.start.use_current:
            p0 = np.array([self.pos[0], self.pos[1], 0.0])
            heading = float(self.rpy[2])
        else:
            p0 = np.array([m.common.start.x, m.common.start.y, 0.0])
            heading = float(m.common.start.psi)

        if isinstance(m, LineTo):
            self.tm.plan_line_to(self.t_sim, p0, heading, m.goal_xypsi, m.duration, m.common.repeat)
        elif isinstance(m, Straight):
            self.tm.plan_straight(self.t_sim, p0, heading, m.segment_distance, float(m.common.speed or 0.0), m.common.repeat)
        elif isinstance(m, Arc):
            if m.angle is not None:
                self.tm.plan_arc_by_angle(self.t_sim, p0, heading, m.radius, (+1 if m.cw else -1)*abs(m.angle),
                                        float(m.common.speed or 0.0), m.common.repeat)
            else:
                self.tm.plan_arc_by_rate(self.t_sim, p0, heading, m.radius, (+1 if m.cw else -1)*abs(m.yaw_rate),
                                        float(m.common.speed or 0.0), m.common.repeat)
        elif isinstance(m, RoundedRectangle):
            self.tm.plan_rounded_rectangle(self.t_sim, p0, heading, m.width, m.height, m.corner_radius,
                                        float(m.common.speed or 0.0), m.cw, m.common.repeat)
        elif isinstance(m, RacetrackCapsule):
            self.tm.plan_racetrack_capsule(self.t_sim, p0, heading, m.straight_length, m.radius,
                                        float(m.common.speed or 0.0), m.cw, m.common.repeat)
        else:
            self.get_logger().warn(f"Unknown mission.type='{self.mission.mission_type}', staying IDLE.")


        # Publish the descriptor exactly once per plan
        if self.last_stamp_us is not None:
            plan_msg = self.tm.to_plan_msg(t0_us=self.last_stamp_us)
            self.plan_pub.publish(plan_msg)
            self.get_logger().info(f"Published TrajectoryPlan(type={plan_msg.type}, state0={plan_msg.state0})")



    # ---------- services ----------
    def _srv_start_mission(self, req, resp):
        if not self.got_odom:
            resp.success = False
            resp.message = "Cannot start: no odometry yet."
            return resp

        # Always (re)plan now — time-dependent mission
        self._plan_mission()
        self.plan_created = True

        # One-shot trigger to leave IDLE on next tick
        self.trajectory_fresh = True
        self.start_requested = True
        self.halt_condition = False

        resp.success = True
        resp.message = "Mission planned and start requested."
        return resp

    def _srv_halt_mission(self, req, resp):
        # Ask the SM to go back to IDLE; we'll hold position there
        self.halt_condition = True
        # Clear any outstanding start request so we don't immediately re-enter
        self.start_requested = False
        # (optional) mark current plan as consumed
        self.trajectory_fresh = False
        self.auto_start = False            # don’t immediately leave IDLE again
        self.suppress_plan_output = True   # force HOLD outputs
        resp.success = True
        resp.message = "Mission halt requested."
        return resp


    # ---------- utils ----------
    def _clock_to_t(self, stamp_us: int) -> float:
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0
    
    def _arrival_check(self) -> bool:
        # If the generator says “no more plan”, or we are close in pos & slow in vel → at destination
        pos_err = np.linalg.norm(self.vel[:0])  # placeholder to avoid lints
        # If you retained the planned final goal, you can check distance to it here.
        # Simpler: rely on plan completion (tm.get_plan_ref returned None) OR small velocity.
        #v_ok = np.linalg.norm(self.vel[:2]) <= self.arr_tol_vel
        v_ok = False
        # You may store final goal and check ||pos - goal|| <= tol. For now, only speed criterion:
        return v_ok
    
    def _publish_nav_state(self):
        msg = UInt8(); 
        msg.data = self.sm.state.value  # IDLE=1, MISSION=2 
        self.nav_state_pub.publish(msg)

    def _clear_active_plan(self):
        """
        Best-effort: clear any active plan in the TrajectoryManager if the API exists.
        Otherwise mark our bookkeeping so we won't rely on a stale plan.
        """
        for fn in ("clear", "clear_plan", "reset"):
            if hasattr(self.tm, fn):
                try:
                    getattr(self.tm, fn)()
                    break
                except Exception:
                    pass
        # local bookkeeping
        self.plan_created = False
        self.trajectory_fresh = False
        self.at_destination = False

    def _is_manual_nav(self, nav_state: int) -> bool:                           
        VS = VehicleStatus
        # PX4 "pilot" modes; extend if you use others
        manual_modes = (
            getattr(VS, "NAVIGATION_STATE_MANUAL", 0),
            getattr(VS, "NAVIGATION_STATE_ALTCTL", 0),
            getattr(VS, "NAVIGATION_STATE_POSCTL", 0),
            getattr(VS, "NAVIGATION_STATE_ACRO", 0),
            getattr(VS, "NAVIGATION_STATE_RATTITUDE", 0),
            getattr(VS, "NAVIGATION_STATE_STAB", 0),
        )

        return nav_state in manual_modes
    
    

def main(args=None):
    rclpy.init(args=args)
    node = NavigatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
