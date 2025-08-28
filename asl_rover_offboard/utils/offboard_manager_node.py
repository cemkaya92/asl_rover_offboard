# offboard_manager_node.py
from __future__ import annotations
import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import (
    VehicleCommand, OffboardControlMode, VehicleOdometry,
    VehicleStatus, VehicleCommandAck
)
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool, Trigger

from ament_index_python.packages import get_package_share_directory

# Reuse your helpers (arm/offboard) and your ParamLoader + topics
from asl_rover_offboard.utils.vehicle_command_utils import (
    create_arm_command, create_disarm_command, create_offboard_mode_command, create_posctl_mode_command
)
from asl_rover_offboard.utils.param_loader import ParamLoader




# ----------------------- Offboard Manager Node -----------------------
class OffboardManagerNode(Node):
    """
    Central manager:
      - Publishes OffboardControlMode keepalive
      - Decides failsafe vs offboard based on state/refs/cmds
      - If safe: set OFFBOARD + ARM
      - If unsafe: switch to POSCTL (and optionally disarm) and LATCH (manual re-enable)
    """
    def __init__(self):
        super().__init__("offboard_manager")

        # params
        self.declare_parameter('vehicle_param_file', 'asl_rover_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_param.yaml')
        self.declare_parameter('disarm_on_trip', False)
        self.declare_parameter('auto_reenter_after_trip', False)  # (default: NO auto re-entry)
        self.declare_parameter('sys_id', 1)

        package_dir = get_package_share_directory('asl_rover_offboard')

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value
        self.disarm_on_trip = bool(self.get_parameter('disarm_on_trip').get_parameter_value().bool_value)
        self.auto_reenter_after_trip = bool(self.get_parameter('auto_reenter_after_trip').get_parameter_value().bool_value)
        self.sys_id = int(self.get_parameter('sys_id').get_parameter_value().integer_value)

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)

        sitl_yaml = ParamLoader(sitl_yaml_path)
        # vehicle_yaml = ParamLoader(vehicle_yaml_path)  # not strictly needed here

        # topics
        odom_topic = sitl_yaml.get_topic("odometry_topic")
        status_topic = sitl_yaml.get_topic("status_topic")
        vehicle_command_topic = sitl_yaml.get_topic("vehicle_command_topic")
        vehicle_command_ack_topic = sitl_yaml.get_topic("vehicle_command_ack_topic")
        control_cmd_topic = sitl_yaml.get_topic("control_command_topic")
        offboard_control_topic = sitl_yaml.get_topic("offboard_control_topic")


        # pubs
        self.cmd_pub = self.create_publisher(VehicleCommand, vehicle_command_topic, 10)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, offboard_control_topic, 10)

        # subs
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.create_subscription(VehicleOdometry, odom_topic, self._odom_cb, qos)
        self.create_subscription(Float32MultiArray, control_cmd_topic, self._cmd_cb, 10)
        self.create_subscription(VehicleStatus, status_topic, self._on_status, qos)
        self.create_subscription(VehicleCommandAck, vehicle_command_ack_topic, self._on_ack, qos)

        # services (to control the latch)  
        self.srv_enable = self.create_service(SetBool, 'offboard_manager/enable_offboard', self._srv_enable_offboard)
        self.srv_clear  = self.create_service(Trigger, 'offboard_manager/clear_trip', self._srv_clear_trip)


        # state
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp_us = 0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w,x,y,z
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3)


        self.have_odom = False
        self.last_cmd: np.ndarray | None = None

        # control flags
        self.offboard_set = False
        self.nav_offboard = False
        self.armed = False
        self.last_ack_ok = False

        # Command tracking / retries
        self._pending = {"offboard": None, "arm": None}   # each entry: dict(timestamp, attempts) or None
        self._ack_timeout = 0.5    # seconds to wait before retry
        self._max_attempts = 5

        # LATCH flags (block offboard re-entry after trip)  
        self.trip_latched = False         # set True on failsafe trip
        self.offboard_blocked = False     # global disable (also set on trip)


        self.get_logger().info(f"OffboardManagerNode initialized with sys_id={self.sys_id}")

        # timers
        self.keepalive_timer = self.create_timer(0.1, self._publish_offboard_keepalive)  # 10 Hz


    def _on_status(self, msg: VehicleStatus):
        try:
            OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD
            ARMED = VehicleStatus.ARMING_STATE_ARMED
        except AttributeError:
            OFFBOARD, ARMED = 14, 2  # fallback (PX4 typical values)
        self.nav_offboard = (msg.nav_state == OFFBOARD)
        self.armed = (msg.arming_state == ARMED)

    def _on_ack(self, ack: VehicleCommandAck):
        # Map enum in a tolerant way
        ACCEPTED = getattr(VehicleCommandAck, "VEHICLE_RESULT_ACCEPTED", 0)
        TEMP_REJ = getattr(VehicleCommandAck, "VEHICLE_RESULT_TEMPORARILY_REJECTED", 1)
        DENIED   = getattr(VehicleCommandAck, "VEHICLE_RESULT_DENIED", 3)

        cmd = int(ack.command)
        ok = (ack.result == ACCEPTED)

        # Which logical key?
        if cmd == VehicleCommand.VEHICLE_CMD_DO_SET_MODE:
            key = "offboard"
        elif cmd == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM:
            key = "arm"
        else:
            key = None

        if key:
            # clear pending
            self._pending[key] = None

        self.last_ack_ok = ok
        if ok:
            self.get_logger().info(f"ACK OK for cmd {cmd}")
        else:
            self.get_logger().warn(f"ACK result {ack.result} for cmd {cmd} (param1={ack.result_param1})")


    # ---------- services ----------
    def _srv_enable_offboard(self, req: SetBool.Request, res: SetBool.Response):
        if req.data:
            self.offboard_blocked = False
            self.trip_latched = False
            res.success = True
            res.message = "Offboard enabled; latch cleared."
            self.get_logger().info(res.message)
        else:
            self.offboard_blocked = True
            res.success = True
            res.message = "Offboard disabled."
            self.get_logger().warn(res.message)
        return res

    def _srv_clear_trip(self, req: Trigger.Request, res: Trigger.Response):
        self.trip_latched = False
        res.success = True
        res.message = "Trip latch cleared."
        self.get_logger().info(res.message)
        return res
    
    # ---------- callbacks ----------
    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0


    def _odom_cb(self, msg: VehicleOdometry):
        self.px4_timestamp_us = msg.timestamp
        self.t_sim = self._timing(msg.timestamp)
        self.pos = np.array(msg.position, dtype=float)
        self.vel = np.array(msg.velocity, dtype=float)
        self.q = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=float)  # w,x,y,z
        self.rpy = self._quat_to_eul(self.q)
        self.omega_body = np.array(msg.angular_velocity, dtype=float)

        if not self.have_odom:
            self.get_logger().info("First odom received.")
            self.have_odom = True


    def _cmd_cb(self, msg: Float32MultiArray):
        self.last_cmd = np.asarray(msg.data, dtype=float)

    # Helpers

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _now_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    def _request_offboard_mode(self):
        """
        VEHICLE_CMD_DO_SET_MODE (176):
        param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED (1)
        param2 = PX4_CUSTOM_MAIN_MODE_OFFBOARD (6)
        param3 = PX4_CUSTOM_SUB_MODE (0)
        """
        now_us = self._now_us()
        self.cmd_pub.publish(create_offboard_mode_command(now_us, self.sys_id))
        self._pending["offboard"] = {
            "timestamp": self._now_s(),
            "attempts": (self._pending["offboard"]["attempts"] + 1 if self._pending["offboard"] else 1),
        }
        self.get_logger().info("Requested OFFBOARD mode")

    def _request_arm(self, arm: bool = True):
        now_us = self._now_us()
        if arm:
            self.cmd_pub.publish(create_arm_command(now_us, self.sys_id))
            self.get_logger().info("Requested ARM")
        else:
            self.cmd_pub.publish(create_disarm_command(now_us, self.sys_id))
            self.get_logger().info("Requested DISARM")
        self._pending["arm"] = {
            "timestamp": self._now_s(),
            "attempts": (self._pending["arm"]["attempts"] + 1 if self._pending["arm"] else 1),
        }

    def _maybe_retry(self, key: str):
        """Retry if no ACCEPTED ack and status not yet reached within timeout."""
        pend = self._pending.get(key)
        if not pend:
            return

        # If status already satisfied, clear pending
        if key == "offboard" and self.nav_offboard:
            self._pending[key] = None
            return
        if key == "arm" and self.armed:
            self._pending[key] = None
            return

        if pend["attempts"] >= self._max_attempts:
            self.get_logger().error(f"{key}: max attempts reached without success")
            self._pending[key] = None
            return

        if (self._now_s() - pend["timestamp"]) >= self._ack_timeout:
            # timed out → retry
            if key == "offboard":
                self.get_logger().warn("Retrying OFFBOARD request…")
                self._request_offboard_mode()
            elif key == "arm":
                self.get_logger().warn("Retrying ARM request…")
                self._request_arm(True)


    # ---------- timers ----------
    def _publish_offboard_keepalive(self):

        # Always safe to publish keepalive (it does not switch modes by itself)       
        now_us = self._now_us()
        offboard = OffboardControlMode()
        offboard.timestamp = now_us
        offboard.position = False
        offboard.velocity = False
        offboard.acceleration = False
        offboard.attitude = False
        offboard.body_rate = False
        offboard.thrust_and_torque = False
        offboard.direct_actuator = True
        self.offboard_ctrl_pub.publish(offboard)

        # 2) Decide whether we want Offboard/Arm active
        want_control = (not self.offboard_blocked) and (not self.trip_latched) and self.have_odom 

        if not want_control:
            self.offboard_set = False
            return
    
        # 3) Stage OFFBOARD first, then ARM (with ACK + status verification)
        if not self.nav_offboard and self._pending["offboard"] is None:
            self._request_offboard_mode()

        # Retry OFFBOARD if needed
        self._maybe_retry("offboard")

        # Once OFFBOARD is active, request ARM (if not yet armed)
        if self.nav_offboard and (not self.armed) and self._pending["arm"] is None:
            self._request_arm(True)

        # Retry ARM if needed
        self._maybe_retry("arm")

        # 4) Consider the switch successful only when BOTH are true
        self.offboard_set = self.nav_offboard and self.armed
        

    # ---------- utils ----------
    @staticmethod
    def _quat_to_eul(q_wxyz: np.ndarray) -> np.ndarray:
        # PX4 gives [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        return r.as_euler('ZYX', degrees=False)  # [yaw, pitch, roll]

def main(args=None):
    rclpy.init(args=args)
    node = OffboardManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
