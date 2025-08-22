import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import ActuatorMotors, OffboardControlMode, VehicleCommand, VehicleStatus, VehicleOdometry, VehicleCommandAck
from asl_rover_offboard.utils.vehicle_command_utils import create_arm_command, create_offboard_mode_command

from asl_rover_offboard.utils.param_loader import ParamLoader
# from asl_rover_offboard.control.control_allocator import ControlAllocator

from ament_index_python.packages import get_package_share_directory
import os

def qos_sensor():
    q = QoSProfile(depth=10)
    q.reliability = ReliabilityPolicy.BEST_EFFORT
    q.history = HistoryPolicy.KEEP_LAST
    q.durability = DurabilityPolicy.VOLATILE
    return q


class MotorCommander(Node):
    def __init__(self):
        super().__init__('motor_commander')

        package_dir = get_package_share_directory('asl_rover_offboard')
        
        # Declare param with default
        self.declare_parameter('vehicle_param_file', 'asl_rover_param.yaml')
        self.declare_parameter('sitl_param_file', 'sitl_params.yaml')

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        sitl_param_file = self.get_parameter('sitl_param_file').get_parameter_value().string_value

        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', sitl_param_file)

        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)

        # UAV parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()

        self.L = 0.5 * self.vehicle_params.base_width
        self.R = self.vehicle_params.wheel_radius
        self.max_wheel_speed = self.vehicle_params.max_linear_speed / self.R
        self.max_angular_speed_rad_s = self.vehicle_params.max_angular_speed * np.pi / 180.0 # rad/s

        self.got_odom = False
        self.nav_offboard = False
        self.armed = False
        self.last_ack_ok = False

        # pub / sub
        self.motor_pub = self.create_publisher(ActuatorMotors, sitl_yaml.get_topic("actuator_control_topic"), 10)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, sitl_yaml.get_topic("offboard_control_topic"), 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, sitl_yaml.get_topic("vehicle_command_topic"), 10)
        # self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, sitl_yaml.get_topic("thrust_setpoints_topic"), 1)
        # self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, sitl_yaml.get_topic("torque_setpoints_topic"), 1)
        
        self.create_subscription(Float32MultiArray, sitl_yaml.get_topic("control_command_topic"), self.control_cmd_callback, 10)
        self.create_subscription(VehicleOdometry, sitl_yaml.get_topic("odometry_topic"), self._on_odom, qos_sensor())
        self.create_subscription(VehicleStatus, sitl_yaml.get_topic("status_topic"), self._on_status, qos_sensor())
        self.create_subscription(VehicleCommandAck, sitl_yaml.get_topic("vehicle_command_ack_topic"), self._on_ack, 10)


        self.sys_id = sitl_yaml.get_nested(["sys_id"],1)

        self.get_logger().info(f"sys_id= {self.sys_id}")

        # Command retry state
        self.offboard_requested = False
        self.arm_requested = False
        self.last_cmd_time = self.get_clock().now()
        self.cmd_period = 0.4  # seconds between retries

        # initial states
        self.latest_motor_cmd = ActuatorMotors()
        self.latest_motor_cmd.control = [0.0] * 12

        
        # static allocation matrices
        

        # self.get_logger().info("[mixing_matrix] =\n" + np.array2string(mixing_matrix, precision=10, suppress_small=True))
        # self.get_logger().info("[mixing_matrix_inv] =\n" + np.array2string(self.throttles_to_normalized_torques_and_thrust, precision=10, suppress_small=True))

        # Timers
        self.motor_command_timer = self.create_timer(0.01, self.motor_command_timer_callback) # 100 Hz
        self.offboard_timer = self.create_timer(0.1, self.publish_offboard_control_mode)  # 5 Hz
        self.command_timer = self.create_timer(0.1, self._command_tick) 

        self.get_logger().info("MotorCommander with Offboard control started")


    def _on_odom(self, _msg):
        if not self.got_odom:
            self.get_logger().info("First odometry received.")
        self.got_odom = True

    def _on_status(self, msg: VehicleStatus):
        # NAVIGATION_STATE_OFFBOARD = 14 on recent PX4; use constant if you have it
        NAV_OFFBOARD = 14
        ARMING_STATE_ARMED = 2
        self.nav_offboard = (msg.nav_state == NAV_OFFBOARD)
        self.armed = (msg.arming_state == ARMING_STATE_ARMED)

    def _on_ack(self, ack: VehicleCommandAck):
        # VEHICLE_RESULT_ACCEPTED = 0
        VEHICLE_RESULT_ACCEPTED = 0
        if ack.result == VEHICLE_RESULT_ACCEPTED:
            self.last_ack_ok = True


    def motor_command_timer_callback(self):
        #a=0.0
        self.motor_pub.publish(self.latest_motor_cmd)



    def publish_offboard_control_mode(self):
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        offboard_msg = OffboardControlMode()
        offboard_msg.timestamp = now_us
        offboard_msg.position = False
        offboard_msg.velocity = False
        offboard_msg.acceleration = False
        offboard_msg.attitude = False
        offboard_msg.body_rate = False
        offboard_msg.thrust_and_torque = False
        offboard_msg.direct_actuator = True
        self.offboard_ctrl_pub.publish(offboard_msg)


    def _ready_to_request(self) -> bool:
        """We only try to switch when:
           - odometry arrived (EKF up), and
           - PX4 is discovered as a subscriber of our command sinks.
        """
        sinks_ok = (self.offboard_ctrl_pub.get_subscription_count() > 0) and (self.cmd_pub.get_subscription_count() > 0)
        return self.got_odom and sinks_ok
        
    def _command_tick(self):
        now = self.get_clock().now()
        elapsed = (now - self.last_cmd_time).nanoseconds / 1e9

        if self.nav_offboard and self.armed:
            # We’re in the desired state; nothing to do
            return

        if not self._ready_to_request():
            # Wait until telemetry and discovery are ready
            return

        if elapsed < self.cmd_period:
            return

        now_us = int(now.nanoseconds / 1000)

        # Retry OFFBOARD until nav_state flips or we get an accepted ACK
        if not self.nav_offboard:
            self.cmd_pub.publish(create_offboard_mode_command(now_us, self.sys_id))
            self.get_logger().debug("Requesting OFFBOARD...")
            self.last_cmd_time = now
            return

        # Retry ARM until armed or accepted ACK
        if not self.armed:
            self.cmd_pub.publish(create_arm_command(now_us, self.sys_id))
            self.get_logger().debug("Requesting ARM...")
            self.last_cmd_time = now
            return


    def control_cmd_callback(self, msg):

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        v_cmd =  np.clip(msg.data[0], -self.vehicle_params.max_linear_speed, self.vehicle_params.max_linear_speed)
        omega_cmd =  np.clip(msg.data[1], -self.max_angular_speed_rad_s, self.max_angular_speed_rad_s) # 25.0*np.pi/180.0 #
        omega_scaled = omega_cmd*(3.0 - 2.0 * np.abs(v_cmd)/self.vehicle_params.max_linear_speed)
        #self.get_logger().info(f"V= {v_cmd} | Omega= {omega_cmd} | OmegaScaled= {omega_scaled}")

        wl_cmd = (v_cmd - omega_scaled * self.L) / self.R
        wr_cmd = (v_cmd + omega_scaled * self.L) / self.R


        # Prepare thrust message
        self.latest_motor_cmd.timestamp = now_us
        self.latest_motor_cmd.control[0] = self.vehicle_params.zero_position_armed + ( wl_cmd / self.max_wheel_speed ) / 2.0
        self.latest_motor_cmd.control[1] = self.vehicle_params.zero_position_armed + ( wr_cmd / self.max_wheel_speed ) / 2.0

        #self.get_logger().info(f"wl_cmd= {wl_cmd} | wr_cmd= {wr_cmd} | max_wheel_speed= {self.max_wheel_speed}")
        #self.get_logger().info(f"norm_omega_left= {self.latest_motor_cmd.control[0]} | norm_omega_right= {self.latest_motor_cmd.control[1]}")




def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()