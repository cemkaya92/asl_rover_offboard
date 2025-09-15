import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

import numpy as np
from std_msgs.msg import Float32MultiArray, UInt8
from px4_msgs.msg import ActuatorMotors

from asl_rover_offboard.navigation.state_machine import NavState

from asl_rover_offboard.utils.param_loader import ParamLoader
# from asl_rover_offboard.control.control_allocator import ControlAllocator

from ament_index_python.packages import get_package_share_directory
import os
import signal, time


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

        # Topics
        nav_state_topic = sitl_yaml.get_topic("nav_state_topic")
        actuator_control_topic = sitl_yaml.get_topic("actuator_control_topic")
        control_command_topic = sitl_yaml.get_topic("control_command_topic")

        # Vehicle parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()

        self.L = 0.5 * self.vehicle_params.base_width
        self.R = self.vehicle_params.wheel_radius
        self.max_wheel_speed = self.vehicle_params.max_linear_speed / self.R
        self.max_angular_speed_rad_s = self.vehicle_params.max_angular_speed * np.pi / 180.0 # rad/s

        self.allow_commands = False  

        self.nav_state = NavState.IDLE # start in IDLE

        self.in_SITL = False

        # QOS Options
        plan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,  # <-- latch last message
        )

        # pub 
        self.motor_pub = self.create_publisher(ActuatorMotors, actuator_control_topic, 10)
        
        # sub
        self.create_subscription(Float32MultiArray, control_command_topic, self._control_cmd_callback, 10)
        self.create_subscription(UInt8, nav_state_topic, self._on_nav_state, plan_qos)


        # initial states
        self.latest_motor_cmd = ActuatorMotors()
        self.latest_motor_cmd.control = [self.vehicle_params.zero_position_armed, 
                                         self.vehicle_params.zero_position_armed] + [0.0] * 10
        
        self.latest_motor_cmd.reversible_flags = 3

        # a ready-to-send neutral msg
        self._neutral_msg = ActuatorMotors()
        self._neutral_msg.control = [self.vehicle_params.zero_position_armed,
                                     self.vehicle_params.zero_position_armed] + [0.0] * 10


        # ------------------------------------

        # static allocation matrices
        # self.get_logger().info("[mixing_matrix] =\n" + np.array2string(mixing_matrix, precision=10, suppress_small=True))
        # self.get_logger().info("[mixing_matrix_inv] =\n" + np.array2string(self.throttles_to_normalized_torques_and_thrust, precision=10, suppress_small=True))

        # Timers
        self.motor_command_timer = self.create_timer(0.01, self._motor_command_timer_callback) # 100 Hz

        self.get_logger().info("MotorCommander with Offboard control started")



    # ---------- callbacks ----------
    def _on_nav_state(self, msg: UInt8):
        
        raw = int(msg.data)
        prev = getattr(self, "nav_state", NavState.UNKNOWN)

        # Normalize to enum
        try:
            state = NavState(raw)
        except ValueError:
            self.get_logger().warn(f"Unknown nav_state {raw}; treating as UNKNOWN")
            state = NavState.UNKNOWN

        self.nav_state = state


        self.allow_commands = (state in {NavState.IDLE, NavState.MISSION})
        
        if not self.allow_commands:
            self._set_latest_to_neutral()
            

    def _motor_command_timer_callback(self):
        
        if self.allow_commands:
            self.motor_pub.publish(self.latest_motor_cmd)
        

        #self.get_logger().info(f"latest_motor_cmd= {self.latest_motor_cmd}")


    def _control_cmd_callback(self, msg):

        if not self.allow_commands:
            return
        
        now_us = int(self.get_clock().now().nanoseconds / 1000)

        v_cmd =  np.clip(msg.data[0], -self.vehicle_params.max_linear_speed, self.vehicle_params.max_linear_speed)
        omega_cmd =  np.clip(msg.data[1], -self.max_angular_speed_rad_s, self.max_angular_speed_rad_s) # 25.0*np.pi/180.0 #
#        omega_scaled = omega_cmd*(3.0 - 2.0 * np.abs(v_cmd)/self.vehicle_params.max_linear_speed)
        #self.get_logger().info(f"V= {v_cmd} | Omega= {omega_cmd} | OmegaScaled= {omega_scaled}")

        wl_cmd = (v_cmd - 0.5 * omega_cmd * self.L) / self.R
        wr_cmd = (v_cmd + 0.5 * omega_cmd * self.L) / self.R


        # Prepare thrust message
        self.latest_motor_cmd.timestamp = now_us
        # forward speed works between [0.5 .. 1] range
        # backward speed works between [-1.0 .. -0.5] range

        if (self.in_SITL):
            self.latest_motor_cmd.reversible_flags = 0
            self.latest_motor_cmd.control[0] = self.vehicle_params.zero_position_armed + ( wl_cmd / self.max_wheel_speed ) / 2.0
            self.latest_motor_cmd.control[1] = self.vehicle_params.zero_position_armed + ( wr_cmd / self.max_wheel_speed ) / 2.0
        else:

            if (wl_cmd>0.0):
                self.latest_motor_cmd.control[0] = self.vehicle_params.zero_position_armed + ( wl_cmd / self.max_wheel_speed ) / 2.0
            else:
                self.latest_motor_cmd.control[0] = -self.vehicle_params.zero_position_armed + ( wl_cmd / self.max_wheel_speed ) / 2.0

            if (wr_cmd>0.0):
                self.latest_motor_cmd.control[1] = self.vehicle_params.zero_position_armed + ( wr_cmd / self.max_wheel_speed ) / 2.0
            else:
                self.latest_motor_cmd.control[1] = -self.vehicle_params.zero_position_armed + ( wr_cmd / self.max_wheel_speed ) / 2.0

        # 

        # self.get_logger().info(f"wl_cmd= {wl_cmd} | wr_cmd= {wr_cmd} | max_wheel_speed= {self.max_wheel_speed}")
        # self.get_logger().info(f"norm_omega_left= {self.latest_motor_cmd.control[0]} | norm_omega_right= {self.latest_motor_cmd.control[1]}")


    # ---------- neutral helpers ----------
    def _set_latest_to_neutral(self):
        z = float(0.0*self.vehicle_params.zero_position_armed)
        self.latest_motor_cmd.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.latest_motor_cmd.control[0] = z
        self.latest_motor_cmd.control[1] = z

    def _publish_neutral_once(self):
        # call only while the context is still valid
        self._neutral_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.motor_pub.publish(self._neutral_msg)



def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()

    exe = SingleThreadedExecutor()
    exe.add_node(node)

    # trap signals so rclpy doesn't tear the context down before we burst neutral
    shutdown = {"req": False}
    def _sig_handler(signum, frame):
        shutdown["req"] = True
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        # spin manually so we can intercept shutdown
        while rclpy.ok() and not shutdown["req"]:
            exe.spin_once(timeout_sec=0.1)
    finally:
        # 1) immediately switch desired command to neutral
        node._set_latest_to_neutral()

        # 2) publish a short neutral burst while the context is STILL valid
        deadline = time.time() + 0.3  # ~300 ms
        while time.time() < deadline and rclpy.ok():
            node._publish_neutral_once()
            exe.spin_once(timeout_sec=0.0)  # flush any pending work
            time.sleep(0.02)               # ~50 Hz

        # 3) clean teardown
        exe.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
