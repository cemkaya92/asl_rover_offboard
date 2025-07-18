import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import ActuatorMotors, OffboardControlMode, VehicleCommand, VehicleThrustSetpoint, VehicleTorqueSetpoint 
from asl_rover_offboard.utils.vehicle_command_utils import create_arm_command, create_offboard_mode_command

from asl_rover_offboard.utils.param_loader import ParamLoader
# from asl_rover_offboard.control.control_allocator import ControlAllocator

from ament_index_python.packages import get_package_share_directory
import os


class MotorCommander(Node):
    def __init__(self):
        super().__init__('motor_commander')

        package_dir = get_package_share_directory('asl_rover_offboard')
        
        # Declare param with default
        self.declare_parameter('vehicle_param_file', 'asl_rover_param.yaml')
        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value

        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', 'sitl_params.yaml')
        vehicle_yaml_path = os.path.join(package_dir, 'config', 'vehicle_parameters', vehicle_param_file)

        # Load parameters
        sitl_yaml = ParamLoader(sitl_yaml_path)
        vehicle_yaml = ParamLoader(vehicle_yaml_path)

        # UAV parameters
        self.vehicle_params = vehicle_yaml.get_vehicle_params()

        # pub / sub
        self.motor_pub = self.create_publisher(ActuatorMotors, sitl_yaml.get_topic("actuator_control_topic"), 10)
        self.offboard_ctrl_pub = self.create_publisher(OffboardControlMode, sitl_yaml.get_topic("offboard_control_topic"), 10)
        self.cmd_pub = self.create_publisher(VehicleCommand, sitl_yaml.get_topic("vehicle_command_topic"), 10)
        # self.thrust_pub = self.create_publisher(VehicleThrustSetpoint, sitl_yaml.get_topic("thrust_setpoints_topic"), 1)
        # self.torque_pub = self.create_publisher(VehicleTorqueSetpoint, sitl_yaml.get_topic("torque_setpoints_topic"), 1)
        
        self.create_subscription(Float32MultiArray, sitl_yaml.get_topic("mpc_command_topic"), self.mpc_cmd_callback, 10)

        # initial states
        self.latest_motor_cmd = ActuatorMotors()
        self.latest_motor_cmd.control = [0.0] * 12

        
        # static allocation matrices
        

        # self.get_logger().info("[mixing_matrix] =\n" + np.array2string(mixing_matrix, precision=10, suppress_small=True))
        # self.get_logger().info("[mixing_matrix_inv] =\n" + np.array2string(self.throttles_to_normalized_torques_and_thrust, precision=10, suppress_small=True))

        # Timers
        self.motor_command_timer = self.create_timer(0.01, self.motor_command_timer_callback) # 100 Hz
        self.offboard_timer = self.create_timer(0.2, self.publish_offboard_control_mode)  # 5 Hz
        self.offboard_set = False
        self.get_logger().info("MotorCommander with Offboard control started")

    def motor_command_timer_callback(self):

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

        # Start Offboard + Arm only after receiving first valid control command
        # if not self.offboard_set and any([abs(x) > 1e-3 for x in self.latest_motor_cmd]):
        if not self.offboard_set:
            self.cmd_pub.publish(create_offboard_mode_command(now_us))
            self.cmd_pub.publish(create_arm_command(now_us))
            self.offboard_set = True
            self.get_logger().info("Sent OFFBOARD and ARM command") 

    
        
        
    def mpc_cmd_callback(self, msg):

        now_us = int(self.get_clock().now().nanoseconds / 1000)

        wl_cmd = msg.data[0]
        wr_cmd = msg.data[1]
        self.get_logger().info(f"omega_left= {wl_cmd} | omega_right= {wr_cmd}")


        # Prepare thrust message
        self.latest_motor_cmd.timestamp = now_us
        self.latest_motor_cmd.control[0] = wl_cmd / self.vehicle_params.max_wheel_speed
        self.latest_motor_cmd.control[1] = wr_cmd / self.vehicle_params.max_wheel_speed




def main(args=None):
    rclpy.init(args=args)
    node = MotorCommander()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()