import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import numpy as np

from std_msgs.msg import Float32MultiArray
from px4_msgs.msg import VehicleOdometry, TimesyncStatus


from asl_rover_offboard.guidance.trajectory import eval_traj
from asl_rover_offboard.utils.first_order_filter import FirstOrderFilter

from asl_rover_offboard.utils.ploter import Logger

from asl_rover_offboard.utils.param_loader import ParamLoader

from ament_index_python.packages import get_package_share_directory
import os

import signal




class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        
        self.declare_parameter('vehicle_param_file', 'asl_rover_param.yaml')
        self.declare_parameter('controller_param_file', 'controller_asl_rover.yaml')

        vehicle_param_file = self.get_parameter('vehicle_param_file').get_parameter_value().string_value
        controller_param_file = self.get_parameter('controller_param_file').get_parameter_value().string_value

        package_dir = get_package_share_directory('asl_rover_offboard')
        
        sitl_yaml_path = os.path.join(package_dir, 'config', 'sitl', 'sitl_params.yaml')
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

        # Controller parameters
        self.control_params = controller_yaml.get_control_params()
        # Vehicle parameters
        vehicle_params = vehicle_yaml.get_vehicle_params()



        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Sub/Pub
        self.sub_odom = self.create_subscription(
            VehicleOdometry, 
            odom_topic, 
            self.odom_callback, qos_profile)
        
        self.sub_sync = self.create_subscription(
            TimesyncStatus, 
            timesync_topic, 
            self.sync_callback, qos_profile)
        #self.pub_motor = self.create_publisher(ActuatorMotors, '/fmu/in/actuator_motors', 10)
        self.motor_cmd_pub = self.create_publisher(
            Float32MultiArray, 
            control_cmd_topic, 
            10)
                        

        # State variables
        self.t0 = None
        self.t_sim = 0.0
        self.px4_timestamp = 0
        self.odom_ready = False

        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.q = np.zeros(4)
        self.rpy = np.zeros(3)
        self.omega_body = np.zeros(3) # Angular velocity in body frame


        # First-order filters for smoother control outputs
        self.thrust_filter = FirstOrderFilter(alpha=0.8)
        self.roll_torque_filter = FirstOrderFilter(alpha=0.5)
        self.pitch_torque_filter = FirstOrderFilter(alpha=0.5)
        self.yaw_torque_filter = FirstOrderFilter(alpha=0.9)

        # Data logging
        self.logger = Logger()
        

        self.timer_mpc = self.create_timer(1.0 / self.control_params.frequency, self.control_loop)  # 100 Hz

        self.get_logger().info("Controller Node initialized.")


    def _timing(self, stamp_us):
        t = stamp_us * 1e-6
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def odom_callback(self, msg):
        self.px4_timestamp = msg.timestamp
        t_now = self._timing(msg.timestamp)
        self.t_sim = t_now

        # Position
        self.pos = np.array(msg.position)

        # Attitude (Quaternion to Euler)
        self.q[0],self.q[1],self.q[2],self.q[3] = msg.q
        self.rpy[2],self.rpy[1],self.rpy[0] = self.quat_to_eul(self.q)
        
        #rot_mat = eul2rotm_py(self.rpy)
        
        # World-frame Velocity (transform from body frame)
        self.vel = np.array(msg.velocity)

        # Body-frame Angular Velocity
        self.omega_body = np.array(msg.angular_velocity)

        if not self.odom_ready:
            self.get_logger().warn("First odometry message is received...")
            self.odom_ready = True
            return
        
        self.odom_ready = True

    def quat_to_eul(self, q_xyzw):
        # PX4: [w, x, y, z]
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([q_xyzw[1], q_xyzw[2], q_xyzw[3], q_xyzw[0]])
        return r.as_euler('ZYX', degrees=False)

    def sync_callback(self, msg):
        self.px4_timestamp = msg.timestamp

    def control_loop(self):
        """
        Runs at
        """
        if not self.odom_ready:
            self.get_logger().warn("Waiting for odometry...")
            return
        
        # Construct the 3-dimensional state vector
        # v_curr = np.linalg.norm(self.vel[:2])  # magnitude of [vx, vy]
        x0 = np.array([self.pos[0], self.pos[1], self.rpy[2]])

        # Get reference trajectory point
        # dt = 1.0 / 50.0
        # v_ref = 1.0
        # w_ref = 25.0 * np.pi / 180.0
        # psi_ref = self.rpy[2] + w_ref * dt
        # pos_x_ref = self.pos[0] + v_ref * np.cos(psi_ref) * dt
        # pos_y_ref = self.pos[1] + v_ref * np.sin(psi_ref) * dt
        # x_ref = np.array([pos_x_ref, pos_y_ref, psi_ref, v_ref, w_ref])

        pos_ref, vel_ref = eval_traj(self.t_sim,x0[0:2])
        
        # Compute velocity and heading references from the trajectory
        vx, vy = vel_ref[0], vel_ref[1]

        psi_ref = np.arctan2(vy, vx)
        v_ref = np.sqrt(vx**2 + vy**2)
        psi_dot_ref = 1.0 * self.angle_diff(psi_ref , self.rpy[2])


        # Construct error terms
        sin_theta = np.sin(self.rpy[2])
        cos_theta = np.cos(self.rpy[2])

        err_x =  cos_theta*(pos_ref[0]-self.pos[0]) + sin_theta*(pos_ref[1]-self.pos[1])
        err_y = -sin_theta*(pos_ref[0]-self.pos[0]) + cos_theta*(pos_ref[1]-self.pos[1])
        err_theta = psi_ref - self.rpy[2]

        # Control Law for  [v, w]s
        v_cmd     = v_ref*np.cos(err_theta) + self.control_params.k1*err_x
        omega_cmd = psi_dot_ref + self.control_params.k2*err_y + self.control_params.k3*np.sin(err_theta)

        u_input = np.array([v_cmd, omega_cmd])


        self.get_logger().info(f"x_ref= {pos_ref[0]} | diff= {pos_ref[0] - x0[0]}")
        self.get_logger().info(f"y_ref= {pos_ref[1]} | diff= {pos_ref[1] - x0[1]}")
        # self.get_logger().info(f"v_ref= {p_ref} | diff= {np.array(v_ref) - self.vel}")

        # self.get_logger().info(f"roll= {self.rpy[0]*180/np.pi} | diff= {(0.0 - self.rpy[0])*180/np.pi}")
        # self.get_logger().info(f"pitch= {self.rpy[1]*180/np.pi} | diff= {(0.0 - self.rpy[1])*180/np.pi}")
        self.get_logger().info(f"yaw_cmd= {psi_ref*180/np.pi} | diff= {(psi_ref - self.rpy[2])*180/np.pi}")
        #u_mpc[0] = (-np.sqrt(u_mpc[0] / (4 * KF_SIM))) / MAX_OMEGA_SIM
        # yaw_command = (yaw_command / self.max_torque)

        # roll_command = (roll_command / self.max_torque)
        # pitch_command = (pitch_command / self.max_torque)

        # thrust_command = (-np.sqrt(0.027*9.8066 / (4 * KF_SIM)) / MAX_OMEGA_SIM )

        # self.logger.log(self.t_sim, self.pos, self.vel, self.rpy, np.array([]), np.array([v_cmd,omega_cmd]))


        # Publish
        msg = Float32MultiArray()
        msg.data = u_input.tolist()
        self.motor_cmd_pub.publish(msg)

   

        #self.get_logger().info(f"t={self.t_sim:.2f} | pos={self.pos.round(2)} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"t={self.t_sim:.2f} | u={np.round(u_mpc, 4)}")
        # self.get_logger().info(f"x={x0}")
        # self.get_logger().info(f"x_ref={x_ref}")
        # self.get_logger().info(f"x_dif={x_ref-x0}")

    def angle_diff(self, a, b):
        """
        Compute the wrapped difference between two angles in radians.
        Both a and b can be scalars or numpy arrays.
        Returns value in [-pi, pi]
        """
        return (a - b + np.pi) % (2 * np.pi) - np.pi

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