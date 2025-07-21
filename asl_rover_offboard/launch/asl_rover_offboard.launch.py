from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

vehicle_params_path = PathJoinSubstitution([
    FindPackageShare('asl_rover_offboard'),
    'config',
    'vehicle_parameters',
    'asl_rover_param.yaml'
])

sitl_params_path = PathJoinSubstitution([
    FindPackageShare('asl_rover_offboard'),
    'config',
    'sitl',
    'sitl_params.yaml'
])

mpc_params_path = PathJoinSubstitution([
    FindPackageShare('asl_rover_offboard'),
    'config',
    'controller',
    'mpc_asl_rover.yaml'
])

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'vehicle_param_file',
            default_value='asl_rover_param.yaml',
            description='Vehicle param file inside config/vehicle_parameters/'
        ),
        DeclareLaunchArgument(
            'mpc_param_file',
            default_value='mpc_asl_rover.yaml',
            description='MPC parameter file inside config/controller/'
        ),

        Node(
            package='asl_rover_offboard',
            executable='motor_commander',
            name='motor_commander',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file')
            }]
        ),
        Node(
            package='asl_rover_offboard',
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'mpc_param_file': LaunchConfiguration('mpc_param_file')
            }]
        )
    ])

