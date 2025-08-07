import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


namePackage = 'asl_rover_offboard'

vehicle_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'vehicle_parameters',
    'asl_rover_param.yaml'
])

sitl_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'sitl',
    'sitl_params.yaml'
])

mpc_params_path = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'controller',
    'controller_asl_rover.yaml'
])

bridge_params = os.path.join(
    get_package_share_directory(namePackage),
    'config',
    'sitl',
    'bridge_parameters.yaml'
)



def generate_launch_description():
    return LaunchDescription([

        DeclareLaunchArgument(
            'vehicle_param_file',
            default_value='asl_rover_param.yaml',
            description='Vehicle param file inside config/vehicle_parameters/'
        ),
        DeclareLaunchArgument(
            'controller_param_file',
            default_value='controller_asl_rover.yaml',
            description='Controller parameter file inside config/controller/'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=[
                '--ros-args',
                '-p',
                f'config_file:={bridge_params}',
            ],
            output='screen',
        ),


        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            arguments=[
                '0.0', '0', '0.0', '0', '0', '0',
                'map', 'asl_rover_0/lidar_2d_v2/link/lidar_2d_v2'
            ],
            output='screen'
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
            executable='backstepping_controller',
            name='backstepping_controller',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'controller_param_file': LaunchConfiguration('controller_param_file')
            }]
        )

    ])

