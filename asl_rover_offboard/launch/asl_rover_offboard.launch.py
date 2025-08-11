import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import Command, PathJoinSubstitution
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


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
    'mpc_controller_asl_rover.yaml'
])

bridge_params = PathJoinSubstitution([
    FindPackageShare(namePackage),
    'config',
    'sitl',
    'bridge_parameters.yaml'
])



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
            package=namePackage,
            executable='laser_scan_sector_filter',
            name='laser_scan_sector_filter',
            output='screen',
            parameters=[{
                'input_topic': '/merged_scan',
                'output_topic': '/scan_filtered',
                'min_angle_deg': 110.0,
                'max_angle_deg': 250.0,
                'mode': 'mask',
            }]
        ),

        Node(
            package=namePackage,
            executable='motor_commander',
            name='motor_commander',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file')
            }]
        ),
        
        Node(
            package=namePackage,
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'controller_param_file': LaunchConfiguration('controller_param_file')
            }]
        ),

        

        Node(
            package='obstacle_detector',          # use the actual package you installed
            executable='obstacle_extractor_node',        # or 'obstacle_extractor_node' if that’s the exec name
            name='obstacle_extractor_node',
            output='screen',
            remappings=[
                ('scan', '/scan_filtered'),
                ('pcl',  '/scan_filtered'),
            ],
            parameters=[{
                'active': True,
                'use_scan': True,
                'use_pcl': False,
                'use_sim_time': True,

                'use_split_and_merge': True,
                'circles_from_visibles': True,
                'discard_converted_segments': True,
                'transform_coordinates': True,

                'min_group_points': 10,
                'max_group_distance': 0.1,
                'distance_proportion': 0.00628,
                'max_split_distance': 0.2,
                'max_merge_separation': 0.2,
                'max_merge_spread': 0.2,
                'max_circle_radius': 0.8,
                'radius_enlargement': 0.3,

                # IMPORTANT: don't use 'map' unless you actually have a map frame
                # and you mean to transform into it. base_link is safer here.
                'frame_id': 'map',
            }]
        )

    ])

