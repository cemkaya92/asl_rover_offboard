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



def generate_launch_description():

    ns_rover  = LaunchConfiguration('ns_rover')
    sys_id  = LaunchConfiguration('sys_id')
    lidar_frame_id  = LaunchConfiguration('lidar_frame_id')
    mission_param_file  = LaunchConfiguration('mission_param_file')



    return LaunchDescription([

        DeclareLaunchArgument('ns_rover',   default_value='rover1'),
        DeclareLaunchArgument('sys_id',   default_value='2'),
        DeclareLaunchArgument('lidar_frame_id',   
                              default_value='laser'),
        
        DeclareLaunchArgument(
            'vehicle_param_file',
            default_value='asl_rover_param.yaml',
            description='Vehicle param file inside config/vehicle_parameters/'
        ),

        DeclareLaunchArgument(
            'sitl_param_file',
            default_value='experiment_coop_params.yaml',
            description='sitl param file inside config/sitl/'
        ),

        DeclareLaunchArgument(
            'controller_param_file',
            default_value='mpc_controller_asl_rover.yaml',
            description='Controller parameter file inside config/controller/'
        ),

        DeclareLaunchArgument(
            'mission_param_file',
            default_value='utari_demo_rover1_params.yaml',
            description='Mission parameter file inside config/mission/'
        ),
    
        Node(
            package=namePackage,
            executable='offboard_manager_node',
            name='offboard_manager_node',
            namespace=ns_rover,
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'disarm_on_trip': False,
                'auto_reenter_after_trip': False,
                'sys_id': sys_id
            }]
        ),

        Node(
            package=namePackage,
            executable='navigator_node',
            name='navigator_node',
            output='screen',
            namespace=ns_rover,
            parameters=[{
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'mission_param_file': LaunchConfiguration('mission_param_file'),
                'control_frequency': 20.0,
                'auto_start': False, 
            }],
        ),

        Node(
            package=namePackage,
            executable='motor_commander',
            name='motor_commander',
            namespace=ns_rover,
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file')
            }]
        ),

        Node(
            package=namePackage,
            executable='mpc_controller',
            name='mpc_controller',
            namespace=ns_rover,
            output='screen',
            parameters=[{
                'vehicle_param_file': LaunchConfiguration('vehicle_param_file'),
                'controller_param_file': LaunchConfiguration('controller_param_file'),
                'sitl_param_file': LaunchConfiguration('sitl_param_file'),
                'mpc_trajectory_topic': 'mpc/trajectory',
                'world_frame': 'map'
            }]
        ),

        Node(
            package=namePackage,
            executable='laser_scan_sector_filter',
            name='laser_scan_sector_filter',
            namespace=ns_rover,
            output='screen',
            parameters=[{
                'input_topic': '/rover1/merged_scan',
                'output_topic': '/rover1/scan_filtered',
                'min_angle_deg': 130.0,
                'max_angle_deg': 230.0,
                'mode': 'mask',
            }]
        ),

        Node(
            package='obstacle_detector',          # use the actual package you installed
            executable='obstacle_extractor_node',        # or 'obstacle_extractor_node' if that’s the exec name
            name='obstacle_extractor_node',
            output='screen',
            remappings=[
                ('scan', [ns_rover,'/scan_filtered']),
                ('pcl',  [ns_rover,'/scan_filtered']),
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
                'radius_enlargement': 0.5,

                # IMPORTANT: don't use 'map' unless you actually have a map frame
                # and you mean to transform into it. base_link is safer here.
                'frame_id': 'map',

                'obstacle_pub_topic': [ns_rover,'/obstacles/raw'],
                'obstacle_visual_pub_topic': [ns_rover,'/obstacles/raw_visualization'],
            }]
        ), 

        Node(
            package='obstacle_detector',          # use the actual package you installed
            executable='obstacle_tracker_node',        # or 'obstacle_extractor_node' if that’s the exec name
            name='obstacle_tracker_node',
            output='screen',
            remappings=[
                #('/obstacles', '/rover/tracked_obstacles'),
                #('/obstacles_visualization',  '/rover/tracked_obstacles_visualization'),
                #('/fmu/out/vehicle_odometry',  '/fmu/out/vehicle_odometry'),
            ],
            parameters=[{
                'active': True,

                'loop_rate': 100.0,
                'tracking_duration': 0.5,
                'min_correspondence_cost': 0.6,
                'std_correspondence_dev': 0.15,
                'process_variance': 0.1,
                'process_rate_variance': 0.1,
                'measurement_variance': 1.0,

                # IMPORTANT: don't use 'map' unless you actually have a map frame
                # and you mean to transform into it. base_link is safer here.
                'frame_id': 'map',

                'obstacle_sub_topic': [ns_rover,'/obstacles/raw'],
                'obstacle_pub_topic': [ns_rover,'/obstacles/tracked'],
                'obstacle_visual_pub_topic': [ns_rover,'/obstacles/tracked_visualization'],
            }]
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            namespace=ns_rover,
            arguments=[
                '0.0', '0', '0.0', '0', '3.14159', '0',
                'map', lidar_frame_id
            ],
            output='screen'
        )

    ])

