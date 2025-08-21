from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # --- Arguments for XRCE Agent ---
    xrce_dev_arg = DeclareLaunchArgument(
        'xrce_dev', default_value='/dev/ttyUSB0',
        description='Serial device for Micro XRCE-DDS Agent (use /dev/px4 if you set a udev rule)'
    )
    xrce_baud_arg = DeclareLaunchArgument(
        'xrce_baud', default_value='921600',
        description='Baudrate for Micro XRCE-DDS Agent'
    )
    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/ttyUSB1',
        description='Serial port for Lidar'
    )

    xrce_ns_arg = DeclareLaunchArgument(
        'xrce_ns', default_value='rover1',
        description='Namespace for Micro XRCE-DDS Agent'
    )
    
    # Paths to your two launch files
    sllidar_launch = PathJoinSubstitution([
        FindPackageShare('sllidar_ros2'),
        'launch',
        'sllidar_a2m8_launch.py'
    ])

    rover_launch = PathJoinSubstitution([
        FindPackageShare('asl_rover_offboard'),
        'launch',
        'asl_rover_offboard_hardware.launch.py'
    ])

    # --- Start Micro XRCE-DDS Agent ---
    # (No sudo needed if you fixed permissions)
    agent_proc = ExecuteProcess(
        cmd=[
            'MicroXRCEAgent', 'serial',
            '--dev', LaunchConfiguration('xrce_dev'),
            '-b',   LaunchConfiguration('xrce_baud'),
            '-n',   LaunchConfiguration('xrce_ns')   # <--- namespace arg
        ],
        output='screen'
    )

    start_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sllidar_launch),
        launch_arguments={
            'serial_port': LaunchConfiguration('lidar_port'),  
            'frame_id': 'laser',
            'inverted': 'false'
        }.items()
    )

    start_rover = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rover_launch)
    )

    # Optional: wait ~2s so the Agent is listening before PX4/rover stack starts publishing/subscribing
    delayed_stack = TimerAction(period=2.0, actions=[start_lidar, start_rover])

    return LaunchDescription([
        xrce_dev_arg, xrce_baud_arg, lidar_port_arg, xrce_ns_arg,
        agent_proc,
        delayed_stack
    ])
