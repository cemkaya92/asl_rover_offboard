from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess,
    LogInfo, RegisterEventHandler, OpaqueFunction, SetLaunchConfiguration
)
from launch.event_handlers import OnProcessIO, OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PROBE_READY_MARKER = 'Telemetry received AND command subscribers discovered.'
XRCE_READY_MARKER = 'participant created'   # seen in Agent logs

def generate_launch_description():

    init_flag = SetLaunchConfiguration('stack_started', '0')

    # --- Arguments for XRCE Agent ---
    xrce_dev_arg = DeclareLaunchArgument(
        'xrce_dev', default_value='/dev/px4',
        description='Serial device for Micro XRCE-DDS Agent (use /dev/px4 if you set a udev rule)'
    )
    xrce_baud_arg = DeclareLaunchArgument(
        'xrce_baud', default_value='921600',
        description='Baudrate for Micro XRCE-DDS Agent'
    )
    xrce_ns_arg = DeclareLaunchArgument(
        'xrce_ns', default_value='rover1',
        description='Namespace for Micro XRCE-DDS Agent'
    )

    lidar_port_arg = DeclareLaunchArgument(
        'lidar_port', default_value='/dev/lidar',
        description='Serial port for Lidar'
    )
    lidar_baud_arg = DeclareLaunchArgument(
        'lidar_baud', default_value='115200',
        description='Serial baud rate for Lidar'
    )

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

    # Deep readiness probe: confirms PX4 pubs/subs & first msgs
    px4_probe = Node(
        package='asl_rover_offboard',
        executable='wait_px4_ready',
        name='wait_px4_ready',
        # pass just the raw ns name, not '/ns'
        arguments=['--ns', LaunchConfiguration('xrce_ns'), '--timeout', '40.0'],
        output='screen'
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

    

    start_lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sllidar_launch),
        launch_arguments={
            'serial_port': LaunchConfiguration('lidar_port'), 
            'serial_baudrate': LaunchConfiguration('lidar_baud'), 
            'frame_id': 'laser',
            'inverted': 'false'
        }.items()
    )

    start_rover = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rover_launch)
    )

    def _start_probe(_ctx, *a, **k):
        return [LogInfo(msg='[XRCE] Agent detected, running PX4 readiness probe...'), px4_probe]

    def _start_stack(_ctx, *a, **k):
        return [LogInfo(msg='[XRCE] PX4 ready. Launching LiDAR + rover...'), start_lidar, start_rover]


    def _start_stack_once(context, *a, **k):
        # run only once even if multiple handlers fire
        started = LaunchConfiguration('stack_started').perform(context)
        if started == '1':
            return []
        # mark as started
        return [
            SetLaunchConfiguration('stack_started', '1'),
            LogInfo(msg=f'[XRCE] PX4 ready. Launching LiDAR + rover...'),
            # bring up relays/bridge if you have them, then the rest:
            start_lidar,
            start_rover,
        ]
    
    # 1) When Agent prints the ready marker, run the probe
    on_agent_seen = RegisterEventHandler(
        OnProcessIO(
            target_action=agent_proc,
            on_stdout=lambda ev: [OpaqueFunction(function=_start_probe)]
                if XRCE_READY_MARKER in ev.text.decode(errors='ignore') else [],
            on_stderr=lambda ev: [OpaqueFunction(function=_start_probe)]
                if XRCE_READY_MARKER in ev.text.decode(errors='ignore') else [],
        )
    )

    # Fire the rest of the stack as soon as the probe prints the READY line
    on_probe_ready = RegisterEventHandler(
        OnProcessIO(
            target_action=px4_probe,
            on_stdout=lambda ev: [OpaqueFunction(function=_start_stack_once)]
                if PROBE_READY_MARKER in ev.text.decode(errors='ignore') else [],
            on_stderr=lambda ev: [OpaqueFunction(function=_start_stack_once)]
                if PROBE_READY_MARKER in ev.text.decode(errors='ignore') else [],
        )
    )

    # Optional: if Agent dies, bring down launch
    on_agent_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=agent_proc,
            on_exit=[LogInfo(msg='[XRCE] Agent exited — shutting down.')]
        )
    )

    
    return LaunchDescription([
        xrce_dev_arg, xrce_baud_arg, xrce_ns_arg,
        lidar_port_arg, lidar_baud_arg,
        init_flag,
        agent_proc,
        on_agent_seen,
        on_probe_ready,     # <--- NEW: react to probe's READY log
        on_agent_exit,
    ])
