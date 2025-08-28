from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def rover(child_path, ns, sid, lfid, mpf):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(child_path),
        launch_arguments={
            'ns_rover': ns,
            'sys_id': str(sid),         # pass text; child can cast to int if needed
            'lidar_frame_id': lfid,
            'mission_param_file': mpf,
        }.items()
    )

def generate_launch_description():
    child_path = PathJoinSubstitution([
        FindPackageShare('asl_rover_offboard'),
        'launch', 'offboard_sitl.launch.py'
    ])

    return LaunchDescription([
        rover(child_path, 
              ns='px4_1', 
              sid=2, 
              lfid='asl_rover_1/rplidar_a2/link/gpu_lidar',
              mpf='utari_demo_rover1_params.yaml'),

        rover(child_path, 
              ns='px4_2', 
              sid=3, 
              lfid='asl_rover_2/rplidar_a2/link/gpu_lidar',
              mpf='utari_demo_rover2_params.yaml'),
    ])
