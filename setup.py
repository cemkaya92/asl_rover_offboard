from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'asl_rover_offboard'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),  # ✅ This installs the marker
        ('share/' + package_name, ['package.xml']),
        # Include launch and config files
        (os.path.join('share', package_name, 'launch'), glob('asl_rover_offboard/launch/*.py')),
        (os.path.join('share', package_name, 'config', 'sitl'), glob('asl_rover_offboard/config/sitl/*.yaml')),
        (os.path.join('share', package_name, 'config', 'controller'), glob('asl_rover_offboard/config/controller/*.yaml')),
        (os.path.join('share', package_name, 'config', 'vehicle_parameters'), glob('asl_rover_offboard/config/vehicle_parameters/*.yaml')),
        (os.path.join('share', package_name, 'model', 'urdf'), glob('asl_rover_offboard/model/urdf/*.urdf')),
        (os.path.join('share', package_name, 'model', 'urdf','meshes'), glob('asl_rover_offboard/model/urdf/meshes/*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='asl-simulation',
    maintainer_email='uluhancem.kaya@uta.edu',
    description='ROS 2 PX4 control system with Backstepping Control and Offboard modes.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'motor_commander = asl_rover_offboard.control.motor_commander:main',
            'backstepping_controller = asl_rover_offboard.control.backstepping_controller:main',
        ],
    },
)
