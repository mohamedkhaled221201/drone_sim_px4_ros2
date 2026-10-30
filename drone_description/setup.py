from setuptools import find_packages, setup
from glob import glob

package_name = 'drone_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*.xacro')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohamed.khaled221201',
    maintainer_email='mohamed.khaled221201@gmail.com',
    description='Quadrotor simulation (Gazebo Harmonic) flown by PX4 SITL over uXRCE-DDS',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'moving_object_patrol = drone_description.moving_object_patrol:main',
            'drone_gui = drone_description.drone_gui:main',
            'object_tracker = drone_description.object_tracker:main',
            'semantic_map_node = drone_description.semantic_map_node:main',
            'odom_tf_broadcaster = drone_description.odom_tf_broadcaster:main',
            'lidar_camera_colorizer = drone_description.lidar_camera_colorizer:main',
            'px4_offboard_bridge = drone_description.px4_offboard_bridge:main',
            'keyboard_teleop = drone_description.keyboard_teleop:main',
            # 'object_detector' is a support library imported by object_tracker.py
            # (HSVDetector/DNNDetector/build_detector/ReIdentifier), not a ROS
            # node, so it has no entry point.
        ],
    },
)
