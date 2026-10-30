import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


# DIAGNOSTIC TEST VARIANT — not the normal launch file.
#
# Only difference from spawn_drone_dynamic.launch.py: it loads
# baylands_dynamic_1box_test.sdf (moving_box_2/3/4 commented out) and only
# brings up the box_1 lidar bridge, so moving_box_1 is the only other
# gpu_lidar sensor competing with the drone's for VRAM. Use this to check
# whether the drone's /model/drone/scan starts publishing with less GPU
# memory pressure — if it does, that confirms the earlier VRAM-allocation
# theory (Mesa "Failed to allocate a buffer" error / 0 publishers on
# /model/drone/scan). Not meant to be kept around long-term; once the real
# fix (BIOS UMA size, and/or the reduced lidar resolution in
# drone_urdf.xacro) is in place, go back to the normal launch file with all
# 4 boxes.
#
# Copy this + baylands_dynamic_1box_test.sdf into your package's launch/
# and worlds/ directories respectively, colcon build, then:
#   ros2 launch drone_description spawn_drone_dynamic_1box_test.launch.py
# and in another terminal:
#   gz topic -i -t /model/drone/scan
#   gz topic -e -n 1 -t /model/drone/scan/points


def generate_launch_description():
    pkg_share = get_package_share_directory('drone_description')

    xacro_file = os.path.join(
        pkg_share,
        'urdf',
        'drone.urdf.xacro'
    )
    bridge_config = os.path.join(
        pkg_share,
        'config',
        'bridge.yaml'
    )
    world_file = os.path.join(
        pkg_share,
        'worlds',
        'baylands_dynamic_1box_test.sdf'
    )

    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]),
        value_type=str
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch',
                'gz_sim.launch.py'
            )
        ),
        launch_arguments={
            'gz_args': ['-r ', world_file]
        }.items()
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[
            {
                'robot_description': robot_description
            }
        ]
    )

    spawn_drone = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'drone',
            '-topic', 'robot_description',
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_config}'
        ],
        output='screen'
    )

    # Diagnostic test: only moving_box_1 exists in the loaded world, so this
    # is the only box lidar bridge needed. moving_box_2/3/4 entries removed
    # (not just commented) to keep this file's intent obvious at a glance.
    box_configs = [
        {'model_name': 'moving_box_1', 'speed': 0.5, 'direction': 1.0},
    ]

    box_lidar_bridges = [
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name=f"box_lidar_bridge_{cfg['model_name']}",
            arguments=[
                f"/model/{cfg['model_name']}/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
            ],
            output='screen'
        )
        for cfg in box_configs
    ]

    return LaunchDescription([
        gazebo,
        robot_state_publisher,
        spawn_drone,
        bridge,
        *box_lidar_bridges,
    ])


# Run this manually in its own terminal (after sourcing install/setup.bash):
#
#   ros2 run drone_description moving_object_patrol --ros-args -p model_name:=moving_box_1 -p speed:=0.5 -p direction:=1.0
#
# (moving_box_2/3/4 patrol nodes are not needed for this test — those boxes
# don't exist in baylands_dynamic_1box_test.sdf.)
