import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_share = get_package_share_directory('drone_description')

    xacro_file = os.path.join(
        pkg_share,
        'urdf',
        'drone_urdf.xacro'
    )
    bridge_config = os.path.join(
        pkg_share,
        'config',
        'bridge.yaml'
    )
    octomap_config = os.path.join(
        pkg_share,
        'config',
        'octomap_params.yaml'
    )
    world_file = os.path.join(
        pkg_share,
        'worlds',
        'baylands_dynamic.sdf'
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

    # NOTE: robot_state_publisher is kept purely for TF (it needs
    # robot_description regardless of how the model actually gets into
    # Gazebo). The drone itself is no longer spawned dynamically — it's
    # now embedded directly in baylands_dynamic.sdf (same as the
    # moving_box_N models), because gpu_lidar sensors on models spawned
    # at runtime via `ros_gz_sim create` were never publishing on this
    # system (0 publishers, no errors, across every config variation
    # tested), while gpu_lidar sensors present in the world file from
    # load — like the boxes' — worked every time. If you ever need to
    # change the drone's spawn pose, edit the <pose> inside
    # baylands_dynamic.sdf's <model name='drone'> block directly.
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

    # Publishes odom -> base_link from /model/drone/odometry (PX4's EKF2
    # estimate, republished by px4_offboard_bridge). Nothing
    # else in this stack publishes this transform (robot_state_publisher
    # only does the URDF's fixed joints), and octomap_server / RViz2
    # both need it to place the lidar cloud in the odom frame.
    # use_sim_time is required here (and on colorizer/octomap_server
    # below) because Gazebo publishes /clock and stamps every sensor
    # message + TF transform with sim time — a node whose own clock is
    # still wall time will see those stamps as impossibly stale/future
    # and TF lookups will fail silently.
    odom_tf_broadcaster = Node(
        package='drone_description',
        executable='odom_tf_broadcaster',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # The gpu_lidar's PointCloud2 header uses Gazebo's internal scoped
    # sensor name ('drone/base_footprint/lidar_sensor') as its frame_id,
    # not the URDF's 'lidar_link' — ros_gz_bridge passes it through
    # as-is (confirmed via `ros2 topic echo /model/drone/scan/points
    # --field header.frame_id`). octomap_server can't transform the
    # cloud without this frame existing in TF, so bridge it with an
    # identity static transform (the sensor is rigidly at lidar_link's
    # origin, so 0 offset/0 rotation is correct). The colorized cloud
    # below is republished under this same frame_id, so this bridge
    # covers both the raw and colorized point clouds.
    lidar_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lidar_frame_bridge',
        arguments=[
            '0', '0', '0', '0', '0', '0',
            'lidar_link',
            'drone/base_footprint/lidar_sensor'
        ],
        output='screen'
    )

    # Same problem as the lidar, on the camera side: Gazebo's camera
    # sensor publishes Image/CameraInfo headers with its own internal
    # scoped sensor name as frame_id (confirmed via `ros2 topic echo
    # /model/drone/camera --field header.frame_id --once`), not any
    # frame robot_state_publisher knows about from the URDF. Without
    # this bridge, lidar_camera_colorizer's tf_buffer.lookup_transform()
    # call (image_frame <- lidar_frame) throws on every callback —
    # caught by its own `except Exception`, logged only via a throttled
    # warn ("TF lookup failed, skipping frame") that's easy to miss —
    # and points_colored never gets published as a result.
    #
    # Bridged to 'camera_link_optical', not 'camera_link': the xacro's
    # camera_to_optical fixed joint (rpy = -1.5708 0 -1.5708) already
    # rotates into the optical convention (X-right, Y-down, Z-forward)
    # that Gazebo's camera plugin publishes in, and that whole chain
    # (base_footprint -> base_link -> camera_gimbal_link -> camera_link
    # -> camera_link_optical) is already published by
    # robot_state_publisher from the URDF. So, exactly like the lidar
    # bridge, this only needs to be an identity transform onto whichever
    # frame_id Gazebo actually uses for the sensor.
    #
    # REPLACE 'drone/base_footprint/camera_sensor' below with the exact string
    # printed by the echo command above before running this launch file.
    camera_frame_bridge = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_frame_bridge',
        arguments=[
            '0', '0', '0', '0', '0', '0',
            'camera_link_optical',
            'drone/base_footprint/camera_sensor'
        ],
        output='screen'
    )

    # Fuses /model/drone/scan/points (lidar geometry) with
    # /model/drone/camera (RGB) + /model/drone/camera_info (intrinsics)
    # into a colored cloud on /model/drone/scan/points_colored, so the
    # resulting octomap's colors match the real scene, not just its
    # occupancy shape. See lidar_camera_colorizer.py for the projection
    # logic. use_sim_time=True — see note on odom_tf_broadcaster above;
    # this node's TF lookups were failing silently on every callback
    # until this was set.
    colorizer = Node(
        package='drone_description',
        executable='lidar_camera_colorizer',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Consumes /model/drone/scan/points_colored and builds a colored 3D
    # occupancy map in the odom frame (no 'map' frame exists in this
    # stack — odom is PX4's EKF2 estimate). Uses the color-tree executable
    # (confirmed available via `ros2 pkg executables octomap_server`)
    # instead of the plain octomap_server_node used before colorization
    # was added. use_sim_time=True for the same reason as the nodes
    # above.
    #
    # OCTOMAP_INTEGRATION_HZ: the drone_urdf.xacro gpu_lidar sensor runs
    # at update_rate=10 (10 Hz), and every one of those frames used to go
    # straight into octomap_server. That matters because octomap_server
    # has no incremental-publish mode — every single integration
    # re-walks and republishes the ENTIRE occupied-voxel set on
    # 'octomap_point_cloud_centers' (-> /map/points), not just what
    # changed. At 10 Hz, that's the full map, fully re-serialized and
    # fully re-sent over DDS, ten times a second, regardless of how much
    # (if anything) actually changed between frames — a cost that grows
    # with map size and is paid by octomap_server itself, by
    # semantic_map_node.py, and by drone_gui.py, all at once, every time.
    # Dropping the integration rate to 3 Hz via topic_tools' throttle
    # node cuts that redundant work by ~70% for a mapping-completeness
    # cost that's negligible IF the drone can't move far between
    # integrations.
    #
    # SPEED BUGFIX: that assumption broke once the vehicle could fly
    # ~6 m/s. At 3 Hz (dt=0.33s), the drone
    # can now cover ~2m between integrations — well past octomap's 0.2m
    # voxel resolution, meaning real gaps/holes in the map at speed, not
    # just a negligible completeness cost. Raised back to the sensor's
    # native 10 Hz (dt=0.1s -> ~0.6m of travel between frames at top
    # speed) per this comment's own original suggestion. This restores
    # the full network/CPU cost this throttle existed to cut — if that
    # becomes a bottleneck again at this speed, the fix is a smarter
    # throttle (e.g. distance-triggered integration instead of a fixed
    # rate) rather than dropping the rate back down and reopening this
    # same gap.
    OCTOMAP_INTEGRATION_HZ = 10.0
    points_colored_throttle = Node(
        package='topic_tools',
        executable='throttle',
        name='points_colored_throttle',
        arguments=[
            'messages',
            '/model/drone/scan/points_colored',
            str(OCTOMAP_INTEGRATION_HZ),
            '/model/drone/scan/points_colored_throttled',
        ],
        output='screen'
    )

    octomap_server = Node(
        package='octomap_server',
        executable='color_octomap_server_node',
        name='octomap_server_node',
        output='screen',
        parameters=[octomap_config, {'use_sim_time': True}],
        remappings=[
            # Fed from the throttled topic above, not directly from the
            # colorizer's raw output — see points_colored_throttle.
            ('cloud_in', '/model/drone/scan/points_colored_throttled'),
            # octomap_server's default OUTPUT topic is
            # 'octomap_point_cloud_centers' (that's what RViz was
            # subscribed to directly, which is why RViz's map updated
            # fine). Nothing was remapping that to '/map/points', which
            # is the topic drone_gui.py and semantic_map_node.py actually
            # subscribe to -- so '/map/points' had zero publishers and
            # the GUI's map tab / semantic_map_node just sat empty.
            ('octomap_point_cloud_centers', '/map/points'),
        ]
    )

    # ---- PX4 SITL + uXRCE-DDS agent -------------------------------------
    # PX4 is the flight controller / estimator / battery model. It runs
    # in STANDALONE mode: Gazebo already loads the drone from
    # baylands_dynamic.sdf, so PX4 must attach to it instead of spawning.
    px4_dir = LaunchConfiguration('px4_dir')
    px4_autostart = LaunchConfiguration('px4_autostart')

    xrce_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen',
    )

    # Delayed so Gazebo (and the drone's sensors) are up before PX4 looks
    # for them; raise if PX4 logs "gz topic not found" on a slow machine.
    px4_sitl = TimerAction(
        period=12.0,
        actions=[ExecuteProcess(
            cmd=[[px4_dir, '/build/px4_sitl_default/bin/px4']],
            cwd=[px4_dir, '/build/px4_sitl_default'],
            additional_env={
                'PX4_GZ_STANDALONE': '1',
                'PX4_SYS_AUTOSTART': px4_autostart,
                'PX4_SIMULATOR': 'gz',
                'PX4_SIM_MODEL': 'gz_drone',
                'PX4_GZ_MODEL_NAME': 'drone',
                'PX4_GZ_WORLD': 'baylands',
            },
            output='screen',
        )],
    )

    px4_offboard_bridge = Node(
        package='drone_description',
        executable='px4_offboard_bridge',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Mission manager: drives the 'mission' flight mode with two types:
    #   - waypoints: fly an ordered list of (x, y, z) world-frame points
    #   - explore: frontier-based autonomous exploration (grid of visited/occupied/unknown)
    # Always publishes /drone/mission_cmd regardless of active mode;
    # px4_offboard_bridge decides whether to obey it. See mission_manager.py.
    mission_manager = Node(
        package='drone_description',
        executable='mission_manager',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Semantic map node: projects object_tracker's 2D detections into the
    # lidar's 3D points (via /model/drone/camera_info) to tag octomap
    # voxels with semantic labels instead of just occupied/free. Depends on
    # object_tracker actually running to have anything to project — with no
    # detections it just idles harmlessly. See semantic_map_node.py.
    semantic_map_node = Node(
        package='drone_description',
        executable='semantic_map_node',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # Object tracker: camera-based detector/tracker driving 'auto' control
    # mode (see keyboard_teleop's 't' toggle) via /drone/tracker_cmd, and
    # feeding 2D detections to semantic_map_node for 3D localization. Reads
    # /model/drone/camera; idles harmlessly with no detections if nothing's
    # in frame. See object_tracker.py.
    object_tracker = Node(
        package='drone_description',
        executable='object_tracker',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )

    # REVERTED: fixed_target_1..3 are static again (baylands_dynamic.sdf's
    # <static>true</static>, no obstacle_lidar, no VelocityControl -- see
    # the comment above fixed_target_1 in that file). They intentionally
    # have no entry here — no lidar bridge, no patrol node — same as
    # before they were briefly made to patrol. object_tracker's HSV
    # detector and semantic_map_node's /map/targets_3d_fixed localization
    # both work off camera + lidar/map data that's already bridged/
    # launched below; a stationary box needs nothing extra wired up to be
    # found and marked permanently on the map. They did keep their bigger
    # (4x) size and their 30m-spaced lane y-positions from that change.

    # There are 4 patrolling boxes in baylands_dynamic.sdf (moving_box_1..4).
    # All 4 now drive with yaw=0 and direction=+1.0, so all 4 move in the
    # SAME world-frame direction (+x), each in its own lane spaced 30m
    # apart in y — see the ONE-DIRECTION / LANE LAYOUT comment above
    # moving_box_1 in the world file. Both their lidar bridges (Gazebo ->
    # ROS, forward-facing scan used for obstacle avoidance) AND the
    # moving_object_patrol nodes that actually drive them are brought up
    # here, one of each per box, driven off the same box_configs list so
    # speed/direction stay in one place.
    box_configs = [
        {'model_name': 'moving_box_1', 'speed': 0.5, 'direction': 1.0},
        {'model_name': 'moving_box_2', 'speed': 0.4, 'direction': 1.0},
        {'model_name': 'moving_box_3', 'speed': 0.6, 'direction': 1.0},
        {'model_name': 'moving_box_4', 'speed': 0.45, 'direction': 1.0},
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

    # name= is set per-instance here (unlike the manual `ros2 run` commands
    # in the old bottom-of-file note) specifically because a launch file —
    # unlike 4 separate terminals — runs all 4 in the same process tree, so
    # they'd otherwise all register as the same default 'moving_object_patrol'
    # node name and `ros2 node list`/`ros2 topic info` couldn't tell them
    # apart. Each still publishes/subscribes its own /model/<model_name>/...
    # topics via the model_name parameter regardless of node name.
    box_patrol_nodes = [
        Node(
            package='drone_description',
            executable='moving_object_patrol',
            name=f"moving_object_patrol_{cfg['model_name']}",
            output='screen',
            parameters=[{
                'model_name': cfg['model_name'],
                'speed': cfg['speed'],
                'direction': cfg['direction'],
            }]
        )
        for cfg in box_configs
    ]

    # ---- Gazebo GUI follow camera ----------------------------------------
    # Makes the Gazebo GUI camera follow the 'drone' model automatically,
    # same as right-click -> Follow in the Entity tree. Delayed until after
    # PX4 has attached (px4_sitl starts at 12 s) so the model definitely
    # exists; if /gui/follow prints 'data: false' on a slow machine, raise
    # follow_delay_sec. Override the camera position relative to the drone
    # with e.g. follow_offset:="-6 0 3" (x y z, meters, world axes).
    follow_delay = LaunchConfiguration('follow_delay_sec')
    follow_offset = LaunchConfiguration('follow_offset')

    follow_drone = TimerAction(
        period=follow_delay,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'service', '-s', '/gui/follow',
                     '--reqtype', 'gz.msgs.StringMsg',
                     '--reptype', 'gz.msgs.Boolean',
                     '--timeout', '2000',
                     '--req', 'data: "drone"'],
                output='screen',
            ),
            ExecuteProcess(
                cmd=['bash', '-c', [
                    'read X Y Z <<< "', follow_offset, '"; ',
                    'gz service -s /gui/follow/offset '
                    '--reqtype gz.msgs.Vector3d --reptype gz.msgs.Boolean '
                    '--timeout 2000 --req "x: $X, y: $Y, z: $Z"',
                ]],
                output='screen',
            ),
        ],
    )

    return LaunchDescription([
        DeclareLaunchArgument('follow_delay_sec', default_value='15.0'),
        DeclareLaunchArgument('follow_offset', default_value='-4 0 2'),
        DeclareLaunchArgument('px4_dir', default_value=os.path.expanduser('~/PX4-Autopilot')),
        DeclareLaunchArgument('px4_autostart', default_value='4901'),
        gazebo,
        xrce_agent,
        px4_sitl,
        px4_offboard_bridge,
        robot_state_publisher,
        bridge,
        odom_tf_broadcaster,
        lidar_frame_bridge,
        camera_frame_bridge,
        colorizer,
        points_colored_throttle,
        octomap_server,
        semantic_map_node,
        object_tracker,
        *box_lidar_bridges,
        *box_patrol_nodes,
        follow_drone,
    ])


# The moving_object_patrol nodes for all 4 boxes launch automatically as
# part of this file (see box_patrol_nodes above) with distinct node names
# (moving_object_patrol_moving_box_1..4) so `ros2 node list` / `ros2 topic
# info` can tell them apart. object_tracker is also launched here now.
# Only keyboard_teleop (needs raw terminal focus, doesn't work launched)
# and drone_gui (Qt window, needs a display — run manually with
# `ros2 run drone_description drone_gui` when you have one and want it)
# still need to be run by hand.
