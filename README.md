# drone_sim_px4_ros2: PX4 + ROS 2 + Gazebo quadrotor with lidar mapping and object tracking

A complete quadrotor simulation in the Gazebo **Baylands** world. **PX4 SITL** flies the drone (estimator, controllers, motor mixing). **ROS 2** builds a colored 3D map from lidar and camera, detects and tracks moving targets, and provides manual and tracking control modes, plus a live Qt GUI.

```
                 Gazebo (Baylands world, drone, 4 patrolling boxes)
                   |  camera / IMU / 3D lidar             ^ motor commands
                   v                                       |
            ros_gz_bridge                            PX4 SITL (EKF2, controllers)
                   |                                       ^  |
                   |                          uXRCE-DDS (px4_msgs, UDP 8888)
                   |                                       |  v
  lidar_camera_colorizer --> octomap_server        px4_offboard_bridge
                   |                |                      ^       |
                   |                v                      |       v
                   |        semantic_map_node <-- object_tracker   /model/drone/odometry
                   |                |                      ^
                   +----------------+--> drone_gui        |
                                                       keyboard_teleop
```

## Contents

1. [Features](#features)
2. [Repository layout](#repository-layout)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Running the simulation](#running-the-simulation)
6. [Controls](#controls)
7. [Selecting a tracking target](#selecting-a-tracking-target)
8. [Key topics](#key-topics)
9. [Configuration](#configuration)
10. [Troubleshooting](#troubleshooting)
11. [License](#license)

## Features

- **PX4 SITL flight stack.** EKF2 state estimation, cascaded controllers and motor mixing run inside PX4. `px4_offboard_bridge` sends velocity setpoints over uXRCE-DDS and republishes PX4's estimate on `/model/drone/odometry`, so the rest of the stack has one odometry source.
- **Custom airframe** (`px4/4901_gz_drone`): 4S 10000 mAh quad, 12" props, about 2.56 kg, with rotor geometry and thrust constants matched to the Gazebo model.
- **Sensors.** Forward camera with `camera_info`, IMU, and a Livox Mid-360-style 3D lidar (360 x 59 deg, 40 rings, 40 m range).
- **Colored point cloud.** `lidar_camera_colorizer` projects every lidar point into the camera image to color it.
- **3D mapping.** A colored OctoMap (0.2 m voxels) from the lidar, published in the `odom` frame.
- **Object detection and tracking.** `object_tracker` finds targets in the camera feed and keeps the chosen one centered at a set standoff distance. Detector backends: HSV color threshold (default), or MobileNet-SSD through OpenCV's DNN module (optional, needs model files). Tracks keep stable ids and can be re-identified after being lost.
- **Semantic map.** `semantic_map_node` matches 2D detections to 3D lidar points, giving each target a real world position. Far targets beyond lidar range use a ground-plane ray fallback. Moving and fixed targets are separated automatically.
- **Moving targets.** Four patrolling vehicles with lidar-based obstacle avoidance (`moving_object_patrol`).
- **Two control modes.** Manual keyboard flight and autonomous target tracking.
- **Qt GUI.** `drone_gui` shows telemetry, battery, camera, a GPU-rendered 3D map, and a target list you can click to switch tracking.

## Repository layout

```
drone_sim_px4_ros2/
├── README.md
├── drone.repos                  external ROS repos (px4_msgs), fetched with vcstool
├── px4/
│   └── 4901_gz_drone            PX4 airframe, copied into PX4-Autopilot (step 4)
└── drone_description/           the ROS 2 package
    ├── launch/                  spawn_drone_dynamic.launch.py (main), display.launch.py,
    │                            spawn_drone_dynamic_1box_test.launch.py
    ├── urdf/drone_urdf.xacro    robot description (fixed joints, TF)
    ├── worlds/                  baylands_dynamic.sdf (world + drone + boxes)
    ├── config/                  bridge.yaml, octomap_params.yaml
    └── drone_description/       Python nodes
```

## Requirements

- Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic. Tested on Ubuntu 24.04.3 with Gazebo Sim 8.11.0. Other combinations may work but are untested.
- About 15 GB free disk space and a machine with a working GPU and OpenGL drivers.
- 16 GB RAM recommended. With about 8 GB, add swap before building PX4 and build with fewer jobs (`make px4_sitl_default -j2`).
- Internet access on first launch. The Baylands world and water model are downloaded from Gazebo Fuel and cached.
- PX4-Autopilot at commit `b3e343c153` (a `main` build after `v1.18.0-beta1`) with `px4_msgs` at commit `f42fb1e94efb3c23edc275deac5b7dad0cce638b`, which is pinned in `drone.repos`. This is the combination the project was tested with. Keep these two in step, or the build fails on unknown message fields.

## Installation

Run everything below in order. Steps 1 to 4 are one-time.

### 1. ROS 2 Jazzy and system packages

Install ROS 2 Jazzy first by following the official guide: <https://docs.ros.org/en/jazzy/Installation.html>. Then:

```bash
sudo apt update
sudo apt install -y git curl python3-pip python3-vcstool python3-colcon-common-extensions \
  python3-rosdep ros-jazzy-ros-gz ros-jazzy-octomap-server ros-jazzy-topic-tools \
  ros-jazzy-xacro ros-jazzy-robot-state-publisher ros-jazzy-cv-bridge \
  ros-jazzy-message-filters ros-jazzy-tf2-ros \
  python3-pyqt5 python3-pyqtgraph python3-opengl python3-opencv python3-matplotlib python3-numpy

sudo rosdep init 2>/dev/null || true
rosdep update
```

The launch file starts `color_octomap_server_node` from the `octomap_server` package. Check that it is installed:

```bash
ros2 pkg executables octomap_server | grep color_octomap_server_node
```

If nothing is printed, see [Troubleshooting](#troubleshooting).

### 2. Micro XRCE-DDS Agent (choose a version)

The agent connects PX4 to ROS 2. **The version matters.** Pick one of the two options below and build it from its release tag, never from the default branch. In both cases use the default PX4 build from step 3 (do not enable `UXRCE_DDS_CLIENT_USE_DDS_V3`).

| | Option A: `v2.4.3` | Option B: `v3.0.2` |
|---|---|---|
| Basis | The version PX4 documents for ROS 2 Jazzy | The version this project was developed and run with |
| Choose it if | You want the documented pairing. Pick this if you are unsure. | You want to match the author's setup exactly |

```bash
AGENT_VERSION=2.4.3        # Option A. For Option B use: AGENT_VERSION=3.0.2
cd ~
git clone -b v$AGENT_VERSION https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd Micro-XRCE-DDS-Agent
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib/
MicroXRCEAgent --help                                # should print usage
ls /usr/local/lib | grep libmicroxrcedds_agent.so.   # the last entry ends with the installed version
```

Notes:

- PX4 lists `v2.4.3` for Jazzy and describes `v2.x` and `v3.x` as separate pairings (`v3.x` is listed for newer ROS 2 releases, together with a PX4 build option). Option B runs `v3.0.2` against the default PX4 build from step 3, which is how the author ran it, but it is not the pairing PX4 documents. If no `/fmu/out/...` topics appear in ROS, use Option A.
- To switch versions later, delete `~/Micro-XRCE-DDS-Agent`, repeat the block above with the other `AGENT_VERSION`, and run `sudo ldconfig /usr/local/lib/`.

Source: <https://docs.px4.io/main/en/middleware/uxrce_dds> (section "Version selection").

### 3. PX4-Autopilot (pinned commit)

This project was developed and tested against a specific PX4 `main` commit, so the steps below check out exactly that one. The launch file expects PX4 in `~/PX4-Autopilot` (override with `px4_dir:=...`).

```bash
cd ~
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
git checkout b3e343c1532fea4608cd373da3dcdb0fcc3fa763
git submodule update --init --recursive
git describe --tags                 # should print v1.18.0-beta1-834-gb3e343c153
bash ./Tools/setup/ubuntu.sh        # installs PX4 dependencies (log out and in afterwards if it asks)
```

PX4 `main` changes constantly. If you use a different commit or a release tag, use the `px4_msgs` commit that matches it, and expect to adjust field names in `px4_offboard_bridge.py` if the build complains.

### 4. Get this repo, then install the airframe into PX4

```bash
mkdir -p ~/drone_ws/src
cd ~/drone_ws/src
git clone https://github.com/mohamedkhaled221201/drone_sim_px4_ros2.git
```

Copy the airframe and register it:

```bash
AF=~/PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes
cp ~/drone_ws/src/drone_sim_px4_ros2/px4/4901_gz_drone $AF/
grep -q 4901_gz_drone $AF/CMakeLists.txt || sed -i '/4001_gz_x500/a\    4901_gz_drone' $AF/CMakeLists.txt
grep -n 4901_gz_drone $AF/CMakeLists.txt
```

The last command must print one line with `4901_gz_drone`. If it prints nothing, open `$AF/CMakeLists.txt` and add `4901_gz_drone` to the list of airframe files by hand, next to the other `gz` entries (for example below `4001_gz_x500`).

Build PX4 SITL (first build takes several minutes):

```bash
cd ~/PX4-Autopilot
make px4_sitl_default
```

If PX4 tries to open a Gazebo window and simulation, close it with `Ctrl+C`. You only need the build. The ROS launch file starts PX4 itself.

### 5. Fetch px4_msgs and build the workspace

```bash
cd ~/drone_ws/src
vcs import . < drone_sim_px4_ros2/drone.repos       # clones px4_msgs at the pinned commit

cd ~/drone_ws
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

Add this to `~/.bashrc` so every new terminal is ready (run it once, after the build):

```bash
grep -qxF "source /opt/ros/jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
grep -qxF "source ~/drone_ws/install/setup.bash" ~/.bashrc || echo "source ~/drone_ws/install/setup.bash" >> ~/.bashrc
```

## Running the simulation

Use a separate terminal for each numbered item. Every terminal needs the workspace sourced (already done if you edited `~/.bashrc`).

### Terminal 1: launch everything

```bash
ros2 launch drone_description spawn_drone_dynamic.launch.py
```

This starts Gazebo with the Baylands world, the XRCE agent, PX4 SITL (12 s after Gazebo), the ROS bridges, the mapping and tracking nodes, the four patrolling boxes and a chase camera. The first run downloads the world from Fuel and can take a few minutes. Wait until PX4 finishes booting and the drone appears in the world.

Launch arguments:

| Argument | Default | Meaning |
|---|---|---|
| `px4_dir` | `~/PX4-Autopilot` | Where PX4 is built |
| `px4_autostart` | `4901` | PX4 airframe id |
| `follow_delay_sec` | `15.0` | Delay before the GUI camera follows the drone |
| `follow_offset` | `-4 0 2` | Follow-camera offset behind and above the drone |

Example: `ros2 launch drone_description spawn_drone_dynamic.launch.py px4_dir:=/opt/PX4-Autopilot`

### Terminal 2: keyboard teleop

```bash
ros2 run drone_description keyboard_teleop
```

This terminal must have focus, not Gazebo. See [Controls](#controls).

### Terminal 3 (optional): telemetry GUI

```bash
ros2 run drone_description drone_gui
```

Needs a display. Shows telemetry, battery, camera, the live 3D map and the target list.

### Quick checks

```bash
ros2 topic list | grep -E "model/drone|drone/|map/|fmu/out"
ros2 topic hz /model/drone/odometry          # PX4 estimate, should be steady
ros2 topic hz /model/drone/scan/points       # lidar, about 10 Hz
ros2 topic hz /model/drone/scan/points_colored
ros2 topic hz /map/points                    # octomap output
```

Everything above should print a rate. `/fmu/out/vehicle_odometry` appearing in the topic list means PX4 and the XRCE agent are connected.

The drone arms and switches to offboard by itself once PX4 is ready. In the launch log, PX4 prints `Ready for takeoff!` and then `px4_offboard_bridge` prints `PX4 confirmed OFFBOARD + ARMED.` You do not need to press anything for that. To confirm the drone flies, click the teleop terminal and hold `space`: the drone climbs.

## Controls

Click the teleop terminal, then:

| Key | Action | Key | Action |
|---|---|---|---|
| `w` / `x` | forward / back | `q` / `e` | yaw left / right |
| `a` / `d` | left / right | `t` | toggle auto tracking |
| `space` / `z` | up / down | | |

- **manual**: your keys fly the drone. Holding two keys blends them into one steady diagonal.
- **auto**: `object_tracker` flies, keeping the selected target centered at its standoff distance. Press `t` again to return to manual.

Manual keys are ignored in auto mode. The current mode is published on `/drone/control_mode`. The drone arms and switches to offboard automatically once PX4 is ready (see [Quick checks](#quick-checks)).

## Selecting a tracking target

In `auto` mode, target `-1` (the default) picks the nearest moving target. To lock onto one, click it in the `drone_gui` target list, or publish its id:

```bash
ros2 topic pub --once /drone/selected_target_id std_msgs/msg/Int32 "{data: 2}"
# back to automatic selection:
ros2 topic pub --once /drone/selected_target_id std_msgs/msg/Int32 "{data: -1}"
```

## Key topics

| Topic | Type | Description |
|---|---|---|
| `/model/drone/odometry` | `nav_msgs/Odometry` | PX4 EKF2 estimate (from `px4_offboard_bridge`) |
| `/model/drone/camera` | `sensor_msgs/Image` | Forward RGB camera |
| `/model/drone/camera_info` | `sensor_msgs/CameraInfo` | Camera intrinsics |
| `/model/drone/imu` | `sensor_msgs/Imu` | IMU |
| `/model/drone/scan/points` | `sensor_msgs/PointCloud2` | Raw lidar cloud |
| `/model/drone/scan/points_colored` | `sensor_msgs/PointCloud2` | Camera-colored lidar cloud |
| `/map/points` | `sensor_msgs/PointCloud2` | OctoMap occupied voxels |
| `/map/targets_3d`, `_fixed`, `_moving` | `Float32MultiArray` | Target world positions `[id, x, y, z, ...]` |
| `/drone/control_mode` | `std_msgs/String` | `manual` or `auto` |
| `/drone/teleop_cmd` | `geometry_msgs/Twist` | Keyboard command |
| `/drone/tracker_cmd` | `geometry_msgs/Twist` | Tracker command |
| `/drone/selected_target_id` | `std_msgs/Int32` | Target to track (`-1` = auto) |

## Configuration

- **`config/bridge.yaml`**: Gazebo to ROS topic bridges.
- **`config/octomap_params.yaml`**: map resolution (0.2 m), sensor range (40 m, matched to the lidar model), height bounds.
- **`object_tracker` parameters**: `detector_backend` (`hsv`, `dnn` or `auto`), standoff distance and altitude, track expiry. Set them in the launch file or with `ros2 param set`.
- **DNN detector (optional)**: put the MobileNet-SSD Caffe files (`deploy.prototxt` and `.caffemodel`) somewhere on disk and point the detector's `model_path` and `config_path` at them. Without them the tracker falls back to HSV automatically.
- **Patrolling boxes**: speeds and directions are in the `box_configs` list in the launch file.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `colcon build` fails on an unknown `px4_msgs` field | `px4_msgs` and PX4 versions differ. Use the PX4 commit from step 3 and the `px4_msgs` commit in `drone.repos`, then rebuild. |
| PX4 says the model or airframe is not found | `4901_gz_drone` was not copied, not added to `CMakeLists.txt`, or PX4 was not rebuilt (step 4). |
| PX4 logs `gz topic not found` | PX4 started before Gazebo finished loading. Raise the `period` of the PX4 `TimerAction` in the launch file above 12 s. |
| Gazebo hangs on first start | The world is being downloaded from Fuel. Wait, and check your internet connection. |
| `MicroXRCEAgent: command not found` | Redo step 2 and run `sudo ldconfig /usr/local/lib/`. |
| PX4 starts but no `/fmu/out/...` topics appear in ROS | Agent and PX4 client mismatch. Rebuild the agent from a release tag as in step 2 (never the default branch), preferably Option A (`v2.4.3`, the pairing PX4 documents for Jazzy), and make sure PX4 was built as in step 3 without `UXRCE_DDS_CLIENT_USE_DDS_V3`. |
| `px4_msgs` build fails with an `empy` / `em` error | `pip install --user -U empy==3.3.4 pyros-genmsg setuptools --break-system-packages`, then rebuild. |
| `points_colored` is empty | `ros_gz_bridge` publishes RELIABLE. Subscribers must use RELIABLE QoS. |
| `color_octomap_server_node` not found | Run `sudo apt install --reinstall ros-jazzy-octomap-server` and check `ros2 pkg executables octomap_server` again. If it is still missing, build the ROS 2 branch of `octomap_mapping` from source inside `~/drone_ws/src`. |
| `octomap_server_node` logs `Nothing to publish, octree is empty`, `Could not open file`, or `Message Filter dropping message` | Normal for the first 30 to 40 seconds, while PX4 boots and the odometry transform does not exist yet. They stop once PX4 confirms OFFBOARD + ARMED. |
| `MicroXRCEAgent` and `gazebo` show `process has died ... exit code -2` when you press `Ctrl+C` | Expected. That is how the launch shuts them down. |
| Drone doesn't move | Check `ros2 topic echo /drone/control_mode`. In `manual`, click the teleop terminal so it has focus. |
| Terminal stuck in raw mode after teleop | Run `reset`. |
| `drone_gui` shows nothing or crashes | It needs a display and OpenGL drivers. Check `python3 -c "import PyQt5, pyqtgraph"`. |

Clean rebuild:

```bash
cd ~/drone_ws && rm -rf build install log
colcon build --symlink-install
```

## License

TODO: choose a license (MIT and Apache-2.0 are common), add a `LICENSE` file, and update the `license` fields in `drone_description/package.xml` and `setup.py`.
