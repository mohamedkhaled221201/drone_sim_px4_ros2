"""
PyQt + pyqtgraph rewrite of the drone telemetry GUI.

WHY THIS REWRITE EXISTS
------------------------
Every fix applied earlier in this pass (vectorized point-cloud decode,
splitting the telemetry/map snapshots, skipping no-op redraws) reduced the
COST of feeding data to the map view. None of them could fix the map view
itself, because the limitation was architectural: matplotlib's Axes3D has
no GPU path and no true incremental-update path. Every redraw was
`ax.cla()` — destroy the entire scene (axes, panes, every grid line as its
own artist) — then rebuild it from nothing and re-run Python-level 3D->2D
projection and depth-sorting for every point, on the CPU, synchronously,
in a library that was never built for live data. RViz doesn't share any
of these limits: it's built on OGRE, a real GPU scene-graph renderer,
which keeps the point cloud in a GPU vertex buffer and only touches what
changed.

This file gets the app onto that same footing:
  - PyQt for the window and widgets (QTimer instead of Tkinter's
    `.after()` — same "callback on a timer" model, different toolkit).
  - pyqtgraph.opengl's GLViewWidget for the map: an actual OpenGL scene.
    The point cloud lives in one GLScatterPlotItem whose GPU buffer is
    updated in place via `.setData()` on every redraw, never torn down
    and rebuilt. Camera orbit/pan/zoom is GLViewWidget's own native mouse
    handling — the same category of interaction RViz gives you, running
    through your regular graphics driver (Mesa/amdgpu on Linux — no
    vendor-specific stack needed for an AMD GPU here).

NEW DEPENDENCIES this introduces vs. the old Tkinter file: PyQt5 and
pyqtgraph. Neither needs anything beyond a normal OpenGL-capable driver.
Add them to the package's requirements/rosdep before running this
(e.g. `pip install PyQt5 pyqtgraph --break-system-packages`, or via your
platform's package manager).

WHAT DIDN'T CARRY OVER FROM drone_gui.py, ON PURPOSE
-----------------------------------------------------
- True RViz-grade GPU picking (ray-casting into the GL scene and
  resolving the nearest rendered fragment) isn't implemented — that's
  real work in its own right. Click-to-select IS implemented, but via a
  cheaper approximation: MapView projects each known target's world
  position through the current view/projection matrices to get its
  on-screen position, then picks whichever projected target lands
  closest to the click (see MapView.PICK_RADIUS_PX). This is accurate
  enough for a handful of well-separated target dots but can mis-pick if
  two targets' dots overlap heavily on screen. The sidebar list still
  works exactly as before as a precise fallback either way.
  NOTE: only the tracked mover has a live marker now (see
  LIVE_MARKER_MODES below), so map clicking can only pick that one; use
  the sidebar TARGETS list to switch to a different target.
- Floating "#id" text labels anchored next to each target dot in 3D
  (pyqtgraph.opengl's GLTextItem exists but text billboarding in a GL
  scene has its own edge cases). Targets are still visually distinguished
  by color/size (selected = bright red + larger, others = accent green),
  and every target's id is visible and clickable in the sidebar list.
Both are straightforward to add later if you want them — flagging them
here rather than quietly dropping them.

THE ROS-FACING DroneGuiNode CLASS BELOW IS UNCHANGED FROM drone_gui.py's
DATA-LAYER PASS, EXCEPT the fixed-target callback (see
_map_targets_3d_fixed_cb). It never depended on Tkinter — it's plain rclpy
plus numpy — so it carries over as-is.
"""

import math
import signal
import sys
import threading
import time
from collections import deque

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.signals import SignalHandlerOptions
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState, Imu, Image as RosImage, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float32MultiArray, Int32, String
from px4_msgs.msg import VehicleStatus, ActuatorMotors

# Fast path for decoding PointCloud2 -> numpy: reinterprets the message's
# raw byte buffer directly, no per-point Python loop. Present since
# Foxy/Galactic; fall back to the structured-array path otherwise.
try:
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    _HAVE_READ_POINTS_NUMPY = True
except ImportError:
    _HAVE_READ_POINTS_NUMPY = False

# Map tab redraws on its own slower timer, and only while it's the visible
# tab (see MapView.maybe_redraw) — a live GPU scatter update is cheap
# compared to matplotlib's old cla()-and-rebuild, but there's still no
# reason to touch it faster than it needs to look "alive".
MAP_REDRAW_MS = 100  # 10 Hz. Cheap now: redraws are skipped unless data actually changed

# Matches RViz's PointCloud2 display set to Color Transformer = AxisColor /
# Axis = Z: each point's color comes from where its own Z value falls
# between the cloud's current min and max Z (recomputed every redraw,
# same as RViz's "Autocompute Value Bounds"), not a single flat color.
# CALMER PALETTE: 'turbo' is a rainbow map (blue/cyan/yellow/red, all at
# full saturation) chosen to match RViz's default -- great for maximum
# perceptual separation, bad for calm viewing since everything is loud.
# 'cividis' keeps height information just as readable (perceptually
# uniform, colorblind-safe) via a muted blue->yellow ramp instead, so a
# tracked object's distinct color (ACCENT green / ALARM_BRIGHT red,
# below) stands out against a quieter background instead of competing
# with equally-saturated terrain. Other calm options if this still reads
# too warm: 'cividis' (blue->yellow, recommended), 'bone' (near-grayscale,
# calmest but least height-differentiation), 'gist_earth' (muted natural
# terrain tones).
MAP_POINT_CMAP_NAME = 'cividis'   # only used as a fallback now, see MAP_TERRAIN_COLORS

# CALM TERRAIN RAMP (low -> high): dark warm grey -> sage -> khaki -> pale
# sand. Low saturation, no blue/red/bright-green anywhere in the ramp, so
# every marker colour below (mint / red / sky-blue) is the ONLY saturated
# thing on screen and reads instantly.
MAP_TERRAIN_COLORS = ['#34383d', '#55605c', '#7d8a75', '#a9a685', '#cfc6a3']
MAP_TERRAIN_ALPHA = 0.85
# Height normalisation ignores the lowest/highest 2% of points so a few
# outlier voxels can't squash the whole ramp into one flat colour.
MAP_TERRAIN_PCT = (2.0, 98.0)

# GLScatterPlotItem sizes are in world units (meters) here, not pixels
# (pxMode=False) — matching octomap's own voxel resolution keeps points
# reading as a coherent surface rather than either disappearing gaps or
# blobs that swallow real detail. If you change octomap_params.yaml's
# resolution, adjust this to match.
MAP_POINT_SIZE_M = 0.22
TARGET_POINT_SIZE_M = 1.3
TARGET_POINT_SIZE_SELECTED_M = 1.6
WAYPOINT_POINT_SIZE_M = 1.3
TRAIL_COLOR = (0.0, 0.9, 0.63, 0.9)      # matches ACCENT '#00e5a0'
WAYPOINT_COLOR = (0.30, 0.70, 1.0, 1.0)  # distinct blue — never confused with targets (accent green/red)

# FIXED TARGETS: once a target is confirmed, its marker stays on the map
# forever (see semantic_map_node.py's separate /map/targets_3d_fixed
# topic), even after object_tracker loses sight of it and the LIVE marker
# (scatter_targets, above) disappears. Drawn RED, on request, as the
# permanent "found it here" mark — this is a different, deliberately less
# saturated red than ALARM_BRIGHT below (which flashes on the camera-tab
# alert and highlights whichever LIVE target is currently selected on the
# map), so a glance still tells "actively tracking this one right now"
# (bright, blinking, on scatter_targets) apart from "confirmed here at
# some point" (steady, dimmer, on this separate scatter item). The FIXED
# TARGETS side panel lists each one's exact (x, y, z) next to its dot for
# an unambiguous readout regardless of color.
FIXED_TARGET_COLOR = (0.90, 0.22, 0.22, 1.0)
FIXED_TARGET_COLOR_HEX = '#e63838'  # same red, as a UI string for the side-panel list text

# FIXED-MARKER SYNC (no repeated markers): the node numbers its fixed
# markers with stable ids and never creates two for one object. The GUI
# follows those ids, so a marker the node merges away disappears here too.
# FIXED_MERGE_RADIUS_M matches semantic_map_node's fixed_target_merge_dist_m:
# a marker without a known node id (or from before a node restart) that
# lies within this xy distance of an incoming one is the SAME object and
# just takes the new position instead of becoming a second dot.
FIXED_MERGE_RADIUS_M = 5.0
# The node only ever refines a marker by a few metres. If the marker with
# the same node id jumps farther than this, the node restarted and reused
# its ids: the old markers are kept as permanent legacy entries.
FIXED_NID_MAX_SHIFT_M = 15.0

# LIVE MARKER POLICY: exactly ONE live marker -- the target currently being
# tracked (selected in the TARGETS panel, or the tracker's active target in
# auto-select) -- drawn ONLY while the drone is in 'auto' control mode AND
# semantic_map_node lists that target as MOVING (/map/targets_3d_moving).
# It follows the mover and leaves an amber path behind it. No other
# detected/moving box gets a marker, and in manual/mission mode there is no
# live marker or path at all. Permanent (red) markers are only for
# confirmed FIXED targets and are always drawn.
# (None would mean "every control mode".)
LIVE_MARKER_MODES = ('auto',)
# PATHS (drone flight trail + tracked-mover trail) are recorded and drawn
# ONLY in these control modes; in manual nothing is recorded or shown, and a
# fresh path starts each time auto mode is entered.
TRAIL_MODES = ('auto',)

# PIN MARKERS: a marker never sits on top of the map points. Its foot is a
# small dot snapped to the top of the map points under the target (so it is
# ON the map), a thin stem rises from it, and the head floats above the
# points. The head is small, so it doesn't hide the cloud.
PIN_SNAP_RADIUS_M = 3.5     # map points within this xy radius define the anchor height
PIN_LIFT_M = 3.0            # head height above the anchor
PIN_FOOT_SIZE_M = 0.4
PIN_HEAD_SIZE_M = 1.0
PIN_HEAD_SIZE_SELECTED_M = 1.3
PIN_STEM_WIDTH = 2.0
TRACKED_TRAIL_COLOR = (1.0, 0.72, 0.15, 0.95)   # amber
TRACKED_TRAIL_MAX_POINTS = 150
TRACKED_TRAIL_MIN_STEP_M = 0.25
FIXED_TARGET_POINT_SIZE_M = 1.6  # 2x the old 0.8m -- confirmed targets should read as clearly bigger/permanent than the live tracked dot (TARGET_POINT_SIZE_M=1.0) or a waypoint (1.3), not smaller than both
# Flight trail keeps at most this many recent points (~17 min at the 2 Hz
# it's actually sampled at below — see DroneGuiNode._odom_cb) so a long
# flight's trail buffer can't grow without bound.
TRAIL_MAX_POINTS = 2000
# Sample odometry into the trail at most this often — odometry itself
# arrives at 100Hz (see drone_urdf.xacro's OdometryPublisher), far denser
# than a visual trail needs; sampling instead of appending every message
# keeps TRAIL_MAX_POINTS covering a meaningful flight duration instead of
# the last ~20 seconds.
TRAIL_SAMPLE_INTERVAL_SEC = 0.5

GRID_EXTENT = 30   # fallback extent (meters) before any map data arrives
GRID_COLOR = (0.29, 0.31, 0.37, 0.22)  # ~#4a4e5e, matches the old GRID_COLOR/ALPHA

BG = '#0d0f14'
PANEL = '#151824'
PANEL_BORDER = '#262b3d'
TRACK = '#20232f'
ACCENT = '#00e5a0'
ACCENT_QCOLOR = QtGui.QColor(ACCENT)
NEGATIVE = '#ff5f6d'
NEGATIVE_QCOLOR = QtGui.QColor(NEGATIVE)
ALARM_BRIGHT = '#ff3b30'
ALARM_DIM = '#7a1520'
TEXT = '#eef0f5'
TEXT_DIM = '#7d8296'
TEXT_FAINT = '#4a4e5e'
# PX4 publishes its uXRCE-DDS topics BEST_EFFORT / VOLATILE; a RELIABLE
# subscriber would silently never connect. Same profile drone_diagnostics.py uses.
PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# VehicleStatus.arming_state / nav_state values (px4_msgs).
ARMING_STATE_ARMED = 2
NAV_STATE_OFFBOARD = 14
NAV_STATE_NAMES = {
    0: 'MANUAL', 1: 'ALTCTL', 2: 'POSCTL', 3: 'AUTO_MISSION', 4: 'AUTO_LOITER',
    5: 'AUTO_RTL', 10: 'ACRO', 12: 'DESCEND', 13: 'TERMINATION', 14: 'OFFBOARD',
    15: 'STAB', 17: 'AUTO_TAKEOFF', 18: 'AUTO_LAND',
}
# BatteryStatus.warning values.
BATTERY_WARNING_NAMES = {0: '', 1: 'LOW', 2: 'CRITICAL', 3: 'EMERGENCY', 4: 'FAILED'}
LOW_BATTERY_FRACTION = 0.20
NUM_MOTORS = 4
# Full-scale of the velocity bars; keep >= px4_offboard_bridge's manual_max_speed_xy_mps.
VELOCITY_BAR_MAX_MPS = 6.0


STALE_AFTER_SEC = 0.75
# Battery is published at ~10 Hz by px4_offboard_bridge, but the old PX4
# source was rate-limited to ~1 Hz, which is slower than STALE_AFTER_SEC and
# made the bar blink off between messages. Give it a generous window.
BATTERY_STALE_AFTER_SEC = 3.0


# Same thresholds as PX4's defaults (BAT_LOW_THR / BAT_CRIT_THR / BAT_EMERGEN_THR).
def battery_warning_level(fraction):
    if fraction is None:
        return 0
    if fraction <= 0.05:
        return 3
    if fraction <= 0.07:
        return 2
    if fraction <= 0.15:
        return 1
    return 0

DETECT_STALE_AFTER_SEC = 0.75
BLINK_HZ = 2.0


def quat_to_yaw(x, y, z, w):
    """Yaw (rotation about Z) from a quaternion, radians, atan2 convention."""
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


# ============================================================================
# ROS side. Unchanged from drone_gui.py's data-layer pass — already
# framework-agnostic, so it needed no changes to move into a Qt app.
# (Exception: the fixed-target callback, see below.)
# ============================================================================
class DroneGuiNode(Node):
    """Pure ROS side: subscribes and stores the latest values."""

    def __init__(self):
        super().__init__('drone_gui_node')
        self.lock = threading.Lock()
        self.vx = self.vy = self.vz = 0.0
        self.wx = self.wy = self.wz = 0.0
        self.yaw = 0.0
        # World-frame position, from odometry — the flight trail below is
        # built from these, not from vx/vy/vz (velocities can't be
        # integrated back into a drift-free position independently of
        # what the odometry source already computed).
        self.x = self.y = self.z = 0.0
        self.rgb_image = None  # latest HxWx3 uint8 array, RGB order
        self.target_detected = False
        self.bbox = (0.0, 0.0, 0.0, 0.0)  # (x, y, w, h) normalized, top-left origin

        # Multi-target state: 'targets' is every candidate object_tracker
        # currently sees (id -> normalized bbox), 'active_target_id' is
        # whichever one it's actually following (-1 = none right now),
        # and 'selected_target_id' is what THIS gui last asked it to
        # follow (-1 = auto). Selected and active can differ briefly (e.g.
        # right after clicking a button, before object_tracker's next
        # publish) or persistently (selected id chosen but not currently
        # visible).
        self.targets = []             # list of {'id': int, 'bbox': (x,y,w,h)}
        self.active_target_id = -1
        self.selected_target_id = -1

        # 3D map state, from semantic_map_node.py: map_points is the
        # accumulated lidar voxel cloud (Nx3 world-frame array, gray dots
        # in the MAP tab), map_targets_3d is {id: (x, y, z)} world-frame
        # position per currently-known target (yellow dots).
        self.map_points = np.empty((0, 3))
        self.map_targets_3d = {}

        # FIXED TARGETS: {gui_id: (x, y, z)} of every target confirmed,
        # from semantic_map_node.py's /map/targets_3d_fixed. Entries stay
        # on the map after the drone flies past and object_tracker loses
        # sight; they only go away if the node itself merges a duplicate
        # (see _map_targets_3d_fixed_cb). _fixed_entries is the bookkeeping
        # behind it: gui_id -> {'xyz': (x, y, z), 'nid': node fixed id or None}.
        self.map_targets_3d_fixed = {}
        self._fixed_entries = {}
        self.map_targets_3d_moving = {}
        self.last_map_targets_3d_moving_t = 0.0
        self._fixed_next_gui_id = 1

        # FLIGHT TRAIL: see TRAIL_SAMPLE_INTERVAL_SEC/TRAIL_MAX_POINTS
        # above for why this is sampled rather than appended on every
        # odometry message. A plain deque(maxlen=...) drops its oldest
        # point automatically once full, so the trail always shows the
        # most recent flying, never an unbounded flight history.
        self.trail = deque(maxlen=TRAIL_MAX_POINTS)
        self._last_trail_sample_t = 0.0

        # BATTERY: PX4's own battery estimate (px4_msgs/BatteryStatus on
        # /fmu/out/battery_status) -- battery_sim.py / flight_controller.py
        # no longer exist in this stack. None until a message arrives, so
        # the GUI shows "--" rather than a misleading "0%". `remaining` is
        # 0..1 and NaN when PX4 has no estimate yet.
        self.battery_voltage = None
        self.battery_current = None
        self.battery_percentage = None   # 0..1 fraction (BatteryStatus.remaining)
        self.battery_warning = 0
        self.battery_connected = True
        self.last_battery_t = 0.0

        # PX4 VEHICLE STATE (/fmu/out/vehicle_status_v4): arming, nav mode,
        # preflight checks. None until first message.
        self.armed = None
        self.nav_state = None
        self.preflight_ok = None
        self.last_status_t = 0.0

        # MOTOR OUTPUTS (/fmu/out/actuator_motors), normalized 0..1 per rotor.
        self.motors = [0.0] * NUM_MOTORS
        self.last_motors_t = 0.0

        # CONTROL MODE ('manual' | 'auto') published by keyboard_teleop on
        # /drone/control_mode.
        # Display-only: keyboard_teleop republishes control_mode at 20 Hz,
        # so publishing it from here too would fight it.
        self.control_mode = ''
        self.last_control_mode_t = 0.0

        # No mission manager in this stack anymore; waypoints stay empty so
        # the map's waypoint layer (MapView.render_map) simply draws nothing.
        self.mission_waypoints = np.empty((0, 3))
        self.last_mission_waypoints_t = 0.0

        self.last_odom_t = 0.0
        self.last_imu_t = 0.0
        self.last_cam_t = 0.0
        self.last_detect_t = 0.0
        self.last_bbox_t = 0.0
        self.last_targets_t = 0.0
        self.last_active_id_t = 0.0
        self.last_map_points_t = 0.0
        self.last_map_targets_3d_t = 0.0
        self.last_map_targets_3d_fixed_t = 0.0

        self.create_subscription(Odometry, '/model/drone/odometry', self._odom_cb, 10)
        self.create_subscription(Imu, '/model/drone/imu', self._imu_cb, 10)
        self.create_subscription(RosImage, '/model/drone/camera', self._camera_cb, 10)
        self.create_subscription(Bool, '/drone/target_detected', self._detect_cb, 10)
        self.create_subscription(Float32MultiArray, '/drone/target_bbox', self._bbox_cb, 10)
        self.create_subscription(Float32MultiArray, '/drone/targets', self._targets_cb, 10)
        self.create_subscription(Int32, '/drone/active_target_id', self._active_id_cb, 10)
        # UPDATE: qos_profile_sensor_data (best-effort) turned out to be
        # the wrong match here, not the fix. `ros2 topic info /map/points
        # --verbose` shows color_octomap_server_node's cloud publisher is
        # RELIABLE, and RViz's PointCloud2 display negotiates RELIABLE too
        # — that's why RViz kept growing while this GUI stalled on a small
        # early cloud. A best-effort *subscriber* doesn't ask for
        # retransmission, so as the map grew and the PointCloud2 message
        # got bigger (more UDP fragments per message), any single dropped
        # fragment silently killed that whole message here — no error,
        # just "keep showing the last one that arrived intact," which
        # looks exactly like the map being stuck at a small explored
        # radius. Subscribing RELIABLE (matching the publisher) fixes
        # that. semantic_map_node.py needs the same change.
        map_points_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(
            PointCloud2, '/map/points', self._map_points_cb, map_points_qos
        )
        self.create_subscription(Float32MultiArray, '/map/targets_3d', self._map_targets_3d_cb, 10)
        self.create_subscription(Float32MultiArray, '/map/targets_3d_moving', self._map_targets_3d_moving_cb, 10)
        self.create_subscription(
            Float32MultiArray, '/map/targets_3d_fixed', self._map_targets_3d_fixed_cb, 10
        )
        self.select_pub = self.create_publisher(Int32, '/drone/selected_target_id', 10)

        # PX4 topics. Names are parameters because px4_msgs versions the
        # topic suffix (e.g. vehicle_status_v4, battery_status_v1/v2): if
        # `ros2 topic list | grep battery` shows a suffix, override with
        #   --ros-args -p battery_topic:=/fmu/out/battery_status_vN
        self.declare_parameter('battery_topic', '/drone/battery_state')
        self.declare_parameter('vehicle_status_topic', '/fmu/out/vehicle_status_v4')
        self.declare_parameter('actuator_motors_topic', '/fmu/out/actuator_motors')
        self.create_subscription(
            BatteryState, self.get_parameter('battery_topic').value, self._battery_cb, 10)
        self.create_subscription(
            VehicleStatus, self.get_parameter('vehicle_status_topic').value, self._vehicle_status_cb, PX4_QOS)
        self.create_subscription(
            ActuatorMotors, self.get_parameter('actuator_motors_topic').value, self._motors_cb, PX4_QOS)
        self.create_subscription(String, '/drone/control_mode', self._control_mode_cb, 10)

    def _odom_cb(self, msg):
        with self.lock:
            self.vx = msg.twist.twist.linear.x
            self.vy = msg.twist.twist.linear.y
            self.vz = msg.twist.twist.linear.z
            q = msg.pose.pose.orientation
            self.yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
            p = msg.pose.pose.position
            self.x, self.y, self.z = p.x, p.y, p.z
            now = time.monotonic()
            self.last_odom_t = now
            if (self.control_mode in TRAIL_MODES and
                    (now - self._last_trail_sample_t) >= TRAIL_SAMPLE_INTERVAL_SEC):
                self.trail.append((p.x, p.y, p.z))
                self._last_trail_sample_t = now

    def _battery_cb(self, msg):
        with self.lock:
            self.battery_voltage = msg.voltage if math.isfinite(msg.voltage) else None
            # BatteryState: negative current = discharging; the GUI shows draw as positive.
            self.battery_current = -msg.current if math.isfinite(msg.current) else None
            self.battery_percentage = msg.percentage if math.isfinite(msg.percentage) else None
            self.battery_warning = battery_warning_level(self.battery_percentage)
            self.battery_connected = bool(msg.present)
            self.last_battery_t = time.monotonic()

    def _vehicle_status_cb(self, msg):
        with self.lock:
            self.armed = (msg.arming_state == ARMING_STATE_ARMED)
            self.nav_state = int(msg.nav_state)
            self.preflight_ok = bool(msg.pre_flight_checks_pass)
            self.last_status_t = time.monotonic()

    def _motors_cb(self, msg):
        # ActuatorMotors.control is a float32[12]; unused channels are NaN.
        vals = [float(c) if math.isfinite(c) else 0.0 for c in msg.control[:NUM_MOTORS]]
        with self.lock:
            self.motors = vals
            self.last_motors_t = time.monotonic()

    def _control_mode_cb(self, msg):
        with self.lock:
            entering_trail_mode = (msg.data in TRAIL_MODES and self.control_mode not in TRAIL_MODES)
            self.control_mode = msg.data
            self.last_control_mode_t = time.monotonic()
            if entering_trail_mode:
                # New auto session -> fresh path (no stale line, no jump
                # from wherever the drone was flown manually).
                self.trail.clear()
                self._last_trail_sample_t = time.monotonic()

    def _imu_cb(self, msg):
        with self.lock:
            self.wx = msg.angular_velocity.x
            self.wy = msg.angular_velocity.y
            self.wz = msg.angular_velocity.z
            self.last_imu_t = time.monotonic()

    def _camera_cb(self, msg):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            with self.lock:
                self.rgb_image = arr.copy()
                self.last_cam_t = time.monotonic()
        except Exception as e:
            self.get_logger().warn(f'camera frame decode failed: {e}')

    def _detect_cb(self, msg):
        with self.lock:
            self.target_detected = msg.data
            self.last_detect_t = time.monotonic()

    def _bbox_cb(self, msg):
        with self.lock:
            if len(msg.data) == 4:
                self.bbox = tuple(msg.data)
            self.last_bbox_t = time.monotonic()

    def _targets_cb(self, msg):
        data = msg.data
        n = len(data) // 5  # [id, x, y, w, h] repeated; drop any trailing partial record
        targets = []
        for i in range(n):
            base = i * 5
            targets.append({
                'id': int(round(data[base])),
                'bbox': tuple(data[base + 1:base + 5]),
            })
        with self.lock:
            self.targets = targets
            self.last_targets_t = time.monotonic()

    def _active_id_cb(self, msg):
        with self.lock:
            self.active_target_id = msg.data
            self.last_active_id_t = time.monotonic()

    def _map_points_cb(self, msg):
        pts = self._decode_cloud_xyz(msg)
        if pts is None:
            return
        with self.lock:
            self.map_points = pts
            self.last_map_points_t = time.monotonic()

    def _decode_cloud_xyz(self, msg):
        """Decode a PointCloud2's x/y/z fields into a plain (N, 3) float32
        array, without iterating the cloud one point at a time in Python.

        The previous implementation did:

            np.array(list(point_cloud2.read_points(
                msg, field_names=('x', 'y', 'z'), skip_nans=True)))

        `list()` forces a full Python-level pass building one record tuple
        per point, then `np.array()` walks that list again to rebuild an
        array. octomap_server republishes its ENTIRE occupied-voxel set on
        every integration (see octomap_params.yaml's comments), so this
        cost was paid, in full, on every single map update, on the same
        thread (via rclpy.spin in the background) that also has to service
        odometry/IMU/camera/target callbacks at higher rates. On a cloud
        with tens of thousands of points this could easily be the single
        most expensive thing happening in the process.

        read_points_numpy() reinterprets the message's raw byte buffer
        directly as a numpy array instead — no per-point loop. NaN
        filtering (replacing the old skip_nans=True) is one vectorized
        np.isfinite mask afterward.
        """
        try:
            if _HAVE_READ_POINTS_NUMPY:
                pts = read_points_numpy(msg, field_names=('x', 'y', 'z'))
            else:
                struct_pts = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'))
                pts = np.stack([struct_pts['x'], struct_pts['y'], struct_pts['z']], axis=-1)
            pts = pts.astype(np.float32, copy=False)
        except Exception as e:
            self.get_logger().warn(f'map points decode failed: {e}')
            return None

        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return pts[np.all(np.isfinite(pts), axis=1)]

    def _map_targets_3d_cb(self, msg):
        data = msg.data
        n = len(data) // 4  # [id, x, y, z] repeated
        targets_3d = {}
        for i in range(n):
            base = i * 4
            tid = int(round(data[base]))
            targets_3d[tid] = (data[base + 1], data[base + 2], data[base + 3])
        with self.lock:
            if targets_3d != self.map_targets_3d:
                self.map_targets_3d = targets_3d
                self.last_map_targets_3d_t = time.monotonic()

    def _map_targets_3d_moving_cb(self, msg):
        data = msg.data
        moving = {}
        for i in range(len(data) // 4):
            base = i * 4
            moving[int(round(data[base]))] = (data[base + 1], data[base + 2], data[base + 3])
        with self.lock:
            if moving != self.map_targets_3d_moving:
                self.map_targets_3d_moving = moving
                self.last_map_targets_3d_moving_t = time.monotonic()

    def _map_targets_3d_fixed_cb(self, msg):
        # NO REPEATED MARKERS. semantic_map_node creates each fixed marker
        # once (one per physical object, stable node id), so this GUI simply
        # mirrors it:
        #   pass 1  an incoming marker whose node id we already know updates
        #           that entry's position (refinement: a long-range camera-
        #           only marker snaps to the real lidar position).
        #   pass 2  anything left is matched BY POSITION to a not-yet-updated
        #           entry within FIXED_MERGE_RADIUS_M (same object, e.g. a
        #           marker from before a node restart) and adopts its node id;
        #           only if nothing is near does it become a new entry. A new
        #           marker near an old one therefore takes the old one's place
        #           instead of adding a second dot.
        #   cleanup entries the node tracked by id but no longer lists were
        #           duplicates the node merged away -> removed here too.
        # Kept on purpose: an EMPTY message never wipes the map, and markers
        # from before a node restart (ids reused at far-away positions) are
        # kept as permanent legacy entries.
        data = msg.data
        incoming = [
            (int(round(data[i * 4])), (data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]))
            for i in range(len(data) // 4)
        ]
        with self.lock:
            if not incoming:
                return
            entries = self._fixed_entries
            by_nid = {e['nid']: g for g, e in entries.items() if e['nid'] is not None}

            # Node restart detection: same node id, far away position.
            for nid, xyz in incoming:
                g = by_nid.get(nid)
                if g is not None:
                    k = entries[g]['xyz']
                    if math.hypot(xyz[0] - k[0], xyz[1] - k[1]) > FIXED_NID_MAX_SHIFT_M:
                        for e in entries.values():
                            e['nid'] = None
                        by_nid = {}
                        break

            touched, leftovers = set(), []
            for nid, xyz in incoming:                       # pass 1: by node id
                g = by_nid.get(nid)
                if g is not None and g not in touched:
                    entries[g]['xyz'] = xyz
                    touched.add(g)
                else:
                    leftovers.append((nid, xyz))

            for nid, xyz in leftovers:                      # pass 2: by position
                best, best_d2 = None, FIXED_MERGE_RADIUS_M ** 2
                for g, e in entries.items():
                    if g in touched:
                        continue
                    d2 = (xyz[0] - e['xyz'][0]) ** 2 + (xyz[1] - e['xyz'][1]) ** 2
                    if d2 <= best_d2:
                        best, best_d2 = g, d2
                if best is None:
                    best = self._fixed_next_gui_id
                    self._fixed_next_gui_id += 1
                    entries[best] = {'xyz': xyz, 'nid': nid}
                else:
                    entries[best]['xyz'] = xyz
                    entries[best]['nid'] = nid
                touched.add(best)

            # Node dropped an id it used to list -> merged duplicate.
            for g in [g for g, e in entries.items() if e['nid'] is not None and g not in touched]:
                del entries[g]

            new_fixed = {g: e['xyz'] for g, e in entries.items()}
            if new_fixed != self.map_targets_3d_fixed:
                self.map_targets_3d_fixed = new_fixed
                self.last_map_targets_3d_fixed_t = time.monotonic()

    def select_target(self, target_id):
        msg = Int32()
        msg.data = target_id
        self.select_pub.publish(msg)
        with self.lock:
            self.selected_target_id = target_id

    def get_snapshot(self):
        """Snapshot telemetry state once, under lock, for a consistent UI
        update without holding the lock across the whole redraw.

        Deliberately excludes map_points/map_targets_3d — those can be a
        many-thousand-point array (the entire octomap voxel set, see
        octomap_params.yaml's comments on why octomap_server republishes
        it in full every time), and copying that under lock on every call
        used to happen here even though this snapshot is pulled at ~33Hz
        for telemetry that never reads the map at all. get_map_snapshot()
        below is the only thing that needs that data, and it only runs at
        MAP_REDRAW_MS (~2Hz) and only while the Map tab is visible.
        """
        with self.lock:
            now = time.monotonic()
            return {
                'vx': self.vx, 'vy': self.vy, 'vz': self.vz,
                'wx': self.wx, 'wy': self.wy, 'wz': self.wz,
                'yaw_deg': math.degrees(self.yaw),
                'image': self.rgb_image,
                'targets': [t.copy() for t in self.targets],
                'active_target_id': self.active_target_id,
                'selected_target_id': self.selected_target_id,
                'target_detected': self.target_detected,
                'bbox': self.bbox,
                'battery_voltage': self.battery_voltage,
                'battery_percentage': self.battery_percentage,
                'battery_current': self.battery_current,
                'battery_warning': self.battery_warning,
                'battery_connected': self.battery_connected,
                'armed': self.armed,
                'nav_state': self.nav_state,
                'preflight_ok': self.preflight_ok,
                'motors': list(self.motors),
                'control_mode': self.control_mode,
                # Unlike map_points/map_targets_3d (excluded below — see
                # this method's own docstring), the fixed-targets dict is
                # small and bounded (one entry per target EVER confirmed,
                # not a many-thousand-point voxel cloud), so copying it
                # here every tick is cheap. Included so the FIXED TARGETS
                # side panel stays current regardless of which tab
                # (camera/map) is currently visible, rather than only
                # refreshing while the Map tab happens to be open (see
                # _update_map's tab-visibility gate).
                'map_targets_3d_fixed': self.map_targets_3d_fixed.copy(),
                'fresh': {
                    'odom': (now - self.last_odom_t) < STALE_AFTER_SEC,
                    'imu': (now - self.last_imu_t) < STALE_AFTER_SEC,
                    'cam': (now - self.last_cam_t) < STALE_AFTER_SEC,
                    'map': (now - self.last_map_points_t) < STALE_AFTER_SEC,
                    'target': (now - self.last_detect_t) < DETECT_STALE_AFTER_SEC,
                    'battery': (now - self.last_battery_t) < BATTERY_STALE_AFTER_SEC,
                    'px4': (now - self.last_status_t) < STALE_AFTER_SEC,
                    'motors': (now - self.last_motors_t) < STALE_AFTER_SEC,
                    'mode': (now - self.last_control_mode_t) < STALE_AFTER_SEC,
                },
                'target_status_known': (now - self.last_detect_t) < DETECT_STALE_AFTER_SEC,
            }

    def get_map_snapshot(self):
        """Snapshot map state only. Called from MapView.render_map's
        caller at MAP_REDRAW_MS (~2Hz), and only while the Map tab is
        visible. Includes active_target_id/selected_target_id alongside
        the map data so the caller has one lock-consistent view of
        everything the map render needs, and can detect "only the
        selection changed, not the map" as a reason to still redraw.
        """
        with self.lock:
            return {
                'map_points': self.map_points,
                'map_targets_3d': self.map_targets_3d.copy(),
                'map_targets_3d_fixed': self.map_targets_3d_fixed.copy(),
                'map_targets_3d_moving': self.map_targets_3d_moving.copy(),
                'control_mode': self.control_mode,
                'map_points_t': self.last_map_points_t,
                'map_targets_3d_t': self.last_map_targets_3d_t,
                'map_targets_3d_fixed_t': self.last_map_targets_3d_fixed_t,
                'map_targets_3d_moving_t': self.last_map_targets_3d_moving_t,
                'active_target_id': self.active_target_id,
                'selected_target_id': self.selected_target_id,
                'trail': np.array(self.trail, dtype=np.float32) if self.trail else np.empty((0, 3), dtype=np.float32),
                'trail_t': self._last_trail_sample_t,
                'mission_waypoints': self.mission_waypoints,
                'mission_waypoints_t': self.last_mission_waypoints_t,
            }


# ============================================================================
# Qt presentation layer.
# ============================================================================

def _mono_font(size, bold=False):
    f = QtGui.QFont('Consolas', size)
    f.setBold(bold)
    return f


def _ui_font(size, bold=False):
    f = QtGui.QFont('Segoe UI', size)
    f.setBold(bold)
    return f


class StatusDot(QtWidgets.QWidget):
    """Small filled circle, green when ok, red when not — same role as
    the old Tkinter status-dot canvases."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._ok = False

    def set_ok(self, ok):
        if ok != self._ok:
            self._ok = ok
            self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        color = ACCENT_QCOLOR if self._ok else NEGATIVE_QCOLOR
        p.setBrush(color)
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(1, 1, 8, 8)


class BarWidget(QtWidgets.QWidget):
    """Labeled, signed value bar: track + a fill growing from the center
    line toward the value's sign, plus a numeric readout. Replaces the
    old Tkinter bar canvases — those existed but were never actually
    updated with live values (see this file's module docstring); this
    version is wired up to repaint on every set_value() call.
    """

    def __init__(self, label, max_val, fmt='%+.2f', parent=None):
        super().__init__(parent)
        self.label = label
        self.max_val = max_val
        self.fmt = fmt
        self.value = 0.0
        self.setFixedHeight(34)
        self.setMinimumWidth(160)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setPen(QtGui.QColor(TEXT_DIM))
        p.setFont(_ui_font(9))
        p.drawText(QtCore.QRectF(0, 0, w * 0.6, 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.label)
        p.setPen(QtGui.QColor(ACCENT))
        p.setFont(_mono_font(10, bold=True))
        p.drawText(QtCore.QRectF(0, 0, w, 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, self.fmt % self.value)

        bar_y, bar_h = 20, 10
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(TRACK))
        p.drawRoundedRect(0, bar_y, w, bar_h, 3, 3)

        frac = max(-1.0, min(1.0, self.value / self.max_val)) if self.max_val else 0.0
        cx = w / 2.0
        fill_color = QtGui.QColor(ACCENT) if frac >= 0 else QtGui.QColor(NEGATIVE)
        p.setBrush(fill_color)
        fill_w = abs(frac) * cx
        if frac >= 0:
            p.drawRoundedRect(QtCore.QRectF(cx, bar_y, fill_w, bar_h), 3, 3)
        else:
            p.drawRoundedRect(QtCore.QRectF(cx - fill_w, bar_y, fill_w, bar_h), 3, 3)
        p.setPen(QtGui.QColor(TEXT_FAINT))
        p.drawLine(QtCore.QPointF(cx, bar_y - 1), QtCore.QPointF(cx, bar_y + bar_h + 1))


class CompassWidget(QtWidgets.QWidget):
    """RViz-style heading compass: N/E/S/W ring + needle."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(100, 100)
        self.heading_deg = 0.0

    def set_heading(self, heading_deg):
        self.heading_deg = heading_deg
        self.update()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.fillRect(self.rect(), QtGui.QColor(TRACK))
        cx, cy, r = 50, 50, 41

        p.setPen(QtGui.QPen(QtGui.QColor(PANEL_BORDER), 2))
        p.setBrush(QtCore.Qt.NoBrush)
        p.drawEllipse(QtCore.QPointF(cx, cy), r, r)

        p.setPen(QtGui.QColor(TEXT_FAINT))
        p.setFont(_ui_font(7))
        for label, ang in (('N', -90), ('E', 0), ('S', 90), ('W', 180)):
            rad = math.radians(ang)
            tx = cx + (r - 14) * math.cos(rad)
            ty = cy + (r - 14) * math.sin(rad)
            p.drawText(QtCore.QRectF(tx - 8, ty - 8, 16, 16), QtCore.Qt.AlignCenter, label)

        needle_ang = math.radians(-90 - self.heading_deg)
        tip = QtCore.QPointF(cx + (r - 10) * math.cos(needle_ang), cy + (r - 10) * math.sin(needle_ang))
        tail = QtCore.QPointF(cx - (r - 40) * math.cos(needle_ang), cy - (r - 40) * math.sin(needle_ang))
        p.setPen(QtGui.QPen(QtGui.QColor(ACCENT), 3))
        p.drawLine(tail, tip)
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(TEXT_DIM))
        p.drawEllipse(QtCore.QPointF(cx, cy), 3, 3)


class MapView(gl.GLViewWidget):
    """RViz-like 3D map: GPU point cloud + target markers + reference
    grid, all rendered through real OpenGL via pyqtgraph.opengl instead
    of matplotlib's CPU-only Axes3D.

    The scene's items (scatter plots, grids) are created ONCE in
    __init__ and updated in place on every redraw via .setData() /
    .setSize() — never destroyed and rebuilt. That's the actual
    structural difference from the old _draw_map: this pushes new vertex
    data to a GPU buffer that already exists, instead of tearing down and
    reconstructing the whole scene graph every time.

    CLICK-TO-SELECT: GLViewWidget has no built-in picking, so this
    manually projects each known target's world position through the
    current view/projection matrices to get its on-screen position, then
    picks whichever projected target is closest to a mouse-release point
    — but only when that release wasn't part of an orbit/pan drag (i.e.
    press and release landed within CLICK_MOVE_THRESHOLD_PX of each
    other). GLViewWidget's own mousePressEvent/mouseMoveEvent still run
    first via super(), so orbit/pan/zoom behavior is untouched; this only
    adds a check on release.
    """

    MIN_EXTENT_XY = 5.0
    MIN_EXTENT_Z = 3.0
    MARGIN = 1.15

    # A mouse-up counts as a "click" (attempt a pick) rather than the end
    # of a camera drag if it moved less than this many pixels from the
    # mouse-down. GLViewWidget itself has no click/drag distinction, so
    # this recreates one purely for picking purposes.
    CLICK_MOVE_THRESHOLD_PX = 4
    # How close (in screen pixels) a click needs to land to a target's
    # projected position to select it. Generous relative to the dot's
    # rendered size since GLScatterPlotItem points don't hit-test
    # themselves — this is an approximation, not exact GPU picking.
    PICK_RADIUS_PX = 24

    targetClicked = QtCore.pyqtSignal(int)

    def __init__(self, on_voxel_count_changed=None, parent=None):
        super().__init__(parent)
        self.setBackgroundColor(BG)
        self.setCameraPosition(distance=60, elevation=25, azimuth=-60)
        self._on_voxel_count_changed = on_voxel_count_changed
        self._current_extent_xy = self.MIN_EXTENT_XY
        self._current_extent_z = self.MIN_EXTENT_Z
        self._press_pos = None
        # [(id, (x, y, z)), ...] cache of the targets drawn in the most
        # recent render_map() call, so picking doesn't need its own
        # separate data path into the node.
        self._targets_screen_cache = []

        try:
            self._cmap = mpl_colors.LinearSegmentedColormap.from_list(
                'calm_terrain', MAP_TERRAIN_COLORS, N=256)
        except Exception:
            self._cmap = mpl_cm.get_cmap('viridis')

        # Floor grid + two back walls, same reasoning as the old
        # _draw_grid: a floor-only grid reads as a flat 2D plane with
        # points floating above it; mirroring it onto two walls gives the
        # box actual 3D depth cues. Created once, resized (cheap: a grid
        # is a few hundred line vertices at most) whenever the auto-fit
        # extent changes.
        self.grid_floor = gl.GLGridItem()
        self.grid_wall_x = gl.GLGridItem()
        self.grid_wall_y = gl.GLGridItem()
        for g in (self.grid_floor, self.grid_wall_x, self.grid_wall_y):
            g.setColor(GRID_COLOR)
            self.addItem(g)

        # Voxel cloud. pxMode=False -> size is in world units (meters),
        # so points keep a consistent physical size regardless of zoom,
        # matching RViz's Size(m) semantics rather than a fixed pixel size.
        self.scatter_points = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=MAP_POINT_SIZE_M, pxMode=False
        )
        self.addItem(self.scatter_points)

        # Target markers, drawn as a second, separately-updated item so
        # its (much smaller, more frequently color-changed) data never
        # forces a re-upload of the whole voxel cloud.
        self.scatter_targets = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=TARGET_POINT_SIZE_M, pxMode=False
        )
        self.addItem(self.scatter_targets)

        # FIXED (permanent) target markers -- see FIXED_TARGET_COLOR above.
        # Separate item from scatter_targets: a fixed marker's position
        # only ever gets appended to, never removed or recolored based on
        # selection state, so there's no reason to share a buffer with the
        # live markers' more dynamic data.
        self.scatter_fixed_targets = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=FIXED_TARGET_POINT_SIZE_M,
            pxMode=False, color=FIXED_TARGET_COLOR,
        )
        self.addItem(self.scatter_fixed_targets)

        # FLIGHT TRAIL: where the drone has actually flown, as a thin
        # connected line rather than a scatter cloud — a trail reads as a
        # PATH at a glance the way a dot cloud with the same points
        # wouldn't. Same "create once, .setData() to update" pattern as
        # the scatters above.
        self.trail_line = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32), color=TRAIL_COLOR, width=2.0, antialias=True
        )
        self.addItem(self.trail_line)

        # MISSION WAYPOINTS: the active mission's target list (see
        # mission_manager.py) — or, in explore mode, its current single
        # frontier target — drawn as diamonds distinct in color from both
        # the voxel cloud and the tracked-object markers, so a mission and
        # an object-tracking session are never visually confused on the
        # same map.
        self.scatter_waypoints = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32), size=WAYPOINT_POINT_SIZE_M,
            pxMode=False, color=WAYPOINT_COLOR,
        )
        self.addItem(self.scatter_waypoints)

        # BLEND MODE: GLScatterPlotItem defaults to 'additive', which SUMS
        # colours where points overlap -> dense terrain washes out to
        # bright/white and a marker under other points turns pale. Use
        # normal alpha blending instead, and give markers a higher depth
        # value so they are always drawn AFTER (on top of) the terrain.
        # Stems for the pin markers (line segments, drawn in pairs).
        self.stems_fixed = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32), color=FIXED_TARGET_COLOR,
            width=PIN_STEM_WIDTH, antialias=True, mode='lines')
        self.stems_live = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32), color=QtGui.QColor(ALARM_BRIGHT).getRgbF(),
            width=PIN_STEM_WIDTH, antialias=True, mode='lines')
        for _it, _d in ((self.stems_fixed, 8), (self.stems_live, 9)):
            self.addItem(_it)
            _it.setGLOptions('translucent')
            _it.setDepthValue(_d)
        self._tracked_trail_pts = []
        self._tracked_trail_id = None
        self.tracked_trail_line = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32), color=TRACKED_TRAIL_COLOR, width=2.5, antialias=True)
        self.addItem(self.tracked_trail_line)
        self.tracked_trail_line.setGLOptions('translucent')
        self.tracked_trail_line.setDepthValue(9)
        self.scatter_points.setGLOptions('translucent')
        self.scatter_points.setDepthValue(0)
        for _item, _d in ((self.scatter_waypoints, 5), (self.scatter_targets, 10),
                          (self.scatter_fixed_targets, 11)):
            _item.setGLOptions('translucent')
            _item.setDepthValue(_d)

    def _resize_grids(self, extent_xy, extent_z):
        step = self._nice_grid_step(extent_xy)
        size_xy = extent_xy * 2
        self.grid_floor.setSize(x=size_xy, y=size_xy)
        self.grid_floor.setSpacing(x=step, y=step)
        self.grid_floor.resetTransform()

        wall_size = max(size_xy, extent_z * 2)
        self.grid_wall_x.setSize(x=wall_size, y=wall_size)
        self.grid_wall_x.setSpacing(x=step, y=step)
        self.grid_wall_x.resetTransform()
        self.grid_wall_x.rotate(90, 0, 1, 0)
        self.grid_wall_x.translate(-extent_xy, 0, extent_z / 2.0)

        self.grid_wall_y.setSize(x=wall_size, y=wall_size)
        self.grid_wall_y.setSpacing(x=step, y=step)
        self.grid_wall_y.resetTransform()
        self.grid_wall_y.rotate(90, 1, 0, 0)
        self.grid_wall_y.translate(0, -extent_xy, extent_z / 2.0)

    @staticmethod
    def _nice_grid_step(extent):
        """Human-friendly grid spacing keeping ~20 lines per axis
        regardless of how far the auto-fit view extends, instead of a
        fixed step that would turn into hundreds of grid lines once the
        view covers a large explored area."""
        target_lines = 20
        raw_step = (2 * extent) / target_lines
        if raw_step <= 0:
            return 1.0
        magnitude = 10 ** math.floor(math.log10(raw_step))
        for m in (1, 2, 5, 10):
            step = m * magnitude
            if step >= raw_step:
                return step
        return 10 * magnitude

    def _compute_extent(self, points, targets_3d, trail=None, waypoints=None, fixed_targets_3d=None):
        xy_max = self.MIN_EXTENT_XY
        z_max = self.MIN_EXTENT_Z
        if points.shape[0]:
            xy_max = max(xy_max, float(np.abs(points[:, :2]).max()))
            z_max = max(z_max, float(points[:, 2].max()))
        if targets_3d:
            txyz = np.array(list(targets_3d.values()), dtype=float)
            xy_max = max(xy_max, float(np.abs(txyz[:, :2]).max()))
            z_max = max(z_max, float(txyz[:, 2].max()))
        if fixed_targets_3d:
            fxyz = np.array(list(fixed_targets_3d.values()), dtype=float)
            xy_max = max(xy_max, float(np.abs(fxyz[:, :2]).max()))
            z_max = max(z_max, float(fxyz[:, 2].max()))
        if trail is not None and trail.shape[0]:
            xy_max = max(xy_max, float(np.abs(trail[:, :2]).max()))
            z_max = max(z_max, float(trail[:, 2].max()))
        if waypoints is not None and waypoints.shape[0]:
            xy_max = max(xy_max, float(np.abs(waypoints[:, :2]).max()))
            z_max = max(z_max, float(waypoints[:, 2].max()))
        return xy_max * self.MARGIN, z_max * self.MARGIN

    def _pin(self, points, xyz):
        """(foot, head) for a marker at xyz. The foot is snapped to the top
        of the map points under the target so it sits ON the cloud; the head
        floats PIN_LIFT_M above it so it never covers the points."""
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        # Cache foot heights per 0.25 m cell until a new cloud arrives.
        if points is not getattr(self, '_pin_cache_pts', None):
            self._pin_cache_pts = points
            self._pin_cache = {}
        key = (round(x * 4), round(y * 4))
        foot_z = self._pin_cache.get(key)
        if foot_z is None:
            foot_z = z
            if points is not None and points.shape[0]:
                d2 = (points[:, 0] - x) ** 2 + (points[:, 1] - y) ** 2
                near = points[d2 <= PIN_SNAP_RADIUS_M ** 2]
                if near.shape[0]:
                    foot_z = float(np.percentile(near[:, 2], 98))
                    self._pin_cache[key] = foot_z
        foot = np.array([x, y, foot_z], dtype=np.float32)
        head = np.array([x, y, foot_z + PIN_LIFT_M], dtype=np.float32)
        return foot, head

    def render_map(self, points, targets_3d, active_id, selected_id, trail=None, waypoints=None,
                    fixed_targets_3d=None, moving_targets_3d=None, control_mode=''):
        if trail is None:
            trail = np.empty((0, 3), dtype=np.float32)
        if waypoints is None:
            waypoints = np.empty((0, 3), dtype=np.float32)
        if fixed_targets_3d is None:
            fixed_targets_3d = {}
        if moving_targets_3d is None:
            moving_targets_3d = {}
        extent_xy, extent_z = self._compute_extent(points, targets_3d, trail, waypoints, fixed_targets_3d)
        # Regenerating the grid geometry is cheap, but there's still no
        # reason to do it every single redraw if the view hasn't actually
        # grown — only touch it when the extent moved meaningfully.
        if (abs(extent_xy - self._current_extent_xy) > 1e-6 or
                abs(extent_z - self._current_extent_z) > 1e-6):
            self._resize_grids(extent_xy, extent_z)
            self._current_extent_xy = extent_xy
            self._current_extent_z = extent_z

        # SPEED PATCH: only recolour + re-upload the cloud when a NEW cloud
        # has arrived (the array object is replaced in _map_points_cb).
        # semantic_map_node republishes the target topics every tick, so
        # render_map runs ~4 Hz even when the map is unchanged; redoing the
        # percentile + colormap + GPU upload of ~90k points each time made
        # the GUI thread sluggish and delayed marker updates.
        if points is not getattr(self, '_last_points_obj', None):
            self._last_points_obj = points
            if points.shape[0]:
                z_vals = points[:, 2]
                lo, hi = np.percentile(z_vals, MAP_TERRAIN_PCT)
                if hi - lo < 1e-3:
                    hi = lo + 1e-3
                norm = mpl_colors.Normalize(vmin=float(lo), vmax=float(hi), clip=True)
                colors = self._cmap(norm(z_vals)).astype(np.float32)  # (N,4) RGBA in 0..1
                colors[:, 3] = MAP_TERRAIN_ALPHA
                self.scatter_points.setData(pos=points, color=colors, size=MAP_POINT_SIZE_M)
            else:
                self.scatter_points.setData(pos=np.empty((0, 3), dtype=np.float32))

        tracked_id = selected_id if selected_id != -1 else active_id
        empty3 = np.empty((0, 3), dtype=np.float32)

        # LIVE MARKER: exactly ONE -- the tracked object -- and only while
        # the drone is in auto mode AND semantic_map_node classifies that
        # object as moving. Nothing else gets a live marker. Fixed boxes
        # are drawn separately below and never get a trail.
        in_auto = LIVE_MARKER_MODES is None or control_mode in LIVE_MARKER_MODES
        live = {}
        if (in_auto and tracked_id != -1 and
                tracked_id in moving_targets_3d and tracked_id in targets_3d):
            live = {tracked_id: targets_3d[tracked_id]}
        trail_id = tracked_id if live else None

        # PATH of the tracked mover: recorded only in auto mode, restarts
        # when the tracked object changes or auto mode ends, and survives
        # short gaps where the mover briefly drops out of the moving list.
        trail_on = control_mode in TRAIL_MODES
        if not trail_on or tracked_id == -1:
            self._tracked_trail_pts = []
            self._tracked_trail_id = None
        elif trail_id is not None and trail_id != self._tracked_trail_id:
            self._tracked_trail_pts = []
            self._tracked_trail_id = trail_id
        if trail_on and trail_id is not None:
            cur = np.array(live[trail_id], dtype=np.float32)
            if (not self._tracked_trail_pts or
                    float(np.linalg.norm(cur - self._tracked_trail_pts[-1])) >= TRACKED_TRAIL_MIN_STEP_M):
                self._tracked_trail_pts.append(cur)
                del self._tracked_trail_pts[:-TRACKED_TRAIL_MAX_POINTS]
        if trail_on and len(self._tracked_trail_pts) >= 2:
            self.tracked_trail_line.setData(
                pos=np.array(self._tracked_trail_pts, dtype=np.float32),
                color=TRACKED_TRAIL_COLOR, width=2.5)
        else:
            self.tracked_trail_line.setData(pos=empty3)

        if live:
            pos, colors, sizes, stems, cache = [], [], [], [], []
            for tid, xyz in live.items():
                foot, head = self._pin(points, xyz)
                is_sel = (tid == trail_id)
                col = QtGui.QColor(ALARM_BRIGHT if is_sel else ACCENT).getRgbF()
                pos += [foot, head]
                colors += [col, col]
                sizes += [PIN_FOOT_SIZE_M, PIN_HEAD_SIZE_SELECTED_M if is_sel else PIN_HEAD_SIZE_M]
                stems += [foot, head]
                cache.append((tid, tuple(head)))
            self.scatter_targets.setData(
                pos=np.array(pos, dtype=np.float32), color=np.array(colors, dtype=np.float32),
                size=np.array(sizes, dtype=np.float32))
            self.stems_live.setData(pos=np.array(stems, dtype=np.float32))
            self._targets_screen_cache = cache
        else:
            self.scatter_targets.setData(pos=empty3)
            self.stems_live.setData(pos=empty3)
            self._targets_screen_cache = []

        # FIXED markers: always drawn, whatever the mode / tracked target.
        if fixed_targets_3d:
            pos, colors, sizes, stems = [], [], [], []
            for xyz in fixed_targets_3d.values():
                foot, head = self._pin(points, xyz)
                pos += [foot, head]
                colors += [FIXED_TARGET_COLOR, FIXED_TARGET_COLOR]
                sizes += [PIN_FOOT_SIZE_M, PIN_HEAD_SIZE_M * 1.3]
                stems += [foot, head]
            self.scatter_fixed_targets.setData(
                pos=np.array(pos, dtype=np.float32), color=np.array(colors, dtype=np.float32),
                size=np.array(sizes, dtype=np.float32))
            self.stems_fixed.setData(pos=np.array(stems, dtype=np.float32))
        else:
            self.scatter_fixed_targets.setData(pos=empty3)
            self.stems_fixed.setData(pos=empty3)

        if self._on_voxel_count_changed:
            self._on_voxel_count_changed(points.shape[0], len(targets_3d))

        if trail_on and trail.shape[0] >= 2:
            self.trail_line.setData(pos=trail, color=TRAIL_COLOR, width=2.0)
        else:
            self.trail_line.setData(pos=np.empty((0, 3), dtype=np.float32))

        if waypoints.shape[0]:
            self.scatter_waypoints.setData(pos=waypoints, color=WAYPOINT_COLOR, size=WAYPOINT_POINT_SIZE_M)
        else:
            self.scatter_waypoints.setData(pos=np.empty((0, 3), dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Click-to-select
    # ------------------------------------------------------------------ #
    def _project_to_screen(self, world_xyz):
        """World-space (x, y, z) -> (screen_x, screen_y) in this widget's
        own pixel coordinates, using the same view/projection matrices
        GLViewWidget itself uses to render. Returns None if the point is
        behind the camera (w <= 0)."""
        mvp = self.projectionMatrix() * self.viewMatrix()
        clip = mvp.map(QtGui.QVector4D(float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]), 1.0))
        if clip.w() <= 0:
            return None
        ndc_x = clip.x() / clip.w()
        ndc_y = clip.y() / clip.w()
        w, h = self.width(), self.height()
        screen_x = (ndc_x * 0.5 + 0.5) * w
        screen_y = (1.0 - (ndc_y * 0.5 + 0.5)) * h
        return screen_x, screen_y

    def _pick_target_at(self, pos):
        """Return the id of whichever cached target projects closest to
        screen-space `pos` (a QPoint), within PICK_RADIUS_PX — or None."""
        best_id, best_dist = None, self.PICK_RADIUS_PX
        for tid, xyz in self._targets_screen_cache:
            screen = self._project_to_screen(xyz)
            if screen is None:
                continue
            dist = math.hypot(screen[0] - pos.x(), screen[1] - pos.y())
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        return best_id

    def mousePressEvent(self, ev):
        self._press_pos = ev.pos()
        super().mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
        if self._press_pos is not None and ev.button() == QtCore.Qt.LeftButton:
            moved = (ev.pos() - self._press_pos).manhattanLength()
            if moved <= self.CLICK_MOVE_THRESHOLD_PX:
                picked = self._pick_target_at(ev.pos())
                if picked is not None:
                    self.targetClicked.emit(picked)
        self._press_pos = None


class DroneGuiApp(QtWidgets.QMainWindow):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle('DRONE TELEMETRY')
        self.resize(1600, 900)
        self.setStyleSheet(f'background-color: {BG};')

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_telemetry_panel(), stretch=0)
        layout.addWidget(self._build_view_panel(), stretch=1)

        # Sentinels distinct from any real value (timestamps are floats
        # starting at 0.0, ids are ints or None) so the first map redraw
        # always happens once, then subsequent ticks only re-render when
        # something in the snapshot actually differs.
        self._last_rendered_map_points_t = None
        self._last_rendered_map_targets_t = None
        self._last_rendered_map_targets_fixed_t = None
        self._last_rendered_map_targets_moving_t = object()
        self._last_rendered_control_mode = object()
        self._last_rendered_active_id = object()
        self._last_rendered_selected_id = object()
        self._last_rendered_trail_t = None
        self._last_rendered_mission_waypoints_t = None

        self.telemetry_timer = QtCore.QTimer(self)
        self.telemetry_timer.timeout.connect(self._update_telemetry)
        self.telemetry_timer.start(30)  # ~33 Hz, matches the old Tkinter loop

        self.map_timer = QtCore.QTimer(self)
        self.map_timer.timeout.connect(self._update_map)
        self.map_timer.start(MAP_REDRAW_MS)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def _panel_frame(self):
        f = QtWidgets.QFrame()
        f.setStyleSheet(f'background-color: {PANEL}; border-radius: 6px;')
        return f

    def _section_label(self, text):
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet(f'color: {TEXT}; background: transparent;')
        lbl.setFont(_ui_font(9, bold=True))
        return lbl

    def _build_telemetry_panel(self):
        frame = self._panel_frame()
        frame.setFixedWidth(480)
        outer = QtWidgets.QVBoxLayout(frame)
        outer.setContentsMargins(15, 15, 15, 15)

        title = QtWidgets.QLabel('DRONE TELEMETRY')
        title.setStyleSheet(f'color: {TEXT}; background: transparent;')
        title.setFont(_ui_font(15, bold=True))
        outer.addWidget(title)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet('background: transparent; border: none;')
        outer.addWidget(scroll, stretch=1)

        inner = QtWidgets.QWidget()
        inner.setStyleSheet('background: transparent;')
        scroll.setWidget(inner)
        v = QtWidgets.QVBoxLayout(inner)
        v.setSpacing(14)

        self.bars = {}
        v.addWidget(self._section_label('VELOCITY (m/s)'))
        for key, label in (('vx', 'Linear X'), ('vy', 'Linear Y'), ('vz', 'Linear Z')):
            bar = BarWidget(label, max_val=VELOCITY_BAR_MAX_MPS)
            self.bars[key] = bar
            v.addWidget(bar)

        v.addWidget(self._section_label('ANGULAR RATE (rad/s)'))
        for key, label in (('wx', 'Roll rate'), ('wy', 'Pitch rate'), ('wz', 'Yaw rate')):
            bar = BarWidget(label, max_val=1.0)
            self.bars[key] = bar
            v.addWidget(bar)

        v.addWidget(self._section_label('HEADING (yaw)'))
        heading_row = QtWidgets.QHBoxLayout()
        self.compass = CompassWidget()
        heading_row.addWidget(self.compass)
        heading_text_col = QtWidgets.QVBoxLayout()
        self.heading_label = QtWidgets.QLabel('+0°')
        self.heading_label.setStyleSheet(f'color: {ACCENT}; background: transparent;')
        self.heading_label.setFont(_mono_font(13, bold=True))
        heading_sub = QtWidgets.QLabel('relative to initial heading')
        heading_sub.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
        heading_sub.setFont(_ui_font(7))
        heading_text_col.addWidget(self.heading_label)
        heading_text_col.addWidget(heading_sub)
        heading_text_col.addStretch(1)
        heading_row.addLayout(heading_text_col)
        heading_row.addStretch(1)
        v.addLayout(heading_row)

        v.addWidget(self._section_label('FLIGHT STATUS (PX4)'))
        self.px4_state_label = QtWidgets.QLabel('no PX4 link')
        self.px4_state_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
        self.px4_state_label.setFont(_mono_font(10, bold=True))
        v.addWidget(self.px4_state_label)
        self.px4_preflight_label = QtWidgets.QLabel('')
        self.px4_preflight_label.setStyleSheet(f'color: {TEXT_DIM}; background: transparent;')
        self.px4_preflight_label.setFont(_ui_font(8))
        v.addWidget(self.px4_preflight_label)

        v.addWidget(self._section_label('BATTERY'))
        battery_row = QtWidgets.QHBoxLayout()
        self.battery_bar = QtWidgets.QProgressBar()
        self.battery_bar.setRange(0, 100)
        self.battery_bar.setValue(0)
        self.battery_bar.setTextVisible(False)
        self.battery_bar.setFixedHeight(14)
        self.battery_bar.setStyleSheet(f'''
            QProgressBar {{ background: {TRACK}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
        ''')
        battery_row.addWidget(self.battery_bar, stretch=1)
        self.battery_label = QtWidgets.QLabel('-- %  /  -- V')
        self.battery_label.setStyleSheet(f'color: {TEXT_DIM}; background: transparent;')
        self.battery_label.setFont(_mono_font(9, bold=True))
        battery_row.addWidget(self.battery_label)
        v.addLayout(battery_row)
        self.battery_detail_label = QtWidgets.QLabel('')
        self.battery_detail_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
        self.battery_detail_label.setFont(_ui_font(8))
        v.addWidget(self.battery_detail_label)

        v.addWidget(self._section_label('MOTOR OUTPUTS (PX4)'))
        self.motor_bars = []
        for i in range(NUM_MOTORS):
            row = QtWidgets.QHBoxLayout()
            name = QtWidgets.QLabel(f'M{i + 1}')
            name.setStyleSheet(f'color: {TEXT_DIM}; background: transparent;')
            name.setFont(_ui_font(9))
            name.setFixedWidth(28)
            row.addWidget(name)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(10)
            bar.setStyleSheet(f'''
                QProgressBar {{ background: {TRACK}; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background: {ACCENT}; border-radius: 3px; }}
            ''')
            row.addWidget(bar, stretch=1)
            self.motor_bars.append(bar)
            v.addLayout(row)

        v.addWidget(self._section_label('CONTROL MODE'))
        self.control_mode_label = QtWidgets.QLabel('mode: --')
        self.control_mode_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
        self.control_mode_label.setFont(_mono_font(9, bold=True))
        v.addWidget(self.control_mode_label)

        v.addWidget(self._section_label('LINK STATUS'))
        self.status_dots = {}
        self.status_labels = {}
        for key, label in (('odom', 'Odometry'), ('imu', 'IMU'), ('cam', 'Camera'),
                            ('map', 'Map Feed'), ('target', 'Target Locked'),
                            ('battery', 'Battery (PX4)'), ('px4', 'PX4 Link'),
                            ('motors', 'Motor Outputs')):
            row = QtWidgets.QHBoxLayout()
            dot = StatusDot()
            self.status_dots[key] = dot
            row.addWidget(dot)
            name_lbl = QtWidgets.QLabel(label)
            name_lbl.setStyleSheet(f'color: {TEXT_DIM}; background: transparent;')
            name_lbl.setFont(_ui_font(9))
            row.addWidget(name_lbl)
            row.addStretch(1)
            status_lbl = QtWidgets.QLabel('')
            status_lbl.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
            status_lbl.setFont(_ui_font(7))
            self.status_labels[key] = status_lbl
            row.addWidget(status_lbl)
            v.addLayout(row)

        v.addWidget(self._section_label('TARGETS (click to follow)'))
        self.targets_list_widget = QtWidgets.QWidget()
        self.targets_list_layout = QtWidgets.QVBoxLayout(self.targets_list_widget)
        self.targets_list_layout.setContentsMargins(0, 0, 0, 0)
        self.targets_list_layout.setSpacing(4)
        v.addWidget(self.targets_list_widget)

        # FIXED TARGETS: read-only list, not click-to-follow like TARGETS
        # above — these are historical/permanent confirmations (see
        # semantic_map_node.py's /map/targets_3d_fixed and
        # FIXED_TARGET_COLOR above), each shown with its exact world-frame
        # position so a static target's location is readable as text, not
        # just as a red dot on the MAP tab.
        v.addWidget(self._section_label('FIXED TARGETS (confirmed positions)'))
        self.fixed_targets_list_widget = QtWidgets.QWidget()
        self.fixed_targets_list_layout = QtWidgets.QVBoxLayout(self.fixed_targets_list_widget)
        self.fixed_targets_list_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_targets_list_layout.setSpacing(2)
        v.addWidget(self.fixed_targets_list_widget)

        v.addStretch(1)
        return frame

    def _build_view_panel(self):
        frame = self._panel_frame()
        v = QtWidgets.QVBoxLayout(frame)
        v.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setStyleSheet(f'''
            QTabWidget::pane {{ border: none; background: {PANEL}; }}
            QTabBar::tab {{ background: {PANEL}; color: {TEXT}; padding: 8px 14px; }}
            QTabBar::tab:selected {{ background: {TRACK}; color: {ACCENT}; }}
        ''')
        v.addWidget(self.tabs)

        camera_tab = QtWidgets.QWidget()
        camera_tab.setStyleSheet(f'background: {PANEL};')
        cv = QtWidgets.QVBoxLayout(camera_tab)

        self.alert_label = QtWidgets.QLabel('• TARGET LOCKED •')
        self.alert_label.setAlignment(QtCore.Qt.AlignCenter)
        self.alert_label.setFont(_ui_font(11, bold=True))
        self.alert_label.setStyleSheet(f'background: {ALARM_DIM}; color: white; padding: 4px;')
        self.alert_label.hide()
        cv.addWidget(self.alert_label)

        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setStyleSheet(f'background: {PANEL};')
        self.camera_label.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_label.setMinimumSize(320, 240)
        cv.addWidget(self.camera_label, stretch=1)

        cam_status_row = QtWidgets.QHBoxLayout()
        self.cam_status_dot = StatusDot()
        cam_status_row.addWidget(self.cam_status_dot)
        self.cam_status_label = QtWidgets.QLabel('LIVE')
        self.cam_status_label.setStyleSheet(f'color: {ACCENT}; background: transparent;')
        self.cam_status_label.setFont(_ui_font(9, bold=True))
        cam_status_row.addWidget(self.cam_status_label)
        cam_status_row.addStretch(1)
        cv.addLayout(cam_status_row)

        self.tabs.addTab(camera_tab, 'CAMERA')

        map_tab = QtWidgets.QWidget()
        map_tab.setStyleSheet(f'background: {PANEL};')
        mv = QtWidgets.QVBoxLayout(map_tab)
        mv.setContentsMargins(0, 0, 0, 0)

        self.map_view = MapView(on_voxel_count_changed=self._on_voxel_count_changed)
        self.map_view.targetClicked.connect(self.node.select_target)
        mv.addWidget(self.map_view, stretch=1)

        self.map_info_label = QtWidgets.QLabel('0 voxels · 0 targets')
        self.map_info_label.setStyleSheet(f'color: {TEXT_DIM}; background: transparent;')
        self.map_info_label.setFont(_ui_font(7))
        mv.addWidget(self.map_info_label)

        self.tabs.addTab(map_tab, 'MAP')
        return frame

    def _on_voxel_count_changed(self, n_voxels, n_targets):
        self.map_info_label.setText(f'{n_voxels} voxels · {n_targets} targets')

    # ------------------------------------------------------------------ #
    # Targets list
    # ------------------------------------------------------------------ #
    def _refresh_targets_panel(self, targets, active_id, selected_id):
        # Small, bounded list (unlike the map data), so a full rebuild on
        # every telemetry tick is genuinely cheap — left as-is rather than
        # adding diffing complexity that wouldn't measurably help.
        layout = self.targets_list_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not targets:
            lbl = QtWidgets.QLabel('No targets detected')
            lbl.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
            lbl.setFont(_ui_font(7))
            layout.addWidget(lbl)
            return

        for t in targets:
            tid = t['id']
            is_selected = (tid == selected_id) or (selected_id == -1 and tid == active_id)
            bg = ALARM_DIM if is_selected else TRACK
            fg = TEXT if is_selected else TEXT_DIM
            btn = QtWidgets.QPushButton(f'Target #{tid}')
            btn.setStyleSheet(f'''
                QPushButton {{ background-color: {bg}; color: {fg}; border: 1px solid {PANEL_BORDER};
                               padding: 6px; text-align: left; }}
                QPushButton:hover {{ background-color: {PANEL}; }}
            ''')
            btn.clicked.connect(lambda checked=False, t=tid: self.node.select_target(t))
            layout.addWidget(btn)

    def _refresh_fixed_targets_panel(self, fixed_targets_3d):
        # Same "small, bounded list -> cheap full rebuild every tick" call
        # as _refresh_targets_panel above.
        layout = self.fixed_targets_list_layout
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not fixed_targets_3d:
            lbl = QtWidgets.QLabel('No confirmed targets yet')
            lbl.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
            lbl.setFont(_ui_font(7))
            layout.addWidget(lbl)
            return

        for tid in sorted(fixed_targets_3d.keys()):
            x, y, z = fixed_targets_3d[tid]
            lbl = QtWidgets.QLabel(f'#{tid}   x={x:+.1f}  y={y:+.1f}  z={z:+.1f} m')
            lbl.setStyleSheet(f'color: {FIXED_TARGET_COLOR_HEX}; background: transparent;')
            lbl.setFont(_mono_font(8))
            layout.addWidget(lbl)

    # ------------------------------------------------------------------ #
    # Update loops
    # ------------------------------------------------------------------ #
    def _update_telemetry(self):
        snap = self.node.get_snapshot()
        blink_on = int(time.time() * BLINK_HZ * 2) % 2 == 0

        for key in ('vx', 'vy', 'vz', 'wx', 'wy', 'wz'):
            self.bars[key].set_value(snap[key])

        heading = ((snap['yaw_deg'] + 180) % 360) - 180
        self.compass.set_heading(heading)
        self.heading_label.setText(f'{heading:+.0f}°')

        for key in ('odom', 'imu', 'cam', 'map'):
            self.status_dots[key].set_ok(snap['fresh'][key])
            self.status_labels[key].setText('')

        for key in ('battery', 'px4', 'motors'):
            self.status_dots[key].set_ok(snap['fresh'][key])
            self.status_labels[key].setText('')

        # PX4 flight state
        if snap['fresh']['px4'] and snap['armed'] is not None:
            nav = NAV_STATE_NAMES.get(snap['nav_state'], f"nav_state={snap['nav_state']}")
            armed_txt = 'ARMED' if snap['armed'] else 'DISARMED'
            self.px4_state_label.setText(f'{armed_txt}  |  {nav}')
            ok_color = ACCENT if snap['armed'] else TEXT_DIM
            self.px4_state_label.setStyleSheet(f'color: {ok_color}; background: transparent;')
            if snap['preflight_ok']:
                self.px4_preflight_label.setText('preflight checks: PASS')
                self.px4_preflight_label.setStyleSheet(f'color: {ACCENT}; background: transparent;')
            else:
                self.px4_preflight_label.setText('preflight checks: FAILING (PX4 will refuse to arm)')
                self.px4_preflight_label.setStyleSheet(f'color: {NEGATIVE}; background: transparent;')
        else:
            self.px4_state_label.setText('no PX4 link')
            self.px4_state_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
            self.px4_preflight_label.setText('')

        # Motor outputs (0..1)
        for bar, val in zip(self.motor_bars, snap['motors']):
            bar.setValue(int(round(max(0.0, min(1.0, val)) * 100)) if snap['fresh']['motors'] else 0)

        # Control mode / mission type (display-only)
        if snap['fresh']['mode'] and snap['control_mode']:
            self.control_mode_label.setText(f"mode: {snap['control_mode'].upper()}")
            self.control_mode_label.setStyleSheet(f'color: {ACCENT}; background: transparent;')
        else:
            self.control_mode_label.setText('mode: --  (keyboard_teleop not running)')
            self.control_mode_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')

        if snap['fresh']['battery'] and snap['battery_percentage'] is not None:
            pct = max(0.0, min(1.0, snap['battery_percentage'])) * 100.0
            self.battery_bar.setValue(int(round(pct)))
            volt_txt = f'{snap["battery_voltage"]:.1f} V' if snap['battery_voltage'] is not None else '-- V'
            self.battery_label.setText(f'{pct:.0f} %  /  {volt_txt}')
            # LOW-BATTERY COLOR: flip the bar/text to the alarm color under
            # 20%, same threshold spirit as PX4's own low-battery
            # warning log, so a low pack is visually obvious at a glance
            # rather than only discoverable by reading the number.
            warn = snap['battery_warning']
            low = pct < LOW_BATTERY_FRACTION * 100.0 or warn >= 1
            chunk_color = ALARM_BRIGHT if low else ACCENT
            detail = []
            if snap['battery_current'] is not None:
                detail.append(f"{snap['battery_current']:.1f} A")
            if BATTERY_WARNING_NAMES.get(warn):
                detail.append(BATTERY_WARNING_NAMES[warn])
            if not snap['battery_connected']:
                detail.append('NOT CONNECTED')
            self.battery_detail_label.setText('  |  '.join(detail))
            self.battery_bar.setStyleSheet(f'''
                QProgressBar {{ background: {TRACK}; border: none; border-radius: 3px; }}
                QProgressBar::chunk {{ background: {chunk_color}; border-radius: 3px; }}
            ''')
            self.battery_label.setStyleSheet(f'color: {chunk_color if low else TEXT_DIM}; background: transparent;')
        else:
            self.battery_bar.setValue(0)
            self.battery_label.setText('-- %  /  -- V')
            self.battery_label.setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')
            self.battery_detail_label.setText('')

        if snap['target_status_known']:
            self.status_dots['target'].set_ok(snap['target_detected'])
            self.status_labels['target'].setText('LOCKED' if snap['target_detected'] else 'SEARCHING')
            color = ACCENT if snap['target_detected'] else TEXT_FAINT
            self.status_labels['target'].setStyleSheet(f'color: {color}; background: transparent;')
        else:
            self.status_dots['target'].set_ok(False)
            self.status_labels['target'].setText('NO TRACKER')
            self.status_labels['target'].setStyleSheet(f'color: {TEXT_FAINT}; background: transparent;')

        if snap['target_detected']:
            self.alert_label.show()
            bg = ALARM_BRIGHT if blink_on else ALARM_DIM
            self.alert_label.setStyleSheet(f'background: {bg}; color: white; padding: 4px;')
        else:
            self.alert_label.hide()

        self._refresh_targets_panel(snap['targets'], snap['active_target_id'], snap['selected_target_id'])
        self._refresh_fixed_targets_panel(snap['map_targets_3d_fixed'])

        self.cam_status_dot.set_ok(snap['fresh']['cam'])
        self.cam_status_label.setText('LIVE' if snap['fresh']['cam'] else 'NO SIGNAL')
        color = ACCENT if snap['fresh']['cam'] else TEXT_FAINT
        self.cam_status_label.setStyleSheet(f'color: {color}; background: transparent;')

        img = snap['image']
        if img is not None:
            self._render_camera_frame(img, snap['targets'], snap['active_target_id'], blink_on)

    def _render_camera_frame(self, img, targets, active_id, blink_on):
        h, w, _ = img.shape
        qimg = QtGui.QImage(img.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
        target_w = max(self.camera_label.width(), 1)
        target_h = max(self.camera_label.height(), 1)
        qimg = qimg.scaled(target_w, target_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

        if targets:
            qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
            painter = QtGui.QPainter(qimg)
            painter.setRenderHint(QtGui.QPainter.Antialiasing)
            img_w, img_h = qimg.width(), qimg.height()
            for t in targets:
                tid = t['id']
                x, y, bw, bh = t['bbox']
                left, top = x * img_w, y * img_h
                rw, rh = bw * img_w, bh * img_h
                if tid == active_id:
                    color = QtGui.QColor(ALARM_BRIGHT if blink_on else ALARM_DIM)
                    pen_w, label, label_w = 3, 'TARGET', 58
                else:
                    color = QtGui.QColor(TEXT_DIM)
                    pen_w, label, label_w = 2, f'#{tid}', 26
                painter.setPen(QtGui.QPen(color, pen_w))
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawRect(QtCore.QRectF(left, top, rw, rh))
                label_y = max(0, top - 16)
                painter.fillRect(QtCore.QRectF(left, label_y, label_w, 14), color)
                painter.setPen(QtGui.QColor(BG))
                painter.setFont(_ui_font(7))
                painter.drawText(QtCore.QRectF(left + 4, label_y, label_w - 4, 14), QtCore.Qt.AlignVCenter, label)
            painter.end()

        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def _update_map(self):
        # Only bother re-rendering the map while its tab is actually the
        # visible one.
        if self.tabs.currentIndex() != 1:
            return
        snap = self.node.get_map_snapshot()
        # octomap_server publishes at whatever rate its input cloud
        # arrives, which may be faster than MAP_REDRAW_MS, or may sit
        # idle once the drone stops discovering new voxels. Either way,
        # re-uploading byte-for-byte the same data to the GPU on a fixed
        # timer is still wasted work even though it's much cheaper than
        # the old matplotlib path — skip it outright when nothing changed.
        changed = (
            snap['map_points_t'] != self._last_rendered_map_points_t or
            snap['map_targets_3d_t'] != self._last_rendered_map_targets_t or
            snap['map_targets_3d_fixed_t'] != self._last_rendered_map_targets_fixed_t or
            snap['map_targets_3d_moving_t'] != self._last_rendered_map_targets_moving_t or
            snap['control_mode'] != self._last_rendered_control_mode or
            snap['active_target_id'] != self._last_rendered_active_id or
            snap['selected_target_id'] != self._last_rendered_selected_id or
            snap['trail_t'] != self._last_rendered_trail_t or
            snap['mission_waypoints_t'] != self._last_rendered_mission_waypoints_t
        )
        if not changed:
            return
        self.map_view.render_map(
            snap['map_points'], snap['map_targets_3d'],
            snap['active_target_id'], snap['selected_target_id'],
            trail=snap['trail'], waypoints=snap['mission_waypoints'],
            fixed_targets_3d=snap['map_targets_3d_fixed'],
            moving_targets_3d=snap['map_targets_3d_moving'], control_mode=snap['control_mode'],
        )
        self._last_rendered_map_points_t = snap['map_points_t']
        self._last_rendered_map_targets_t = snap['map_targets_3d_t']
        self._last_rendered_map_targets_fixed_t = snap['map_targets_3d_fixed_t']
        self._last_rendered_map_targets_moving_t = snap['map_targets_3d_moving_t']
        self._last_rendered_control_mode = snap['control_mode']
        self._last_rendered_active_id = snap['active_target_id']
        self._last_rendered_selected_id = snap['selected_target_id']
        self._last_rendered_trail_t = snap['trail_t']
        self._last_rendered_mission_waypoints_t = snap['mission_waypoints_t']


def _spin_ros(node):
    rclpy.spin(node)


def main(args=None):
    # rclpy.init() installs its own SIGINT handler by default, which shuts
    # the context down asynchronously the moment Ctrl+C is pressed. That
    # collides with the background spin thread below and with our own
    # shutdown sequence further down. Disable it and own shutdown ourselves.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DroneGuiNode()

    spin_thread = threading.Thread(target=_spin_ros, args=(node,), daemon=True)
    spin_thread.start()

    app = QtWidgets.QApplication(sys.argv)
    window = DroneGuiApp(node)
    window.show()

    # Python only delivers signal handlers between bytecode instructions
    # on the main thread, and Qt's C++ event loop can otherwise block that
    # indefinitely. A short-interval no-op timer keeps handing control
    # back to the interpreter often enough for Ctrl+C to be noticed
    # promptly, mirroring the old Tkinter .after()-callback trick.
    keepalive = QtCore.QTimer()
    keepalive.timeout.connect(lambda: None)
    keepalive.start(200)
    signal.signal(signal.SIGINT, lambda signum, frame: app.quit())

    try:
        app.exec_()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
