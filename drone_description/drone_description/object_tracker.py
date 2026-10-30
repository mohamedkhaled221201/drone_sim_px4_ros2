#!/usr/bin/env python3
"""
object_tracker.py

Finds targets in the drone's forward camera feed and drives the drone to keep
the selected one centred in view at a roughly constant standoff distance.

DETECTION: pluggable backend (object_detector.py), selected by
`detector_backend`: 'hsv' (default, colour threshold + shape filters),
'dnn' (MobileNet-SSD via OpenCV), or 'auto' (dnn, falling back to hsv).
Every backend returns the same Detection(area, x, y, w, h) list, so tracking,
selection, re-identification and control are backend-agnostic.

TRACKING: frame-to-frame nearest-centroid matching gives each physical box a
stable integer id; an appearance-histogram gallery (ReIdentifier) may reclaim
an old id for a returning box. Tracks unseen for `track_expiry_sec` are dropped.

SELECTION: /drone/selected_target_id (Int32). -1 = auto (nearest by estimated
3D range, preferring movers listed on /map/targets_3d_moving, with switching
hysteresis). A specific id locks to that id and never fails over to another.

BEHAVIOUR:
  - Target visible: yaw to centre it, hold a standoff (standoff_horizontal_m
    back, standoff_altitude_m above), forward speed from the pinhole range
    estimate (bbox width), vertical loop holds the look-down setpoint.
  - Target lost: approach its last known map position if known, else spin to
    search in the direction it was last turning.

DISTANCE: distance_m = target_width_m / (2 * bbox_width_frac * tan(hfov/2)).

Publishes:
  geometry_msgs/Twist        /drone/tracker_cmd
  std_msgs/Bool              /drone/target_detected
  std_msgs/Float32MultiArray /drone/target_bbox      [x, y, w, h] normalised
  std_msgs/Int32             /drone/active_target_id
  std_msgs/Float32MultiArray /drone/targets          [id, x, y, w, h, ...] normalised.
        Published straight from the camera callback (no timer latency). The
        message's layout.dim[0].label carries the IMAGE CAPTURE TIME as
        "sec:nanosec" so semantic_map_node can use the drone pose at capture
        time. The layout change is backward compatible (consumers read .data).

Subscribes:
  sensor_msgs/Image /model/drone/camera, sensor_msgs/Imu /model/drone/imu,
  std_msgs/Int32 /drone/selected_target_id, nav_msgs/Odometry,
  /map/targets_3d, /map/targets_3d_fixed, /map/targets_3d_moving,
  geometry_msgs/PointStamped /drone/approach_goal
"""
import math
import time

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import Imu
from collections import deque

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, Float32MultiArray, Int32, MultiArrayDimension

from .object_detector import build_detector, ReIdentifier


def _quat_to_pitch(x, y, z, w):
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    return math.asin(sinp)


class ObjectTracker(Node):

    def __init__(self):
        super().__init__('object_tracker')

        # --- Detection backend (see object_detector.py) ---
        self.declare_parameter('detector_backend', 'hsv')
        self.declare_parameter('max_targets', 8)
        # OpenCV HSV (H 0-179, S/V 0-255). Yellow paint (0.9, 0.7, 0.1); the
        # saturation floor (140) is what keeps dry-grass/dirt terrain out.
        self.declare_parameter('hsv_lower', [8, 140, 100])
        self.declare_parameter('hsv_upper', [35, 255, 255])
        self.declare_parameter('min_contour_area_px', 50)
        self.declare_parameter('morph_kernel_size', 5)
        # Ground-patch rejection (shape filters, see HSVDetector docstring).
        self.declare_parameter('min_solidity', 0.85)
        self.declare_parameter('min_aspect_ratio', 0.35)
        self.declare_parameter('max_aspect_ratio', 3.0)
        self.declare_parameter('min_extent', 0.5)
        # 0.0 = texture filter disabled (ROS params cannot default to None).
        self.declare_parameter('max_texture_variance', 0.0)
        self.declare_parameter('debug_detection', False)
        self.declare_parameter('dnn_model_path', '')
        self.declare_parameter('dnn_config_path', '')
        self.declare_parameter('dnn_target_classes', ['person'])
        self.declare_parameter('dnn_confidence_threshold', 0.5)

        self.min_contour_area_px = self.get_parameter('min_contour_area_px').value
        self.debug_detection = self.get_parameter('debug_detection').value
        self.max_targets = self.get_parameter('max_targets').value

        max_texture_variance = self.get_parameter('max_texture_variance').value
        if max_texture_variance <= 0.0:
            max_texture_variance = None

        detector_kwargs = dict(
            hsv_lower=self.get_parameter('hsv_lower').value,
            hsv_upper=self.get_parameter('hsv_upper').value,
            min_contour_area_px=self.min_contour_area_px,
            morph_kernel_size=self.get_parameter('morph_kernel_size').value,
            max_targets=self.max_targets,
            min_solidity=self.get_parameter('min_solidity').value,
            min_aspect_ratio=self.get_parameter('min_aspect_ratio').value,
            max_aspect_ratio=self.get_parameter('max_aspect_ratio').value,
            min_extent=self.get_parameter('min_extent').value,
            max_texture_variance=max_texture_variance,
            debug=self.debug_detection,
        )
        dnn_model_path = self.get_parameter('dnn_model_path').value
        dnn_config_path = self.get_parameter('dnn_config_path').value
        if dnn_model_path and dnn_config_path:
            detector_kwargs.update(
                model_path=dnn_model_path,
                config_path=dnn_config_path,
                target_classes=self.get_parameter('dnn_target_classes').value,
                confidence_threshold=self.get_parameter('dnn_confidence_threshold').value,
            )
        self.detector = build_detector(
            backend=self.get_parameter('detector_backend').value,
            logger=self.get_logger(),
            **detector_kwargs,
        )

        # --- Re-identification ---
        self.declare_parameter('reid_enabled', True)
        self.declare_parameter('reid_similarity_threshold', 0.7)
        self.declare_parameter('reid_gallery_timeout_sec', 15.0)
        self.reid_enabled = self.get_parameter('reid_enabled').value
        self.reidentifier = ReIdentifier(
            similarity_threshold=self.get_parameter('reid_similarity_threshold').value,
            gallery_timeout_sec=self.get_parameter('reid_gallery_timeout_sec').value,
        )

        self.get_logger().info(
            f'Object detector ready: backend={type(self.detector).__name__}, '
            f're-id {"enabled" if self.reid_enabled else "disabled"}'
        )

        # --- Multi-target tracking ---
        self.declare_parameter('track_match_max_dist_frac', 0.15)
        self.declare_parameter('track_expiry_sec', 1.5)
        self.track_match_max_dist_frac = self.get_parameter('track_match_max_dist_frac').value
        self.track_expiry_sec = self.get_parameter('track_expiry_sec').value

        # tid -> {'cx', 'cy', 'last_seen', 'bbox', 'hist'}
        self.tracks = {}
        self._next_track_id = 0
        # tid -> (x, y, w, h) in PIXELS, this frame only.
        self.last_frame_targets = {}
        self.last_frame_img_size = (0, 0)
        self.last_camera_time = 0.0
        # Header stamp of the last processed image (published with /drone/targets).
        self.last_frame_stamp = None

        self.selected_id = -1

        # --- Control gains / limits ---
        self.declare_parameter('kp_yaw', 2.8)
        self.declare_parameter('max_yaw_rate_rps', 2.2)
        self.declare_parameter('kd_yaw', 0.9)
        self.declare_parameter('yaw_deadband', 0.03)
        self.declare_parameter('kp_lateral', 1.1)
        self.declare_parameter('lateral_deadband_extra', 0.12)
        self.declare_parameter('max_lateral_cmd', 0.4)

        # --- Distance estimation (pinhole model) ---
        self.declare_parameter('camera_hfov_deg', 80.0)
        self.declare_parameter('target_width_m', 3.0)
        # Fixed camera mount tilt (drone_urdf.xacro camera_gimbal_joint pitch);
        # must match semantic_map_node's gimbal_pitch_rad.
        self.declare_parameter('camera_mount_pitch_rad', 0.559)

        # --- Standoff (Livox Mid-360 FOV-driven: 20 m up / 30 m back ->
        #     33.7 deg look-down, slant range 36 m). Scale both together. ---
        self.declare_parameter('standoff_horizontal_m', 30.0)
        self.declare_parameter('standoff_altitude_m', 20.0)
        self.declare_parameter('kp_forward_distance', 0.06)
        self.declare_parameter('kd_forward_distance', 0.015)
        self.declare_parameter('distance_deadband_m', 1.5)
        self.declare_parameter('distance_ema_alpha', 0.15)
        self.declare_parameter('distance_median_window', 5)
        self.declare_parameter('max_distance_rate_mps', 2.0)
        # Softer backwards tilt while retreating (keeps target in frame).
        self.declare_parameter('retreat_speed_scale', 0.6)
        # 0.0 disables the lateral assist while retreating (avoids roll rocking).
        self.declare_parameter('retreat_lateral_scale', 0.0)
        self.declare_parameter('kp_vertical', 0.6)
        self.declare_parameter('kd_vertical', 0.08)
        self.declare_parameter('vertical_deadband', 0.08)
        self.declare_parameter('max_vertical_cmd', 0.5)
        self.declare_parameter('lost_timeout_sec', 0.6)
        self.declare_parameter('search_yaw_rate_rps', 0.5)
        self.declare_parameter('publish_rate_hz', 20.0)

        # --- APPROACH (far / out-of-sight target) ---
        self.declare_parameter('auto_prefer_moving', True)
        self.declare_parameter('approach_enabled', True)
        self.declare_parameter('approach_arrival_margin_m', 5.0)
        self.declare_parameter('approach_kp_distance', 0.05)
        self.declare_parameter('approach_max_forward', 1.0)
        self.declare_parameter('approach_kp_yaw', 1.2)
        self.declare_parameter('approach_max_yaw_rate_rps', 0.8)
        self.declare_parameter('approach_kp_altitude', 0.12)
        self.declare_parameter('approach_max_vertical', 0.6)
        self.declare_parameter('approach_goal_max_age_sec', 180.0)

        # Auto-selection hysteresis.
        self.declare_parameter('auto_switch_margin', 1.2)
        self.declare_parameter('auto_switch_dwell_sec', 1.0)

        # --- Oscillation damping ---
        self.declare_parameter('ema_alpha', 0.35)
        # IMU pitch is positive nose-down; gain -1.0 cancels the pitch->dy coupling.
        self.declare_parameter('pitch_compensation_gain', -1.0)
        self.declare_parameter('image_aspect_ratio', 640.0 / 480.0)

        # --- Pitch stability (slew limits, rate filter) ---
        self.declare_parameter('distance_rate_ema_alpha', 0.1)
        self.declare_parameter('max_forward_slew_per_sec', 0.3)
        self.declare_parameter('max_lateral_slew_per_sec', 0.3)
        self.declare_parameter('max_vertical_slew_per_sec', 0.5)
        self.declare_parameter('max_yaw_slew_rad_per_sec2', 4.0)

        g = self.get_parameter
        self.kp_yaw = g('kp_yaw').value
        self.max_yaw_rate = g('max_yaw_rate_rps').value
        self.kd_yaw = g('kd_yaw').value
        self.yaw_deadband = g('yaw_deadband').value
        self.kp_lateral = g('kp_lateral').value
        self.lateral_deadband_extra = g('lateral_deadband_extra').value
        self.max_lateral_cmd = g('max_lateral_cmd').value
        self.camera_hfov_deg = g('camera_hfov_deg').value
        self.target_width_m = g('target_width_m').value
        self.camera_mount_pitch_rad = g('camera_mount_pitch_rad').value
        self.hfov_tan_half = math.tan(math.radians(self.camera_hfov_deg) / 2.0)
        self.standoff_horizontal_m = g('standoff_horizontal_m').value
        self.standoff_altitude_m = g('standoff_altitude_m').value
        self.distance_threshold_m = math.hypot(self.standoff_horizontal_m, self.standoff_altitude_m)
        self.kp_forward_distance = g('kp_forward_distance').value
        self.kd_forward_distance = g('kd_forward_distance').value
        self.retreat_speed_scale = g('retreat_speed_scale').value
        self.retreat_lateral_scale = g('retreat_lateral_scale').value
        self.kp_vertical = g('kp_vertical').value
        self.kd_vertical = g('kd_vertical').value
        self.vertical_deadband = g('vertical_deadband').value
        self.max_vertical_cmd = g('max_vertical_cmd').value
        self.lost_timeout = g('lost_timeout_sec').value
        self.search_yaw_rate = g('search_yaw_rate_rps').value
        self.auto_switch_margin = g('auto_switch_margin').value
        self.auto_switch_dwell_sec = g('auto_switch_dwell_sec').value
        self.ema_alpha = g('ema_alpha').value
        self.pitch_compensation_gain = g('pitch_compensation_gain').value
        self.distance_rate_alpha = g('distance_rate_ema_alpha').value
        self.max_forward_slew = g('max_forward_slew_per_sec').value
        self.max_lateral_slew = g('max_lateral_slew_per_sec').value
        self.max_vertical_slew = g('max_vertical_slew_per_sec').value
        self.max_yaw_slew = g('max_yaw_slew_rad_per_sec2').value
        self.distance_rate_f = 0.0
        self._out_x = self._out_y = self._out_z = self._out_yaw = 0.0
        self._last_pub_t = time.monotonic()
        self.image_aspect_ratio = g('image_aspect_ratio').value
        half_hfov_rad = math.radians(self.camera_hfov_deg) / 2.0
        self.half_vfov_rad = math.atan(math.tan(half_hfov_rad) / self.image_aspect_ratio)

        # Vertical setpoint: the target is held at a fixed look-down angle,
        # measured from horizontal, minus the camera's own mount tilt.
        look_down_angle_rad = math.atan2(self.standoff_altitude_m, self.standoff_horizontal_m)
        angle_beyond_mount_rad = look_down_angle_rad - self.camera_mount_pitch_rad
        if self.half_vfov_rad > 1e-6:
            self.dy_setpoint = max(-1.0, min(1.0, angle_beyond_mount_rad / self.half_vfov_rad))
        else:
            self.dy_setpoint = 0.0
        if abs(angle_beyond_mount_rad) > self.half_vfov_rad:
            self.get_logger().warn(
                f'standoff geometry (altitude={self.standoff_altitude_m:.0f}m, '
                f'horizontal={self.standoff_horizontal_m:.0f}m) implies a look-down '
                f'angle {math.degrees(angle_beyond_mount_rad):.1f} deg beyond the '
                f'camera\'s mount tilt, which exceeds the vertical FOV '
                f'(+-{math.degrees(self.half_vfov_rad):.1f} deg) -- dy_setpoint is '
                f'clamped to the frame edge. Widen standoff_horizontal_m or reduce '
                f'standoff_altitude_m.'
            )

        self.last_dx = 0.0
        self.last_dy = 0.0
        self.dx_f = 0.0
        self.dy_f = 0.0
        self.distance_f = 0.0
        self.prev_dy = 0.0
        self.prev_dx = 0.0
        self.prev_distance_m = 0.0
        self.was_visible = False
        self.pitch = 0.0
        # Search spin continues in the direction the target was last turning.
        self.last_yaw_sign = 1.0
        self.last_bbox = (0.0, 0.0, 0.0, 0.0)

        # Auto-select hysteresis state.
        self.auto_track_id = None
        self._last_auto_switch_time = 0.0

        # Distance robustness + approach state.
        self._dist_hist = deque(maxlen=int(g('distance_median_window').value))
        self.distance_ema_alpha = g('distance_ema_alpha').value
        self.distance_deadband = g('distance_deadband_m').value
        self.max_distance_rate = g('max_distance_rate_mps').value
        self.approach_enabled = g('approach_enabled').value
        self.auto_prefer_moving = g('auto_prefer_moving').value
        self._moving_map = {}
        self.approach_arrival_margin = g('approach_arrival_margin_m').value
        self.approach_kp_distance = g('approach_kp_distance').value
        self.approach_max_forward = g('approach_max_forward').value
        self.approach_kp_yaw = g('approach_kp_yaw').value
        self.approach_max_yaw = g('approach_max_yaw_rate_rps').value
        self.approach_kp_alt = g('approach_kp_altitude').value
        self.approach_max_vert = g('approach_max_vertical').value
        self.approach_goal_max_age = g('approach_goal_max_age_sec').value
        self._odom_xyz = None
        self._odom_yaw = 0.0
        self._live_map = {}
        self._fixed_map = {}
        self._goal_cache = {}
        self._manual_goal = None
        self._approaching = False

        self.pub = self.create_publisher(Twist, '/drone/tracker_cmd', 10)
        self.detected_pub = self.create_publisher(Bool, '/drone/target_detected', 10)
        self.bbox_pub = self.create_publisher(Float32MultiArray, '/drone/target_bbox', 10)
        self.active_id_pub = self.create_publisher(Int32, '/drone/active_target_id', 10)
        self.targets_pub = self.create_publisher(Float32MultiArray, '/drone/targets', 10)
        self.create_subscription(RosImage, '/model/drone/camera', self._camera_cb, 10)
        self.create_subscription(Imu, '/model/drone/imu', self._imu_cb, 10)
        self.create_subscription(Int32, '/drone/selected_target_id', self._select_cb, 10)
        self.create_subscription(Odometry, '/model/drone/odometry', self._odom_cb, 10)
        self.create_subscription(Float32MultiArray, '/map/targets_3d', self._live_map_cb, 10)
        self.create_subscription(Float32MultiArray, '/map/targets_3d_fixed', self._fixed_map_cb, 10)
        self.create_subscription(Float32MultiArray, '/map/targets_3d_moving', self._moving_map_cb, 10)
        self.create_subscription(PointStamped, '/drone/approach_goal', self._goal_cb, 10)

        self.publish_rate_hz = g('publish_rate_hz').value
        self.create_timer(1.0 / self.publish_rate_hz, self._publish)

        self.get_logger().info(
            'Object tracker ready. Publishing /drone/tracker_cmd, /drone/target_detected, '
            '/drone/target_bbox, /drone/active_target_id and /drone/targets (image-stamped, '
            'published from the camera callback). Default selection is auto; publish an id '
            "on /drone/selected_target_id to lock onto one. Switch the drone to 'auto' mode "
            "(press 't' in keyboard_teleop) to use it."
        )

    def _select_cb(self, msg):
        if msg.data != self.selected_id:
            self.get_logger().info(
                f'target selection changed: {self.selected_id} -> {msg.data} '
                f"({'auto' if msg.data == -1 else 'id ' + str(msg.data)})"
            )
        self.selected_id = msg.data

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z, q.x, q.y, q.z, q.w)):
            return
        self._odom_xyz = (p.x, p.y, p.z)
        self._odom_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                    1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _parse_targets(msg):
        d = msg.data
        return {int(round(d[i])): (d[i + 1], d[i + 2], d[i + 3]) for i in range(0, len(d) - 3, 4)}

    def _live_map_cb(self, msg):
        self._live_map = self._parse_targets(msg)
        now = time.monotonic()
        for tid, xyz in self._live_map.items():
            self._goal_cache[tid] = (xyz, now)

    def _moving_map_cb(self, msg):
        self._moving_map = self._parse_targets(msg)

    def _fixed_map_cb(self, msg):
        self._fixed_map = self._parse_targets(msg)

    def _goal_cb(self, msg):
        p = msg.point
        if all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            self._manual_goal = (p.x, p.y, p.z)
        else:
            self._manual_goal = None   # publish NaN to clear

    def _approach_goal(self, active_id):
        """Best known target position (odom frame), or None."""
        if self._manual_goal is not None:
            return self._manual_goal
        now = time.monotonic()
        if (self.selected_id == -1 and self.auto_prefer_moving and self._moving_map
                and self._odom_xyz is not None):
            ox, oy, _ = self._odom_xyz
            return min(self._moving_map.values(), key=lambda p: math.hypot(p[0] - ox, p[1] - oy))
        tid = self.selected_id if self.selected_id != -1 else (
            self.auto_track_id if self.auto_track_id is not None else active_id)
        if tid is not None and tid in self._goal_cache:
            xyz, t = self._goal_cache[tid]
            if (now - t) <= self.approach_goal_max_age:
                return xyz
        if self.selected_id == -1 and self._fixed_map and self._odom_xyz is not None:
            ox, oy, _ = self._odom_xyz
            return min(self._fixed_map.values(), key=lambda p: math.hypot(p[0] - ox, p[1] - oy))
        return None

    def _approach_cmd(self, goal):
        """(forward, vertical, yaw) toward goal, or None once inside the
        standoff (then the normal search spin takes over)."""
        if not self.approach_enabled or self._odom_xyz is None:
            return None
        x, y, z = self._odom_xyz
        gx, gy, gz = goal
        dist_h = math.hypot(gx - x, gy - y)
        if dist_h <= self.standoff_horizontal_m + self.approach_arrival_margin:
            return None
        yaw_err = math.atan2(gy - y, gx - x) - self._odom_yaw
        yaw_err = math.atan2(math.sin(yaw_err), math.cos(yaw_err))
        yaw_cmd = max(-self.approach_max_yaw, min(self.approach_max_yaw, self.approach_kp_yaw * yaw_err))
        facing = max(0.0, math.cos(yaw_err)) ** 2 if abs(yaw_err) < math.radians(60) else 0.0
        fwd = min(self.approach_max_forward,
                  self.approach_kp_distance * (dist_h - self.standoff_horizontal_m)) * facing
        alt_err = (gz + self.standoff_altitude_m) - z
        vert = max(-self.approach_max_vert, min(self.approach_max_vert, self.approach_kp_alt * alt_err))
        return fwd, vert, yaw_cmd

    def _imu_cb(self, msg):
        q = msg.orientation
        self.pitch = _quat_to_pitch(q.x, q.y, q.z, q.w)

    def _publish_targets(self):
        """Publish every currently-visible candidate, stamped with the image
        capture time (layout.dim[0].label = 'sec:nanosec'), straight from the
        camera callback so there is no timer latency."""
        m = Float32MultiArray()
        img_w, img_h = self.last_frame_img_size
        flat = []
        if img_w > 0 and img_h > 0:
            for tid, (x, y, w, h) in self.last_frame_targets.items():
                flat.extend([float(tid), x / img_w, y / img_h, w / img_w, h / img_h])
        m.data = flat
        st = self.last_frame_stamp
        if st is not None:
            dim = MultiArrayDimension()
            dim.label = f'{st.sec}:{st.nanosec}'
            dim.size = 0
            dim.stride = 0
            m.layout.dim = [dim]
        self.targets_pub.publish(m)

    def _camera_cb(self, msg):
        try:
            frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        except Exception as e:
            self.get_logger().warn(f'camera frame decode failed: {e}')
            return
        stamp = msg.header.stamp

        img_h, img_w = frame.shape[:2]

        detections = self.detector.detect(frame)

        now = time.monotonic()
        img_diag = math.hypot(img_w, img_h)
        match_thresh_px = self.track_match_max_dist_frac * img_diag
        det_centroids = [(d.x + d.w / 2.0, d.y + d.h / 2.0) for d in detections]

        # Greedy nearest-centroid matching of existing tracks to detections.
        used_dets = set()
        matched = {}  # tid -> detection index
        for tid, tinfo in self.tracks.items():
            best_j, best_d = None, None
            for j, (cx, cy) in enumerate(det_centroids):
                if j in used_dets:
                    continue
                d = math.hypot(cx - tinfo['cx'], cy - tinfo['cy'])
                if d <= match_thresh_px and (best_d is None or d < best_d):
                    best_j, best_d = j, d
            if best_j is not None:
                used_dets.add(best_j)
                matched[tid] = best_j

        current_frame_targets = {}  # tid -> (x, y, w, h) px, this frame only

        for tid, j in matched.items():
            det = detections[j]
            cx, cy = det_centroids[j]
            self.tracks[tid]['cx'] = cx
            self.tracks[tid]['cy'] = cy
            self.tracks[tid]['last_seen'] = now
            self.tracks[tid]['bbox'] = (det.x, det.y, det.w, det.h)
            # Keep the appearance signature fresh for re-ID.
            if self.reid_enabled:
                self.tracks[tid]['hist'] = self.reidentifier.compute_histogram(
                    frame, self.tracks[tid]['bbox'])
            current_frame_targets[tid] = self.tracks[tid]['bbox']

        for j, det in enumerate(detections):
            if j in used_dets:
                continue
            bbox = (det.x, det.y, det.w, det.h)
            cx, cy = det_centroids[j]

            # Try to reclaim a recently-lost id before minting a new one.
            tid = self.reidentifier.match(frame, bbox, now) if self.reid_enabled else None
            if tid is not None:
                self.get_logger().info(f'track id={tid} re-identified after being lost')
            else:
                tid = self._next_track_id
                self._next_track_id += 1
                if self.debug_detection:
                    self.get_logger().info(f'new track id={tid} area={det.area:.0f} bbox={bbox}')

            hist = self.reidentifier.compute_histogram(frame, bbox) if self.reid_enabled else None
            self.tracks[tid] = {'cx': cx, 'cy': cy, 'last_seen': now, 'bbox': bbox, 'hist': hist}
            current_frame_targets[tid] = bbox

        # Drop stale tracks, offering their last appearance to the re-identifier.
        expired = [t for t, info in self.tracks.items()
                   if (now - info['last_seen']) > self.track_expiry_sec]
        for tid in expired:
            if self.reid_enabled:
                self.reidentifier.offer_hist(tid, self.tracks[tid].get('hist'), now)
            del self.tracks[tid]

        self.last_frame_targets = current_frame_targets
        self.last_frame_img_size = (img_w, img_h)
        self.last_camera_time = now
        self.last_frame_stamp = stamp
        self._publish_targets()

        if self.debug_detection and current_frame_targets:
            self.get_logger().info(
                f'frame: {len(current_frame_targets)} target(s) visible: '
                f'{sorted(current_frame_targets.keys())}'
            )

    def _estimate_distance_m(self, bbox_w_px, img_w):
        """Pinhole slant-range estimate from bbox width; inf for degenerate boxes."""
        bbox_w_frac = bbox_w_px / img_w if img_w > 0 else 0.0
        if bbox_w_frac > 1e-4:
            return self.target_width_m / (2.0 * bbox_w_frac * self.hfov_tan_half)
        return float('inf')

    def _select_active_target(self):
        """Pick which track to follow this frame. Returns (id, bbox_px) or
        (None, None). Auto mode: nearest by estimated 3D range (movers first
        when auto_prefer_moving), with switch hysteresis. An explicit
        selection never fails over to a different id."""
        frame_fresh = (time.monotonic() - self.last_camera_time) < self.lost_timeout
        if not frame_fresh or not self.last_frame_targets:
            return None, None

        if self.selected_id == -1:
            targets = self.last_frame_targets
            if self.auto_prefer_moving:
                movers = {t: b for t, b in targets.items() if t in self._moving_map}
                if movers:
                    targets = movers
            img_w, _img_h = self.last_frame_img_size
            distances = {tid: self._estimate_distance_m(bbox[2], img_w)
                         for tid, bbox in targets.items()}
            best_id = min(distances, key=distances.get)

            now = time.monotonic()
            current = self.auto_track_id
            still_visible = current is not None and current in distances
            within_dwell = (now - self._last_auto_switch_time) < self.auto_switch_dwell_sec
            if still_visible and within_dwell:
                if distances[best_id] * self.auto_switch_margin > distances[current]:
                    best_id = current

            if best_id != self.auto_track_id:
                self._last_auto_switch_time = now
                # A genuine switch is a reacquisition of a different target:
                # make the EMA filters snap instead of blending.
                self.was_visible = False

            self.auto_track_id = best_id
            return best_id, targets[best_id]

        if self.selected_id in self.last_frame_targets:
            return self.selected_id, self.last_frame_targets[self.selected_id]

        return None, None

    def _publish(self):
        msg = Twist()

        # /drone/targets is now published from _camera_cb (image-stamped).
        # Here only clear the GUI list if the camera has stopped.
        frame_fresh = (time.monotonic() - self.last_camera_time) < self.lost_timeout
        if not frame_fresh:
            self.targets_pub.publish(Float32MultiArray())

        active_id, active_bbox_px = self._select_active_target()
        target_visible = active_bbox_px is not None
        self.active_id_pub.publish(Int32(data=active_id if active_id is not None else -1))

        if target_visible:
            img_w, img_h = self.last_frame_img_size
            x, y, w, h = active_bbox_px
            cx = x + w / 2.0
            cy = y + h / 2.0
            self.last_dx = (cx - img_w / 2.0) / (img_w / 2.0)
            self.last_dy = (cy - img_h / 2.0) / (img_h / 2.0)
            self.last_bbox = (x / img_w, y / img_h, w / img_w, h / img_h)

        self.detected_pub.publish(Bool(data=target_visible))
        bbox_msg = Float32MultiArray()
        bbox_msg.data = list(self.last_bbox) if target_visible else [0.0, 0.0, 0.0, 0.0]
        self.bbox_pub.publish(bbox_msg)

        if target_visible:
            self._approaching = False
            raw_dx, raw_dy = self.last_dx, self.last_dy
            bbox_w_frac = self.last_bbox[2]

            if bbox_w_frac > 1e-4:
                raw_distance_m = self.target_width_m / (2.0 * bbox_w_frac * self.hfov_tan_half)
            else:
                raw_distance_m = float('inf')

            # A bbox touching the image border is clipped: its width under-reads,
            # so the range over-reads. Don't feed such frames to the range filter.
            bx0, bw0 = self.last_bbox[0], self.last_bbox[2]
            edge_clipped = (bx0 <= 0.005) or (bx0 + bw0 >= 0.995)
            if not self.was_visible:
                self._dist_hist.clear()
                if math.isfinite(raw_distance_m):
                    self._dist_hist.append(raw_distance_m)
                self.dx_f = raw_dx
                self.dy_f = raw_dy
                self.distance_f = raw_distance_m
                # Snap derivative memory so reacquisition doesn't spike.
                self.prev_dy = raw_dy
                self.prev_dx = raw_dx
            else:
                a = self.ema_alpha
                self.dx_f = a * raw_dx + (1 - a) * self.dx_f
                self.dy_f = a * raw_dy + (1 - a) * self.dy_f
                if math.isfinite(raw_distance_m) and not edge_clipped:
                    self._dist_hist.append(raw_distance_m)
                    med = float(np.median(self._dist_hist))
                    ad = self.distance_ema_alpha
                    self.distance_f = ad * med + (1 - ad) * self.distance_f

            dx, dy = self.dx_f, self.dy_f
            distance_m = self.distance_f

            # Yaw: dx > 0 (target right of centre) -> negative angular.z.
            dx_error = 0.0 if abs(dx) < self.yaw_deadband else dx
            d_dx = (dx - self.prev_dx) * self.publish_rate_hz
            self.prev_dx = dx
            yaw_cmd = max(-self.max_yaw_rate, min(self.max_yaw_rate,
                          -self.kp_yaw * dx_error - self.kd_yaw * d_dx))
            if abs(yaw_cmd) > 1e-6:
                self.last_yaw_sign = math.copysign(1.0, yaw_cmd)

            # Lateral assist, only for larger offsets (yaw handles small ones).
            lateral_deadband = self.yaw_deadband + self.lateral_deadband_extra
            lateral_dx = 0.0 if abs(dx) < lateral_deadband else dx
            lateral_cmd = max(-self.max_lateral_cmd,
                               min(self.max_lateral_cmd, -self.kp_lateral * lateral_dx))
            if distance_m < self.distance_threshold_m:
                lateral_cmd *= self.retreat_lateral_scale

            # Forward speed from range error, soft deadband.
            distance_error = distance_m - self.distance_threshold_m
            if abs(distance_error) <= self.distance_deadband:
                distance_error = 0.0
            else:
                distance_error -= math.copysign(self.distance_deadband, distance_error)

            # Closing/receding rate term, low-passed.
            raw_rate = 0.0 if not self.was_visible else (distance_m - self.prev_distance_m) * self.publish_rate_hz
            if not self.was_visible:
                self.distance_rate_f = 0.0
            else:
                ra = self.distance_rate_alpha
                self.distance_rate_f = ra * raw_rate + (1.0 - ra) * self.distance_rate_f
            distance_rate = max(-self.max_distance_rate, min(self.max_distance_rate, self.distance_rate_f))
            self.prev_distance_m = distance_m
            self.was_visible = True

            forward_cmd = self.kp_forward_distance * distance_error + self.kd_forward_distance * distance_rate
            forward_cmd *= max(0.0, 1.0 - abs(dx))
            if forward_cmd < 0.0:
                forward_cmd *= self.retreat_speed_scale
            forward_cmd = max(-1.0, min(1.0, forward_cmd))

            # Feed-forward body-pitch compensation (fixed camera mount).
            if self.pitch_compensation_gain != 0.0 and self.half_vfov_rad > 1e-6:
                dy = dy - self.pitch_compensation_gain * (self.pitch / self.half_vfov_rad)

            # Vertical: hold the target at dy_setpoint (look-down angle).
            dy_offset = dy - self.dy_setpoint
            dy_error = 0.0 if abs(dy_offset) < self.vertical_deadband else dy_offset
            d_dy = (dy - self.prev_dy) * self.publish_rate_hz
            self.prev_dy = dy
            vertical_cmd = -self.kp_vertical * dy_error - self.kd_vertical * d_dy
            vertical_cmd = max(-self.max_vertical_cmd, min(self.max_vertical_cmd, vertical_cmd))

            msg.linear.x = forward_cmd
            msg.linear.y = lateral_cmd
            msg.linear.z = vertical_cmd
            msg.angular.z = yaw_cmd
        else:
            # Lost: approach last known position, else search-spin in the
            # direction the target was last turning.
            self.was_visible = False
            approach = None
            goal = self._approach_goal(active_id)
            if goal is not None:
                approach = self._approach_cmd(goal)
            if approach is not None:
                fwd, vert, yaw_c = approach
                msg.linear.x = fwd
                msg.linear.z = vert
                msg.angular.z = yaw_c
                if not self._approaching:
                    self.get_logger().info(
                        f'Target not visible: approaching last known position '
                        f'({goal[0]:.1f}, {goal[1]:.1f}, {goal[2]:.1f}).')
                self._approaching = True
            else:
                self._approaching = False
                msg.angular.z = self.search_yaw_rate * self.last_yaw_sign

        # Slew-rate limit every output.
        now_t = time.monotonic()
        dt = max(0.001, min(0.2, now_t - self._last_pub_t))
        self._last_pub_t = now_t

        def _slew(prev, target, rate):
            if rate <= 0.0:
                return target
            step = rate * dt
            return prev + max(-step, min(step, target - prev))

        self._out_x = _slew(self._out_x, msg.linear.x, self.max_forward_slew)
        self._out_y = _slew(self._out_y, msg.linear.y, self.max_lateral_slew)
        self._out_z = _slew(self._out_z, msg.linear.z, self.max_vertical_slew)
        self._out_yaw = _slew(self._out_yaw, msg.angular.z, self.max_yaw_slew)
        msg.linear.x = self._out_x
        msg.linear.y = self._out_y
        msg.linear.z = self._out_z
        msg.angular.z = self._out_yaw

        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
