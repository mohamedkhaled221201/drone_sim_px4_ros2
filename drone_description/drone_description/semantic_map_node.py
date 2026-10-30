#!/usr/bin/env python3
"""
semantic_map_node.py

Fuses the drone's 3D lidar map (built by octomap_server, published on
/map/points) with object_tracker's camera detections to produce real 3D world
positions for every tracked target.

CORE IDEA: object_tracker tells us, per frame, which 2D image region (bbox)
each target occupies. Transform the octomap's world-frame points into the
camera frame and project them with the pinhole model; points whose projection
lands inside a bbox are physically part of (or behind) that target. Their
centroid is a real, depth-sensed 3D position for it.

EXTRINSIC -- BASE_LINK TO CAMERA OPTICAL FRAME (from drone_urdf.xacro):
    base_to_gimbal:     xyz=(0.15, 0, 0.02)  rpy=(0, 0, 0)
    camera_gimbal_joint xyz=(0, 0, 0)        rpy=(0, 0.559, 0)   <- look-down tilt
    camera_to_optical:  xyz=(0, 0, 0)        rpy=(-1.5708, 0, -1.5708)
gimbal_pitch_rad below MUST match the xacro's camera_gimbal_joint pitch, or every
fused position is computed with the wrong camera orientation (this drifted once).
Rotation base<-optical = Ry(gimbal_pitch) @ R_CAMERA_TO_OPTICAL.T (the transpose
is required: R_CAMERA_TO_OPTICAL maps camera_link -> optical, we need the reverse).

Subscribes:
  nav_msgs/Odometry              /model/drone/odometry
  sensor_msgs/PointCloud2        /map/points              (octomap_server)
  sensor_msgs/CameraInfo         /model/drone/camera_info
  std_msgs/Float32MultiArray     /drone/targets           (object_tracker; ALL
                                                           visible candidates)

Publishes (all flattened [id, x, y, z, id, x, y, z, ...], world frame):
  /map/targets_3d          currently-live targets
  /map/targets_3d_fixed    every non-moving target ever confirmed, never pruned
                           (fixed_target_merge_dist_m keeps a static target from
                           being re-recorded as a duplicate when its track id churns)
  /map/targets_3d_moving   subset of the live targets currently classified as
                           MOVING (windowed velocity). drone_gui.py draws its
                           tracked-mover marker and path only for ids listed here.
"""
import math
from collections import deque

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32MultiArray

# read_points_numpy reinterprets the raw byte buffer directly (no per-point
# Python loop). Fall back to the structured-array path on older builds.
try:
    from sensor_msgs_py.point_cloud2 import read_points_numpy
    _HAVE_READ_POINTS_NUMPY = True
except ImportError:
    _HAVE_READ_POINTS_NUMPY = False


def _quat_to_rotmat(x, y, z, w):
    """Standard unit-quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _rotation_y(theta):
    """Rotation matrix for a rotation of `theta` radians about Y."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


# camera_link (X-forward, Y-left, Z-up) -> camera_link_optical (X-right, Y-down,
# Z-forward). Equivalent to the xacro's rpy=(-1.5708, 0, -1.5708).
R_CAMERA_TO_OPTICAL = np.array([
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])


class SemanticMapNode(Node):
    """Target localization only; octomap_server owns the 3D occupancy map."""

    def __init__(self):
        super().__init__('semantic_map_node')

        self.declare_parameter('map_frame', 'odom')       # must match octomap_server's frame_id

        # --- camera extrinsic params (see module docstring) ---
        self.declare_parameter('gimbal_offset_xyz', [0.15, 0.0, 0.02])
        self.declare_parameter('gimbal_pitch_rad', 0.559)               # must match drone_urdf.xacro

        # --- target localization params ---
        self.declare_parameter('targets_publish_rate_hz', 5.0)
        # Min lidar points inside a bbox to trust it. Was lowered to 2 and that
        # backfired: two returns on very different parts of a 4.5 m-wide vehicle
        # gave centroids too far apart to merge -> multiple fixed markers for one
        # vehicle. 3 gives a centroid much more representative of the true centre.
        self.declare_parameter('min_points_per_target', 3)
        self.declare_parameter('position_ema_alpha', 0.3)      # smoothing on each target's world_xyz
        self.declare_parameter('target_timeout_sec', 3.0)      # drop a live target if unmatched this long
        # FIXED-TARGET DEDUP: object_tracker can't tell same-colored boxes apart
        # once a track is fully lost and reacquired -- it mints a fresh id. For a
        # STATIC target that would stack ghost dots on one real box. Any
        # newly-confirmed centroid within this radius of an ALREADY-fixed target
        # is treated as a re-sighting of the same object. 5.0 m: above the
        # vehicle's own diagonal (~5.5 m) plus noise is not needed for merging,
        # but comfortably below the ~30 m lane spacing between distinct targets
        # (baylands_dynamic.sdf). Shrink it if targets are ever placed closer.
        self.declare_parameter('fixed_target_merge_dist_m', 5.0)
        # GROUND-PATCH REJECTION (3D stage): a bbox match whose points span more
        # than this on any axis is a broad patch of ground/terrain (or canopy),
        # not a vehicle. Vehicles are 3.9 x 3.0 x 2.7 m (~5.5 m diagonal);
        # 4.5 m is above any legitimate partial view yet far below a terrain patch.
        self.declare_parameter('max_target_extent_m', 4.5)

        # MOTION CLASSIFICATION: proximity dedup assumes a target hasn't moved
        # between sightings. Velocity is measured as NET displacement over
        # motion_window_sec (not frame-to-frame), so zero-mean centroid jitter
        # cancels instead of being amplified into a false "moving" reading.
        # moving_speed_threshold_m_s only has to clear the residual noise floor
        # (and stay under the slowest real patrol speed, ~0.5 m/s).
        self.declare_parameter('motion_window_sec', 1.0)
        self.declare_parameter('moving_speed_threshold_m_s', 0.15)
        # REACQUISITION-BY-MOTION: a detection under a tid not in self._targets
        # is checked against every live target's predicted position (last
        # position + velocity * elapsed). A hit within this gate = same object,
        # churned id; its history is carried forward under the new tid.
        self.declare_parameter('reacquisition_gate_dist_m', 4.0)

        # PERF: octomap republishes its FULL cloud every integration, so crop to a
        # sphere around the drone before the projection matmul. Keep a little
        # above the lidar max range (90 m in octomap_params.yaml).
        self.declare_parameter('fusion_max_range_m', 100.0)

        self.map_frame = self.get_parameter('map_frame').value

        gimbal_offset = np.array(self.get_parameter('gimbal_offset_xyz').value, dtype=np.float64)
        gimbal_pitch = self.get_parameter('gimbal_pitch_rad').value
        # base_link -> camera_link_optical: translation is the gimbal mount point;
        # rotation is the gimbal tilt composed with the TRANSPOSE of
        # R_CAMERA_TO_OPTICAL (ROTATION-DIRECTION BUGFIX, see module docstring).
        self._t_base_optical = gimbal_offset
        self._R_base_optical = _rotation_y(gimbal_pitch) @ R_CAMERA_TO_OPTICAL.T

        self.targets_publish_rate = self.get_parameter('targets_publish_rate_hz').value
        self.min_points_per_target = self.get_parameter('min_points_per_target').value
        self.ema_alpha = self.get_parameter('position_ema_alpha').value
        self.target_timeout = self.get_parameter('target_timeout_sec').value
        self.fixed_target_merge_dist = self.get_parameter('fixed_target_merge_dist_m').value
        self.max_target_extent = self.get_parameter('max_target_extent_m').value
        self.motion_window_sec = self.get_parameter('motion_window_sec').value
        self.moving_speed_threshold = self.get_parameter('moving_speed_threshold_m_s').value
        self.reacquisition_gate_dist = self.get_parameter('reacquisition_gate_dist_m').value
        self.fusion_max_range = self.get_parameter('fusion_max_range_m').value

        # Latest drone pose (world/odom frame); None until first odometry.
        self._latest_pos = None
        self._latest_rotmat = None

        # Latest occupied-voxel centers from octomap_server, world frame, (M, 3).
        self._world_pts = np.empty((0, 3))

        # (fx, fy, cx, cy, width, height); None until first CameraInfo.
        self._cam_intrinsics = None

        # Live per-target state: id -> {'world_xyz', 'last_seen', 'velocity',
        # 'history', 'is_moving'}.
        self._targets = {}

        # FIXED TARGETS: id -> world_xyz for every non-moving target ever
        # confirmed. Never removed by _prune_stale_targets.
        self._fixed_targets = {}

        # /map/points is RELIABLE on the publisher side; a BEST_EFFORT subscriber
        # silently dropped large clouds, so match RELIABLE.
        map_points_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(Odometry, '/model/drone/odometry', self._odom_cb, 10)
        self.create_subscription(PointCloud2, '/map/points', self._map_points_cb, map_points_qos)
        self.create_subscription(CameraInfo, '/model/drone/camera_info', self._caminfo_cb,
                                 qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/drone/targets', self._targets_cb, 10)

        self.targets_pub = self.create_publisher(Float32MultiArray, '/map/targets_3d', 10)
        self.fixed_targets_pub = self.create_publisher(Float32MultiArray, '/map/targets_3d_fixed', 10)
        self.moving_targets_pub = self.create_publisher(Float32MultiArray, '/map/targets_3d_moving', 10)

        self.create_timer(1.0 / self.targets_publish_rate, self._publish_targets)

        self.get_logger().info(
            'Semantic map node started: fusing octomap points ("/map/points") with '
            'object_tracker detections -> /map/targets_3d (live), '
            '/map/targets_3d_fixed (permanent, never pruned), '
            '/map/targets_3d_moving (live movers) '
            f'(min_points_per_target={self.min_points_per_target}, '
            f'target_timeout={self.target_timeout:.1f}s, '
            f'fixed_target_merge_dist={self.fixed_target_merge_dist:.1f}m, '
            f'max_target_extent={self.max_target_extent:.1f}m). '
            'Waiting on odometry + camera_info before targets can localize.'
        )

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        pos = np.array([p.x, p.y, p.z])
        quat = np.array([q.x, q.y, q.z, q.w])
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
            return  # garbage odometry before the first physics step

        norm = np.linalg.norm(quat)
        if norm < 1e-6:
            return  # degenerate quaternion
        quat = quat / norm

        self._latest_pos = pos
        self._latest_rotmat = _quat_to_rotmat(*quat)

    def _map_points_cb(self, msg: PointCloud2):
        pts = self._decode_cloud_xyz(msg)
        if pts is not None:
            self._world_pts = pts

    def _decode_cloud_xyz(self, msg: PointCloud2):
        """Decode x/y/z into a plain (N, 3) float64 array without a per-point
        Python loop; NaN filtering is one vectorized isfinite mask."""
        try:
            if _HAVE_READ_POINTS_NUMPY:
                pts = read_points_numpy(msg, field_names=('x', 'y', 'z'))
            else:
                struct_pts = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'))
                pts = np.stack([struct_pts['x'], struct_pts['y'], struct_pts['z']], axis=-1)
            pts = pts.astype(np.float64, copy=False)
        except Exception as e:
            self.get_logger().warn(f'map points decode failed: {e}')
            return None

        if pts.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return pts[np.all(np.isfinite(pts), axis=1)]

    # ------------------------------------------------------------------ #
    # Camera intrinsics
    # ------------------------------------------------------------------ #
    def _caminfo_cb(self, msg: CameraInfo):
        k = msg.k  # row-major 3x3: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        self._cam_intrinsics = (k[0], k[4], k[2], k[5], msg.width, msg.height)

    # ------------------------------------------------------------------ #
    # Target fusion: project the map into the current camera view and match
    # points to each target's bbox.
    # ------------------------------------------------------------------ #
    def _targets_cb(self, msg: Float32MultiArray):
        if self._latest_pos is None or self._cam_intrinsics is None or self._world_pts.shape[0] == 0:
            return  # nothing to project against yet

        data = msg.data
        n = len(data) // 5
        if n == 0:
            return

        # PERF: bound cost by sensing range, not total explored-map size.
        near_mask = np.linalg.norm(self._world_pts - self._latest_pos, axis=1) <= self.fusion_max_range
        world_pts = self._world_pts[near_mask]  # (M, 3)
        if world_pts.shape[0] == 0:
            self._prune_stale_targets()
            return

        fx, fy, cx, cy, img_w, img_h = self._cam_intrinsics

        # World -> camera-optical frame for every point, in one shot.
        cam_pos_world = self._latest_pos + self._latest_rotmat @ self._t_base_optical
        cam_rot_world = self._latest_rotmat @ self._R_base_optical
        cam_pts = (cam_rot_world.T @ (world_pts - cam_pos_world).T).T  # (M, 3)

        z = cam_pts[:, 2]
        in_front = z > 1e-3  # behind-camera / degenerate points can't project
        if not np.any(in_front):
            self._prune_stale_targets()
            return

        u = np.full_like(z, np.nan)
        v = np.full_like(z, np.nan)
        u[in_front] = fx * cam_pts[in_front, 0] / z[in_front] + cx
        v[in_front] = fy * cam_pts[in_front, 1] / z[in_front] + cy

        now = self.get_clock().now().nanoseconds / 1e9

        for i in range(n):
            base = i * 5
            tid = int(round(data[base]))
            bx, by, bw, bh = data[base + 1:base + 5]

            # Normalized [0,1] top-left bbox -> pixel bounds.
            px0, px1 = bx * img_w, (bx + bw) * img_w
            py0, py1 = by * img_h, (by + bh) * img_h

            in_bbox = (
                in_front
                & (u >= px0) & (u <= px1)
                & (v >= py0) & (v <= py1)
            )
            count = int(np.count_nonzero(in_bbox))
            if count < self.min_points_per_target:
                continue  # not enough depth samples inside this bbox to trust yet

            matched_pts = world_pts[in_bbox]

            # GROUND-PATCH REJECTION (see max_target_extent_m): reject before the
            # match ever reaches _targets/_fixed_targets (fixed ones never prune).
            extent = matched_pts.max(axis=0) - matched_pts.min(axis=0)
            if float(extent.max()) > self.max_target_extent:
                self.get_logger().warn(
                    f'target id={tid}: rejected, matched points span '
                    f'{extent.max():.1f}m (max allowed {self.max_target_extent:.1f}m) '
                    f'-- likely a ground/terrain patch, not the box.',
                    throttle_duration_sec=5.0,
                )
                continue

            centroid = matched_pts.mean(axis=0)

            existing = self._targets.get(tid)

            # REACQUISITION-BY-MOTION: tid is new -- check whether any live
            # target's velocity predicts it should be right about here. If so,
            # carry its history forward under this tid instead of duplicating.
            if existing is None:
                for other_tid, other in self._targets.items():
                    dt_pred = now - other['last_seen']
                    predicted = other['world_xyz'] + other['velocity'] * dt_pred
                    if np.linalg.norm(centroid - predicted) <= self.reacquisition_gate_dist:
                        existing = self._targets.pop(other_tid)
                        break

            if existing is None:
                world_xyz = centroid
                history = deque()
            else:
                a = self.ema_alpha
                world_xyz = a * centroid + (1 - a) * existing['world_xyz']
                history = existing['history']

            history.append((now, world_xyz))
            # Only the last motion_window_sec matters for the velocity estimate.
            while len(history) > 1 and (now - history[0][0]) > self.motion_window_sec:
                history.popleft()

            oldest_t, oldest_xyz = history[0]
            window_span = now - oldest_t
            # WINDOWED VELOCITY: net displacement over the retained window. The
            # window must be at least half-full before it is trusted; until then
            # `settled` is False and the target is treated as not moving.
            settled = window_span >= (self.motion_window_sec * 0.5)
            if settled:
                velocity = (world_xyz - oldest_xyz) / window_span
            else:
                velocity = existing['velocity'] if existing is not None else np.zeros(3)

            is_moving = settled and float(np.linalg.norm(velocity)) > self.moving_speed_threshold

            self._targets[tid] = {
                'world_xyz': world_xyz, 'last_seen': now,
                'velocity': velocity, 'history': history,
                'is_moving': is_moving,
            }

            # Fixed record: never deleted by _prune_stale_targets. MOTION GATE:
            # gated on is_moving alone (not `settled` too) so a static target
            # still gets a marker on its first confirmed sighting; `is_moving`
            # defaults to False until real, settled motion data says otherwise.
            if not is_moving:
                # DEDUP: if another tid already has a fixed marker within merge
                # distance, this is the same static object under a churned id.
                #
                # SETTLED-ONLY MERGE: an unsettled sighting may MINT its own
                # entry but may not merge into (and drag) someone else's -- a
                # passing mover's first, unsettled centroid would otherwise
                # corrupt a real static marker via the EMA blend.
                #
                # RETROACTIVE RECONCILE: also runs when tid already has an
                # entry, so an entry minted while unsettled is folded into the
                # older marker as soon as it settles.
                fixed_tid = tid
                if settled:
                    for other_tid, other_xyz in list(self._fixed_targets.items()):
                        if other_tid == tid:
                            continue
                        if np.linalg.norm(other_xyz - world_xyz) <= self.fixed_target_merge_dist:
                            self._fixed_targets.pop(tid, None)
                            fixed_tid = other_tid
                            break

                # JITTER FIX: blend into the existing fixed position with the same
                # ema_alpha instead of snapping to each new partial-vehicle view.
                existing_fixed = self._fixed_targets.get(fixed_tid)
                if existing_fixed is None:
                    self._fixed_targets[fixed_tid] = world_xyz
                else:
                    a = self.ema_alpha
                    self._fixed_targets[fixed_tid] = a * world_xyz + (1 - a) * existing_fixed

        self._prune_stale_targets()

    def _prune_stale_targets(self):
        now = self.get_clock().now().nanoseconds / 1e9
        stale = [tid for tid, t in self._targets.items() if (now - t['last_seen']) > self.target_timeout]
        for tid in stale:
            del self._targets[tid]

    def _publish_targets(self):
        msg = Float32MultiArray()
        moving_msg = Float32MultiArray()
        flat = []
        moving_flat = []
        for tid, t in self._targets.items():
            x, y, z = t['world_xyz']
            entry = [float(tid), float(x), float(y), float(z)]
            flat.extend(entry)
            if t.get('is_moving', False):
                moving_flat.extend(entry)
        msg.data = flat
        self.targets_pub.publish(msg)

        # Live movers only (same [id, x, y, z] format); empty when none move.
        moving_msg.data = moving_flat
        self.moving_targets_pub.publish(moving_msg)

        # Fixed targets: built from the never-pruned dict.
        fixed_msg = Float32MultiArray()
        fixed_flat = []
        for tid, world_xyz in self._fixed_targets.items():
            x, y, z = world_xyz
            fixed_flat.extend([float(tid), float(x), float(y), float(z)])
        fixed_msg.data = fixed_flat
        self.fixed_targets_pub.publish(fixed_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SemanticMapNode()
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
