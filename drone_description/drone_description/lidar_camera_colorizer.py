#!/usr/bin/env python3
"""
lidar_camera_colorizer.py

Fuses the drone's gpu_lidar PointCloud2 with the RGB camera to produce
a colored point cloud: geometry (x,y,z) comes from the lidar, color
comes from projecting each lidar point into the camera image via the
camera's intrinsics (from /model/drone/camera_info) and the known
lidar->camera TF (both trace back to base_link, already published by
robot_state_publisher).

Points that fall outside the camera's field of view keep their
geometry (so occupancy/shape from the lidar is never lost) but are
colored a neutral grey, since there's no color information for them.

Output is published in the SAME frame_id as the input lidar cloud
(the Gazebo-scoped 'drone/base_footprint/lidar_sensor' frame), so the
existing lidar_frame_bridge static transform still lets octomap_server
resolve it into 'odom' exactly as before.
"""
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo
from sensor_msgs_py import point_cloud2 as pc2
try:
    from sensor_msgs_py.point_cloud2 import read_points_numpy
except ImportError:  # very old sensor_msgs_py
    read_points_numpy = None
from tf2_ros import Buffer, TransformListener
from cv_bridge import CvBridge

_NEUTRAL_RGB = (128, 128, 128)  # used for points outside camera FOV


def _quat_to_rotmat(x, y, z, w):
    """Unit quaternion -> 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _read_xyz(msg):
    """(N, 3) float32 x/y/z of a PointCloud2 with no per-point Python loop."""
    if read_points_numpy is not None:
        pts = read_points_numpy(msg, field_names=('x', 'y', 'z'))
    else:
        st = pc2.read_points(msg, field_names=('x', 'y', 'z'))
        pts = np.stack([st['x'], st['y'], st['z']], axis=-1)
    return np.asarray(pts, dtype=np.float32).reshape(-1, 3)


class LidarCameraColorizer(Node):

    def __init__(self):
        super().__init__('lidar_camera_colorizer')

        self.declare_parameter('lidar_topic', '/model/drone/scan/points')
        self.declare_parameter('image_topic', '/model/drone/camera')
        self.declare_parameter('camera_info_topic', '/model/drone/camera_info')
        self.declare_parameter('output_topic', '/model/drone/scan/points_colored')
        self.declare_parameter('sync_slop_sec', 0.1)

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(
            PointCloud2,
            self.get_parameter('output_topic').value,
            10
        )

        # ros_gz_bridge publishes ALL three of these topics as RELIABLE
        # here, not BEST_EFFORT like a typical sensor topic — confirmed
        # individually via `ros2 topic info <topic> --verbose` on the
        # lidar, image, and camera_info topics. qos_profile_sensor_data
        # (best-effort) silently fails to connect to a reliable-only
        # publisher, with no error on either side — that was the whole
        # cause of points_colored staying empty. All three subscribers
        # use this same reliable profile instead.
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        lidar_sub = message_filters.Subscriber(
            self, PointCloud2,
            self.get_parameter('lidar_topic').value,
            qos_profile=reliable_qos
        )
        image_sub = message_filters.Subscriber(
            self, Image,
            self.get_parameter('image_topic').value,
            qos_profile=reliable_qos
        )
        caminfo_sub = message_filters.Subscriber(
            self, CameraInfo,
            self.get_parameter('camera_info_topic').value,
            qos_profile=reliable_qos
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [lidar_sub, image_sub, caminfo_sub],
            queue_size=10,
            slop=self.get_parameter('sync_slop_sec').value
        )
        self.sync.registerCallback(self._callback)

        self.get_logger().info('lidar_camera_colorizer ready, waiting for synced lidar/camera pairs.')

    def _callback(self, lidar_msg: PointCloud2, image_msg: Image, caminfo_msg: CameraInfo):
        try:
            transform = self.tf_buffer.lookup_transform(
                image_msg.header.frame_id,
                lidar_msg.header.frame_id,
                lidar_msg.header.stamp,
            )
        except Exception as e:
            self.get_logger().warn(f'TF lookup failed, skipping frame: {e}', throttle_duration_sec=2.0)
            return

        try:
            cv_img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge conversion failed: {e}', throttle_duration_sec=2.0)
            return
        h, w = cv_img.shape[:2]

        fx, fy = caminfo_msg.k[0], caminfo_msg.k[4]
        cx, cy = caminfo_msg.k[2], caminfo_msg.k[5]

        # VECTORIZED (was a pure-Python loop over every point, which was
        # the slowest stage of the mapping pipeline). skip_nans only drops
        # NaN, not +Inf (a gpu_lidar reports Inf for beams that hit nothing),
        # so keep only fully finite points.
        try:
            pts = _read_xyz(lidar_msg)
        except Exception as e:
            self.get_logger().warn(f'point decode failed: {e}', throttle_duration_sec=2.0)
            return
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if pts.shape[0] == 0:
            return

        tr = transform.transform.translation
        q = transform.transform.rotation
        R = _quat_to_rotmat(q.x, q.y, q.z, q.w)
        t = np.array([tr.x, tr.y, tr.z])
        cam = pts.astype(np.float64) @ R.T + t          # lidar frame -> camera frame

        rgb = np.empty((pts.shape[0], 3), dtype=np.uint8)
        rgb[:] = _NEUTRAL_RGB
        zc = cam[:, 2]
        front = zc > 0.1
        u = np.full(zc.shape, -1, dtype=np.int64)
        v = np.full(zc.shape, -1, dtype=np.int64)
        u[front] = (fx * cam[front, 0] / zc[front] + cx).astype(np.int64)
        v[front] = (fy * cam[front, 1] / zc[front] + cy).astype(np.int64)
        ok = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rgb[ok] = cv_img[v[ok], u[ok]]

        packed = ((rgb[:, 0].astype(np.uint32) << 16)
                  | (rgb[:, 1].astype(np.uint32) << 8)
                  | rgb[:, 2].astype(np.uint32))

        # x, y, z, rgb (float32 bit-pattern of 0x00RRGGBB): 16 B/point.
        buf = np.empty((pts.shape[0], 4), dtype=np.float32)
        buf[:, :3] = pts
        buf[:, 3] = packed.view(np.float32)

        out_msg = PointCloud2()
        out_msg.header = lidar_msg.header
        out_msg.height = 1
        out_msg.width = int(pts.shape[0])
        out_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        out_msg.is_bigendian = False
        out_msg.point_step = 16
        out_msg.row_step = 16 * out_msg.width
        out_msg.is_dense = True
        out_msg.data = buf.tobytes()
        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraColorizer()
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
