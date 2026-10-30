#!/usr/bin/env python3
"""
odom_tf_broadcaster.py

Bridges /model/drone/odometry -> the TF tree as odom -> base_link.

Nothing else in this stack currently publishes this transform:
robot_state_publisher only publishes the FIXED joints from
drone_urdf.xacro (base_link -> lidar_link, base_link -> camera_link,
etc.), not the drone's actual moving pose in the world. octomap_server
(and RViz, and anything else that wants to reason about the drone in a
world frame via tf2) needs this link to exist.

'odom' is used as the map root frame directly (no separate 'map'
frame) since there's no real localization/loop-closure happening here
-- odometry IS the ground truth in this sim.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped


class OdomTfBroadcaster(Node):

    def __init__(self):
        super().__init__('odom_tf_broadcaster')
        self.declare_parameter('odom_topic', '/model/drone/odometry')
        self.declare_parameter('parent_frame', 'odom')
        self.declare_parameter('child_frame', 'base_link')

        self.parent_frame = self.get_parameter('parent_frame').value
        self.child_frame = self.get_parameter('child_frame').value

        self.br = TransformBroadcaster(self)
        self.create_subscription(
            Odometry,
            self.get_parameter('odom_topic').value,
            self._odom_cb,
            10,
        )
        self.get_logger().info(
            f'Broadcasting {self.parent_frame} -> {self.child_frame} from odometry.'
        )

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = p.x
        t.transform.translation.y = p.y
        t.transform.translation.z = p.z
        t.transform.rotation = q
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = OdomTfBroadcaster()
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
