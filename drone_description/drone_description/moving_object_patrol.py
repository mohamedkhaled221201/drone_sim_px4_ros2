import math
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class MovingObjectPatrol(Node):
    def __init__(self):
        super().__init__('moving_object_patrol')

        # Which Gazebo model this instance drives — lets the SAME node
        # script control any of the moving_box_N models in
        # dynamic_world.sdf; spawn_drone_dynamic_launch.py launches one
        # instance of this node per box, each with its own model_name (and
        # its own ROS node `name=` so they don't clash), speed, and
        # direction. cmd_vel/scan topic names are built from this rather
        # than hardcoded, so adding another box is a launch-file change,
        # not a code change.
        self.declare_parameter('model_name', 'moving_box_1')
        self.model_name = self.get_parameter('model_name').value

        self.pub = self.create_publisher(Twist, f'/model/{self.model_name}/cmd_vel', 10)
        self.create_subscription(
            LaserScan, f'/model/{self.model_name}/scan', self._scan_cb, 10
        )

        # Constant forward speed, single direction — keep this comfortably
        # BELOW the drone's max_speed (1.0 m/s in flight_controller.py) so
        # the drone can catch up and keep pace; 0.5 m/s (2x margin) is a
        # good default.
        self.declare_parameter('speed', 0.5)        # m/s, forward (+/-x)
        self.declare_parameter('direction', 1.0)    # +1.0 = +x, -1.0 = -x

        # Obstacle avoidance tuning.
        self.declare_parameter('obstacle_distance', 2.5)   # m, start avoiding
        self.declare_parameter('clear_distance', 3.5)      # m, stop avoiding
        self.declare_parameter('avoid_speed', 0.4)         # m/s sideways
        self.declare_parameter('front_cone_deg', 25.0)     # +/- deg counted as "ahead"
        self.declare_parameter('slow_factor', 0.5)         # forward speed multiplier while avoiding

        # GRAVITY-DROP STARTUP DELAY: the world now spawns this box 3m
        # above the ground and lets gravity drop it, instead of spawning
        # it pre-placed with gravity off (see baylands_dynamic.sdf's
        # SPAWN/GRAVITY comment on the moving_box models). VelocityControl
        # fully overrides whichever velocity axes it's given, every
        # physics step, from the moment it gets its first Twist -- and
        # _publish() below always sends linear.z = 0.0. Publishing
        # immediately on node start would pin the box's height at spawn
        # before gravity ever got a chance to move it, so it would hover
        # instead of fall. Waiting here leaves gravity + the ground
        # collision as the only things acting on the box for the first
        # few seconds, long enough to fall ~3m (free-fall time is only
        # about 0.8s; the rest is settle/bounce margin) and come to rest
        # before patrol driving takes over.
        self.declare_parameter('startup_delay_sec', 3.0)
        self.startup_delay = self.get_parameter('startup_delay_sec').value
        self._start_time = time.monotonic()

        self.speed = self.get_parameter('speed').value
        self.direction = self.get_parameter('direction').value
        self.obstacle_distance = self.get_parameter('obstacle_distance').value
        self.clear_distance = self.get_parameter('clear_distance').value
        self.avoid_speed = self.get_parameter('avoid_speed').value
        self.front_cone = math.radians(self.get_parameter('front_cone_deg').value)
        self.slow_factor = self.get_parameter('slow_factor').value

        self.latest_scan = None
        self.avoiding = False
        self.avoid_dir = 0.0  # +1.0 = steer toward +y, -1.0 = steer toward -y

        self.create_timer(0.1, self._publish)
        self.get_logger().info(
            f'[{self.model_name}] Moving box started: falling under gravity for '
            f'{self.startup_delay:.1f}s before patrol begins, then constant '
            f"{self.speed:.2f} m/s, direction {'+x' if self.direction > 0 else '-x'}, "
            f'lidar-based obstacle avoidance enabled.'
        )

    def _scan_cb(self, msg):
        self.latest_scan = msg

    def _scan_summary(self, scan):
        """Return (min_range_ahead, mean_range_left, mean_range_right)."""
        range_max = scan.range_max
        angle_min = scan.angle_min
        angle_inc = scan.angle_increment

        front_vals = []
        left_vals = []
        right_vals = []

        for i, r in enumerate(scan.ranges):
            if r != r or math.isinf(r) or r <= 0.0:  # nan / inf / bogus reading
                r = range_max
            angle = angle_min + i * angle_inc

            if abs(angle) <= self.front_cone:
                front_vals.append(r)
            elif math.radians(10) <= angle <= math.radians(90):
                left_vals.append(r)
            elif -math.radians(90) <= angle <= -math.radians(10):
                right_vals.append(r)

        min_front = min(front_vals) if front_vals else range_max
        left_space = sum(left_vals) / len(left_vals) if left_vals else range_max
        right_space = sum(right_vals) / len(right_vals) if right_vals else range_max
        return min_front, left_space, right_space

    def _publish(self):
        # Don't publish ANYTHING yet -- not even a zero-velocity Twist --
        # while still inside the gravity-drop window. VelocityControl only
        # starts overriding this model's velocity once it has received at
        # least one message; skipping the publish entirely (rather than
        # sending zeros) is what actually lets it free-fall. See the
        # startup_delay_sec declaration above for the full rationale.
        if (time.monotonic() - self._start_time) < self.startup_delay:
            return

        vx = self.speed * self.direction
        vy = 0.0

        if self.latest_scan is not None:
            min_front, left_space, right_space = self._scan_summary(self.latest_scan)

            if not self.avoiding and min_front < self.obstacle_distance:
                # Obstacle just appeared ahead: commit to steering toward
                # whichever side currently has more room.
                self.avoiding = True
                self.avoid_dir = 1.0 if left_space >= right_space else -1.0
                self.get_logger().info(
                    f'[{self.model_name}] Obstacle at {min_front:.2f} m ahead, '
                    f"steering {'+y' if self.avoid_dir > 0 else '-y'} around it."
                )
            elif self.avoiding and min_front > self.clear_distance:
                # Clear again: stop the sideways nudge and go straight.
                self.avoiding = False
                self.avoid_dir = 0.0
                self.get_logger().info(f'[{self.model_name}] Path clear, resuming straight patrol.')

            if self.avoiding:
                vy = self.avoid_dir * self.avoid_speed
                vx *= self.slow_factor

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MovingObjectPatrol()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # See flight_controller.py: rclpy's default SIGINT handler may have
        # already shut the context down by the time Ctrl+C unwinds to here.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
