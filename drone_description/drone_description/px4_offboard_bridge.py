#!/usr/bin/env python3
"""
px4_offboard_bridge.py

Replaces flight_controller.py + state_estimator.py + battery_sim.py's roles
with a real PX4 autopilot (SITL now, real Pixhawk hardware later with no
code changes -- uXRCE-DDS talks to a companion-computer client the same way
whether that client lives inside PX4 SITL or on a physical Pixhawk).

WHAT THIS NODE DOES, split into the same three jobs the old stack split
across three files:

1. COMMAND ARBITRATION (flight_controller.py's job): the same three-mode
   arbitration keyboard_teleop.py's docstring describes -- 'manual' listens
   to /drone/teleop_cmd, 'auto' to /drone/tracker_cmd (object_tracker.py),
   'mission' to /drone/mission_cmd (mission_manager.py) -- selected by
   /drone/control_mode, each falling back to a zero command if its source
   goes quiet for its *_timeout_sec. What's NEW is what happens to the
   selected Twist afterward: instead of feeding an LQR/MPC inner loop that
   computes tilt angles and mixes them into four motor speeds, it's
   converted to a PX4 TrajectorySetpoint velocity command and handed to
   PX4's own position/velocity/attitude/rate controllers and motor mixer --
   the same cascade a real Pixhawk runs in the air.

2. STATE ESTIMATE (state_estimator.py's job): PX4's own EKF2 fuses IMU +
   GPS (both simulated by PX4 SITL's own Gazebo bridge directly, not via
   ROS) and publishes the fused estimate as px4_msgs/VehicleOdometry on
   /fmu/out/vehicle_odometry. This node converts that PX4-native estimate
   (position/velocity in NED, attitude FRD-to-NED) into the same
   nav_msgs/Odometry shape, ENU/FLU convention, that used to arrive on
   /model/drone/odometry from Gazebo ground truth -- and republishes it on
   that SAME topic name, so odom_tf_broadcaster.py, mission_manager.py,
   drone_gui.py, semantic_map_node.py, and octomap's TF chain all keep
   working unmodified. bridge.yaml's old GZ_TO_ROS entry for
   /model/drone/odometry (ground truth) must be removed so this node is the
   topic's only publisher -- two publishers on one topic just race.

3. BATTERY: PX4 SITL's own battery is static (no load model), so this node
   runs a physics-based 4S pack model (SimBattery below) driven by the real
   thrust demand and publishes sensor_msgs/BatteryState on
   /drone/battery_state at 10 Hz. drone_gui.py subscribes to that topic.
   NOTE: PX4 itself never sees this pack, so its low-battery failsafes do
   not fire from it.

WHAT THIS NODE DELIBERATELY DOES NOT DO, because PX4 already does it
natively and it's the more realistic (and more robust) place for it:
  - Geofencing: configure PX4's own geofence instead (GF_ACTION,
    GF_MAX_HOR_DIST, GF_MAX_VER_DIST, or a polygon fence uploaded via
    QGroundControl/MAVSDK) rather than clamping velocity here. PX4 will
    RTL/hold/land per GF_ACTION when it's breached -- more robust than a
    soft push-back term in a companion-computer node, and it's what a real
    Pixhawk deployment would actually use.
  - Lidar-based obstacle avoidance: PX4's Collision Prevention module
    (CP_DIST and friends) consumes px4_msgs/ObstacleDistance on
    /fmu/in/obstacle_distance and shrinks velocity setpoints itself. The
    natural next piece is a small lidar_to_px4_obstacle_distance.py node
    that bins /model/drone/scan/points (or /map/points) into that message
    -- deliberately not folded into this file, since it's a separate,
    self-contained job and this one is already the load-bearing node.

FRAME CONVENTIONS (get this wrong and *everything* flies sideways):
  - Incoming Twist commands (teleop/tracker/mission) are BODY-frame, FLU
    (ROS convention: x=forward, y=left, z=up) -- see keyboard_teleop.py's
    docstring. PX4 wants body-frame FRD (x=forward, y=right, z=down) for
    the yaw-rotation step below, and its TrajectorySetpoint.velocity field
    is in the LOCAL NED WORLD frame (x=north, y=east, z=down), not body.
  - PX4's VehicleOdometry position/velocity are NED world frame; its
    attitude quaternion `q` rotates FRD body -> NED world. Both are
    converted to ENU world / FLU body for the republished Odometry, using
    the same style of quat/euler helpers flight_controller.py and
    state_estimator.py already used elsewhere in this stack.

VERSION NOTE: px4_msgs field names (BatteryStatus especially) have
shifted across PX4 releases -- this was written against the field names
current as of PX4 v1.15/v1.16's uXRCE-DDS topic set. If colcon build
complains about an unknown field, check that against whichever px4_msgs
branch matches your PX4 firmware version and adjust the one or two lines
that reference it; the arbitration/frame-conversion logic around it
doesn't change.
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
    SensorCombined,
    VehicleLandDetected,
)


# PX4 publishes/expects best-effort, volatile, shallow-history QoS on every
# /fmu/in and /fmu/out topic -- this is NOT the rclpy default (reliable,
# keep-last-10), and a mismatched QoS fails to connect *silently*, the same
# trap lidar_camera_colorizer.py's own QOS comment already hit once on the
# camera/lidar bridge topics. Every px4_msgs pub/sub below uses this.
PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


# ---- realistic battery model (see class docstring) ----
class SimBattery:
    """Physics-based 4S LiPo model driven by thrust demand.

    Constant-power load through a pack with internal resistance:
        Voc(soc) = cells * ocv(soc)
        R*I^2 - Voc*I + P = 0   ->   I = (Voc - sqrt(Voc^2 - 4*R*P)) / (2*R)
        V = Voc - I*R
    so current RISES as the pack empties (same power, lower voltage) and
    voltage sags under load, like a real pack. Charge is coulomb-counted.
    """
    # Approximate resting LiPo cell voltage vs state of charge.
    _OCV = [(0.00, 3.27), (0.05, 3.61), (0.10, 3.69), (0.20, 3.73),
            (0.30, 3.77), (0.40, 3.80), (0.50, 3.84), (0.60, 3.87),
            (0.70, 3.92), (0.80, 3.98), (0.90, 4.07), (1.00, 4.20)]

    def __init__(self, cells=4, capacity_ah=10.0, r_ohm=0.020,
                 hover_power_w=320.0, avionics_w=8.0, idle_w=6.0,
                 initial_soc=1.0, drain_accel=1.0, thrust_tau_s=0.3):
        self.cells = cells
        self.capacity_ah = capacity_ah
        self.r = r_ohm
        self.hover_w = hover_power_w
        self.avionics_w = avionics_w
        self.idle_w = idle_w
        self.accel = drain_accel
        self.tau = thrust_tau_s
        self.soc = max(0.0, min(1.0, initial_soc))
        self._ratio_f = 0.0
        self.voltage = cells * self.ocv(self.soc)
        self.current = 0.0

    def ocv(self, soc):
        soc = max(0.0, min(1.0, soc))
        for (s0, v0), (s1, v1) in zip(self._OCV, self._OCV[1:]):
            if soc <= s1:
                return v0 + (v1 - v0) * (soc - s0) / (s1 - s0)
        return self._OCV[-1][1]

    def step(self, dt, armed, flying, thrust_ratio):
        """dt seconds; thrust_ratio = |specific force| / g (1.0 = hover)."""
        if dt <= 0.0:
            return
        target = thrust_ratio if (armed and flying) else 0.0
        self._ratio_f += (dt / (self.tau + dt)) * (target - self._ratio_f)
        if armed and flying:
            p = self.hover_w * max(self._ratio_f, 0.0) ** 1.5
        elif armed:
            p = self.idle_w
        else:
            p = 0.0
        p += self.avionics_w

        voc = self.cells * self.ocv(self.soc)
        disc = voc * voc - 4.0 * self.r * p
        if disc < 0.0:                      # pack cannot deliver P: brown-out
            self.current = voc / (2.0 * self.r)
            self.voltage = voc / 2.0
        else:
            self.current = (voc - disc ** 0.5) / (2.0 * self.r)
            self.voltage = voc - self.current * self.r
        self.soc = max(0.0, self.soc - self.current * dt * self.accel / 3600.0 / self.capacity_ah)



def quat_to_yaw(w, x, y, z):
    """Yaw (rotation about the vertical axis) from a quaternion, aerospace
    convention -- the same formula works whether the quaternion is
    FLU/ENU or FRD/NED, since yaw-about-vertical drops out identically in
    both; only which vertical axis and which 'zero heading' it's measured
    from differs, which the call site below accounts for separately."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def euler_flu_enu_to_quat(roll, pitch, yaw):
    """roll/pitch/yaw (FLU body, ENU world, standard ROS/REP-103 sense) ->
    quaternion (x, y, z, w), for republishing PX4's attitude in the
    convention everything downstream (RViz, drone_gui.py) already expects
    from the old ground-truth /model/drone/odometry topic."""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class Px4OffboardBridge(Node):

    def __init__(self):
        super().__init__('px4_offboard_bridge')

        # --- mode arbitration params, same names/values flight_controller.py used ---
        self.declare_parameter('teleop_timeout_sec', 0.5)
        self.declare_parameter('tracker_timeout_sec', 0.5)
        self.declare_parameter('mission_timeout_sec', 0.5)

        self.teleop_timeout = self.get_parameter('teleop_timeout_sec').value
        self.tracker_timeout = self.get_parameter('tracker_timeout_sec').value
        self.mission_timeout = self.get_parameter('mission_timeout_sec').value

        # --- SPEED SCALING ---
        # keyboard_teleop and object_tracker both publish NORMALIZED Twists
        # (each axis roughly -1..+1), and this node used to pass them to PX4
        # unchanged, i.e. 1.0 == 1.0 m/s. That, not the physics, was the
        # drone's top speed. These give the full-stick speed in m/s per
        # source. Manual defaults are raised; auto (tracker) stays at 1.0 so
        # its tuned gains behave as before -- raise it deliberately, since
        # the tracker's loops were tuned against ~1 m/s of authority.
        # PX4 still enforces its own limits on top (MPC_XY_VEL_MAX,
        # MPC_Z_VEL_MAX_UP/DN, MPC_TILTMAX_AIR, MPC_ACC_HOR_MAX): a value
        # here above those is silently clipped by PX4.
        self.declare_parameter('manual_max_speed_xy_mps', 5.0)
        self.declare_parameter('manual_max_speed_z_mps', 2.0)
        # Was 1.0 m/s: closing a 60 m gap took about a minute. Raised so
        # AUTO/tracking reaches standoff quickly; PX4's MPC_XY_VEL_MAX still caps it.
        self.declare_parameter('auto_max_speed_xy_mps', 3.0)
        self.declare_parameter('auto_max_speed_z_mps', 1.5)
        self.manual_speed_xy = self.get_parameter('manual_max_speed_xy_mps').value
        self.manual_speed_z = self.get_parameter('manual_max_speed_z_mps').value
        self.auto_speed_xy = self.get_parameter('auto_max_speed_xy_mps').value
        self.auto_speed_z = self.get_parameter('auto_max_speed_z_mps').value

        # --- SMOOTH TILT (auto/mission only) ---
        # PX4 turns every velocity-setpoint change into a tilt change, so a
        # jittery or stepped setpoint shows up as sharp pitch/roll. The
        # tracker used to be the only smoother, and its limits are in
        # NORMALIZED units, which the x3 auto speed scale silently tripled
        # (0.8/s * 3 m/s = 2.4 m/s^2, ~14 deg of tilt per step). Limiting
        # here, on the final m/s command, protects every source at once:
        # a first-order low-pass (removes jitter, softens the start of a
        # change) followed by a hard acceleration cap (bounds the tilt).
        # Manual keeps its instant response.
        self.declare_parameter('auto_max_accel_xy_mps2', 1.2)   # ~7 deg of tilt at the limit
        self.declare_parameter('auto_max_accel_z_mps2', 1.0)
        self.declare_parameter('auto_cmd_filter_tau_sec', 0.35)  # 0 disables the low-pass
        self.auto_acc_xy = self.get_parameter('auto_max_accel_xy_mps2').value
        self.auto_acc_z = self.get_parameter('auto_max_accel_z_mps2').value
        self.auto_cmd_tau = self.get_parameter('auto_cmd_filter_tau_sec').value
        self._sm_cmd = [0.0, 0.0, 0.0]   # smoothed (vx, vy, vz) body FLU, m/s
        self._sm_t = None

        # --- offboard/arm handshake params ---
        self.declare_parameter('control_rate_hz', 20.0)          # PX4 requires >= 2 Hz OffboardControlMode; comfortably above it
        self.declare_parameter('setpoints_before_offboard', 15)  # ~0.75s of streaming before requesting the mode switch
        self.declare_parameter('auto_arm', True)                 # False = stream setpoints + let something external arm/switch mode
        self.declare_parameter('handshake_retry_sec', 2.0)       # re-send mode-switch/arm this often until PX4 confirms both
        self.declare_parameter('rearm_on_command', True)         # after PX4 auto-disarms (idle on ground / after landing), re-arm as soon as a non-descending command arrives

        self.control_rate = self.get_parameter('control_rate_hz').value
        self.setpoints_before_offboard = self.get_parameter('setpoints_before_offboard').value
        self.auto_arm = self.get_parameter('auto_arm').value
        self.handshake_retry = self.get_parameter('handshake_retry_sec').value
        self.rearm_on_command = self.get_parameter('rearm_on_command').value

        # --- command arbitration state (mirrors flight_controller.py's mode gating) ---
        self.mode = 'manual'
        self.teleop_cmd = Twist()
        self.tracker_cmd = Twist()
        self.mission_cmd = Twist()
        self.last_teleop_time = None
        self.last_tracker_time = None
        self.last_mission_time = None
        self._last_yaw_enu = 0.0

        self.create_subscription(String, '/drone/control_mode', self._mode_cb, 10)
        self.create_subscription(Twist, '/drone/teleop_cmd', self._teleop_cb, 10)
        self.create_subscription(Twist, '/drone/tracker_cmd', self._tracker_cb, 10)
        self.create_subscription(Twist, '/drone/mission_cmd', self._mission_cb, 10)

        # --- PX4 uXRCE-DDS I/O ---
        self.offboard_mode_pub = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', PX4_QOS)
        self.setpoint_pub = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', PX4_QOS)
        self.command_pub = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', PX4_QOS)

        self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self._px4_odom_cb, PX4_QOS)
        self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status_v4', self._px4_status_cb, PX4_QOS)

        # --- BATTERY: PX4's SITL battery does not model load (it reports a
        # fixed -1 A and a flat voltage), so the pack is simulated HERE from
        # the drone's real thrust demand. Thrust comes from the accelerometer
        # (|specific force| / g == 1.0 in hover), so it needs no extra DDS
        # topics. Publishes sensor_msgs/BatteryState on /drone/battery_state.
        self.declare_parameter('battery_cells', 4)
        self.declare_parameter('battery_capacity_ah', 10.0)
        self.declare_parameter('battery_internal_resistance_ohm', 0.020)  # whole pack
        self.declare_parameter('battery_hover_power_w', 320.0)   # electrical power at hover (~20 A at 16 V)
        self.declare_parameter('battery_avionics_power_w', 8.0)
        self.declare_parameter('battery_idle_power_w', 6.0)      # armed, on the ground
        self.declare_parameter('battery_initial_soc', 1.0)       # 0..1; lower it to test low-battery behaviour
        self.declare_parameter('battery_drain_accel', 1.0)       # >1 drains faster than real time (testing only)
        self.battery = SimBattery(
            cells=self.get_parameter('battery_cells').value,
            capacity_ah=self.get_parameter('battery_capacity_ah').value,
            r_ohm=self.get_parameter('battery_internal_resistance_ohm').value,
            hover_power_w=self.get_parameter('battery_hover_power_w').value,
            avionics_w=self.get_parameter('battery_avionics_power_w').value,
            idle_w=self.get_parameter('battery_idle_power_w').value,
            initial_soc=self.get_parameter('battery_initial_soc').value,
            drain_accel=self.get_parameter('battery_drain_accel').value,
        )
        self._bat_last_t = None
        self._landed = True
        self.create_subscription(SensorCombined, '/fmu/out/sensor_combined', self._px4_sensor_cb, PX4_QOS)
        self.create_subscription(VehicleLandDetected, '/fmu/out/vehicle_land_detected', self._px4_land_detected_cb, PX4_QOS)
        self.create_timer(0.1, self._publish_battery)

        # --- republished, ROS-native topics -- SAME names/types the old
        # ground-truth/state_estimator/battery_sim publishers used, so
        # every downstream node needs zero code changes. bridge.yaml's
        # ground-truth /model/drone/odometry GZ_TO_ROS entry must be
        # removed so this is the topic's only publisher.
        self.odom_pub = self.create_publisher(Odometry, '/model/drone/odometry', 10)
        self.battery_pub = self.create_publisher(BatteryState, '/drone/battery_state', 10)

        self.vehicle_status = None
        self._warned_frame = False
        self.offboard_setpoint_counter = 0
        self._handshake_done = False
        self._last_handshake_time = -1e9

        self.create_timer(1.0 / self.control_rate, self._control_tick)

        self.get_logger().info(
            'px4_offboard_bridge ready: arbitrating teleop/tracker/mission '
            f'Twist commands into PX4 offboard velocity setpoints at {self.control_rate:.0f} Hz. '
            f'auto_arm={self.auto_arm}.'
        )

    # ---- command inputs (unchanged contract from flight_controller.py) ----

    def _mode_cb(self, msg):
        self.mode = msg.data

    def _teleop_cb(self, msg):
        self.teleop_cmd = msg
        self.last_teleop_time = self._now()

    def _tracker_cb(self, msg):
        self.tracker_cmd = msg
        self.last_tracker_time = self._now()

    def _mission_cb(self, msg):
        self.mission_cmd = msg
        self.last_mission_time = self._now()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _active_twist(self):
        """Same fallback-to-zero-on-timeout pattern used throughout this
        stack (see moving_object_patrol.py's scan staleness handling): a
        mode whose command source has gone quiet commands nothing, rather
        than replaying a stale Twist forever."""
        now = self._now()
        if self.mode == 'auto':
            if self.last_tracker_time is not None and (now - self.last_tracker_time) <= self.tracker_timeout:
                return self.tracker_cmd
            return Twist()
        if self.mode == 'mission':
            if self.last_mission_time is not None and (now - self.last_mission_time) <= self.mission_timeout:
                return self.mission_cmd
            return Twist()
        # manual (default/fallback for any unrecognized mode string)
        if self.last_teleop_time is not None and (now - self.last_teleop_time) <= self.teleop_timeout:
            return self.teleop_cmd
        return Twist()

    # ---- PX4 telemetry inputs ----

    def _px4_status_cb(self, msg):
        self.vehicle_status = msg

    def _px4_odom_cb(self, msg: VehicleOdometry):
        # The conversion below assumes NED world frame. Refuse to publish
        # (and say so once) if PX4 ever reports anything else, instead of
        # silently flying sideways on a wrongly-rotated pose.
        if msg.pose_frame != VehicleOdometry.POSE_FRAME_NED:
            if not self._warned_frame:
                self.get_logger().error(
                    f'VehicleOdometry.pose_frame={msg.pose_frame}, expected NED '
                    f'({VehicleOdometry.POSE_FRAME_NED}); odometry NOT republished.'
                )
                self._warned_frame = True
            return
        self._publish_ros_odometry(msg)

    def _px4_land_detected_cb(self, msg):
        self._landed = bool(msg.landed)

    def _px4_sensor_cb(self, msg):
        # Integrate in PX4 (sim) time so the drain is correct at any
        # real-time factor. Skip the first sample and any large gap.
        t = msg.timestamp * 1e-6
        last, self._bat_last_t = self._bat_last_t, t
        if last is None:
            return
        dt = t - last
        if dt <= 0.0 or dt > 0.5:
            return
        a = msg.accelerometer_m_s2
        ratio = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) / 9.80665
        st = self.vehicle_status
        armed = st is not None and st.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.battery.step(dt, armed, not self._landed, ratio)

    def _publish_battery(self):
        bat = self.battery
        out = BatteryState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.voltage = float(bat.voltage)
        out.current = float(-bat.current)  # BatteryState convention: negative = discharging
        out.capacity = float(bat.capacity_ah)
        out.design_capacity = float(bat.capacity_ah)
        out.charge = float(bat.soc * bat.capacity_ah)
        out.percentage = float(bat.soc)
        out.cell_voltage = [float(bat.voltage / bat.cells)] * bat.cells
        out.present = True
        out.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIPO
        out.power_supply_status = (
            BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            if bat.current > 1.0 else BatteryState.POWER_SUPPLY_STATUS_UNKNOWN
        )
        out.power_supply_health = (
            BatteryState.POWER_SUPPLY_HEALTH_DEAD
            if bat.soc <= 0.0 else BatteryState.POWER_SUPPLY_HEALTH_GOOD
        )
        self.battery_pub.publish(out)

    def _publish_ros_odometry(self, px4: VehicleOdometry):
        # NED (north, east, down) world -> ENU (east, north, up) world:
        # swap x/y, negate z. Same swap for velocity.
        x_enu, y_enu, z_enu = px4.position[1], px4.position[0], -px4.position[2]
        vx_enu, vy_enu, vz_enu = px4.velocity[1], px4.velocity[0], -px4.velocity[2]

        # px4.q rotates FRD body -> NED world, [w, x, y, z]. Pull yaw/pitch/
        # roll out of it with the same Euler extraction flight_controller.py
        # already used (quat_to_euler), then convert NED/FRD angles to
        # ENU/FLU: yaw_enu = pi/2 - yaw_ned (NED yaw is 0=north/+cw-toward-
        # east; ENU yaw is 0=east/+ccw-toward-north), roll unchanged,
        # pitch negated (FRD pitch-up is negative; FLU pitch-up is positive).
        w, x, y, z = px4.q[0], px4.q[1], px4.q[2], px4.q[3]
        yaw_ned = quat_to_yaw(w, x, y, z)
        sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
        pitch_ned = math.asin(sinp)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll_ned = math.atan2(sinr_cosp, cosr_cosp)

        yaw_enu = (math.pi / 2.0) - yaw_ned
        roll_enu, pitch_enu = roll_ned, -pitch_ned
        qx, qy, qz, qw = euler_flu_enu_to_quat(roll_enu, pitch_enu, yaw_enu)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'
        msg.pose.pose.position.x = float(x_enu)
        msg.pose.pose.position.y = float(y_enu)
        msg.pose.pose.position.z = float(z_enu)
        msg.pose.pose.orientation.x = float(qx)
        msg.pose.pose.orientation.y = float(qy)
        msg.pose.pose.orientation.z = float(qz)
        msg.pose.pose.orientation.w = float(qw)
        # twist is body-frame in this stack's existing convention (see
        # state_estimator.py's FRAME BUGFIX note) -- rotate the ENU world
        # velocity into FLU body using the yaw just derived (roll/pitch's
        # contribution to horizontal velocity ignored, same simplification
        # state_estimator.py made).
        cy, sy = math.cos(yaw_enu), math.sin(yaw_enu)
        vx_body = vx_enu * cy + vy_enu * sy
        vy_body = -vx_enu * sy + vy_enu * cy
        msg.twist.twist.linear.x = float(vx_body)
        msg.twist.twist.linear.y = float(vy_body)
        msg.twist.twist.linear.z = float(vz_enu)
        self.odom_pub.publish(msg)

        self._last_yaw_enu = yaw_enu  # cached for the control tick's body->NED rotation below

    # ---- control tick: arbitrate -> convert -> stream to PX4 ----

    def _control_tick(self):
        # Keep streaming OffboardControlMode every tick regardless of
        # whether PX4 has been heard from yet -- it needs to already be
        # receiving this before it will accept the mode switch below.
        ob = OffboardControlMode()
        ob.position = False
        ob.velocity = True
        ob.acceleration = False
        ob.attitude = False
        ob.body_rate = False
        ob.thrust_and_torque = False
        ob.direct_actuator = False
        ob.timestamp = self._px4_time()
        self.offboard_mode_pub.publish(ob)

        cmd = self._active_twist()

        # Normalized command (clamped to -1..1) -> m/s for the active source.
        if self.mode == 'auto':
            k_xy, k_z = self.auto_speed_xy, self.auto_speed_z
        else:
            k_xy, k_z = self.manual_speed_xy, self.manual_speed_z
        clamp = lambda v: max(-1.0, min(1.0, v))
        cx, cy_, cz = clamp(cmd.linear.x) * k_xy, clamp(cmd.linear.y) * k_xy, clamp(cmd.linear.z) * k_z

        # Smooth + acceleration-limit auto/mission commands (see SMOOTH TILT).
        now_t = self._now()
        if self.mode in ('auto', 'mission'):
            dt = 1.0 / self.control_rate if self._sm_t is None else max(1e-3, min(0.2, now_t - self._sm_t))
            tgt = [cx, cy_, cz]
            alpha = 1.0 if self.auto_cmd_tau <= 0.0 else 1.0 - math.exp(-dt / self.auto_cmd_tau)
            out = []
            for i, (t_i, lim) in enumerate(zip(tgt, (self.auto_acc_xy, self.auto_acc_xy, self.auto_acc_z))):
                prev = self._sm_cmd[i]
                want = prev + (t_i - prev) * alpha
                step = lim * dt
                out.append(prev + max(-step, min(step, want - prev)))
            self._sm_cmd = out
            cx, cy_, cz = out
        else:
            # Track the raw command in manual so switching into auto starts
            # from the current velocity request instead of from zero/stale.
            self._sm_cmd = [cx, cy_, cz]
        self._sm_t = now_t

        # Body FLU -> body FRD: forward unchanged, right = -left, down = -up.
        vx_frd = cx
        vy_frd = -cy_
        vz_frd = -cz

        # Body FRD -> world NED, using the last yaw PX4 reported (converted
        # back to NED sense, since that's the rotation PX4's own convention
        # expects here).
        yaw_ned = (math.pi / 2.0) - self._last_yaw_enu
        cy, sy = math.cos(yaw_ned), math.sin(yaw_ned)
        vx_ned = vx_frd * cy - vy_frd * sy
        vy_ned = vx_frd * sy + vy_frd * cy
        vz_ned = vz_frd

        sp = TrajectorySetpoint()
        sp.position = [float('nan'), float('nan'), float('nan')]  # velocity control -- position ignored
        sp.velocity = [float(vx_ned), float(vy_ned), float(vz_ned)]
        sp.acceleration = [float('nan'), float('nan'), float('nan')]
        sp.yaw = float('nan')
        # ROS angular.z is +ccw about world-up (ENU z); PX4 yawspeed is
        # about world-down (NED z), which flips the sign.
        sp.yawspeed = float(-cmd.angular.z)
        sp.timestamp = self._px4_time()
        self.setpoint_pub.publish(sp)

        self._offboard_arm_handshake(cmd)

    @staticmethod
    def _cmd_is_active(cmd):
        return (abs(cmd.linear.x) > 1e-3 or abs(cmd.linear.y) > 1e-3
                or abs(cmd.linear.z) > 1e-3 or abs(cmd.angular.z) > 1e-3)

    def _maybe_rearm(self, cmd):
        """PX4 auto-disarms an armed-but-idle vehicle after COM_DISARM_PRE
        (10 s default) and after landing (COM_DISARM_LAND). The one-shot
        handshake never re-arms, so afterwards teleop commands did nothing.
        Re-arm here when a real, non-descending command arrives. A descend
        command never re-arms, so holding 'down' on the ground stays safe."""
        if not (self.auto_arm and self.rearm_on_command):
            return
        st = self.vehicle_status
        if st is None or st.arming_state == VehicleStatus.ARMING_STATE_ARMED:
            return
        if not self._cmd_is_active(cmd) or cmd.linear.z < 0.0:
            return
        now = self._now()
        if (now - self._last_handshake_time) < self.handshake_retry:
            return
        self._last_handshake_time = now
        if st.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(
            f'Vehicle is disarmed but a command arrived; re-arming (nav_state={st.nav_state}, '
            f'preflight_ok={st.pre_flight_checks_pass}).')

    def _offboard_arm_handshake(self, cmd):
        """Switch to OFFBOARD and arm, retrying until PX4 CONFIRMS both.

        The old version fired each command exactly once, at tick 15
        (~0.75 s after this node started). In the launch file PX4 SITL is
        delayed 12 s, so that single burst went to nobody and was never
        retried -- the drone would sit disarmed forever. Now: nothing is
        counted or sent until PX4 has actually published VehicleStatus
        (proof the uXRCE-DDS link + PX4 are alive), setpoints must have
        streamed for setpoints_before_offboard ticks after that, and the
        commands are then re-sent every handshake_retry_sec until
        vehicle_status reports nav_state == OFFBOARD and (if auto_arm)
        arming_state == ARMED. After that it stops, so a deliberate
        disarm or a PX4 failsafe is NOT silently overridden.
        """
        st = self.vehicle_status
        if st is None:
            return
        if self._handshake_done:
            self._maybe_rearm(cmd)
            return
        if self.offboard_setpoint_counter < self.setpoints_before_offboard:
            self.offboard_setpoint_counter += 1
            return

        in_offboard = st.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        armed = st.arming_state == VehicleStatus.ARMING_STATE_ARMED
        if in_offboard and (armed or not self.auto_arm):
            self._handshake_done = True
            self.get_logger().info('PX4 confirmed OFFBOARD' + (' + ARMED.' if self.auto_arm else '.'))
            return

        now = self._now()
        if (now - self._last_handshake_time) < self.handshake_retry:
            return
        self._last_handshake_time = now
        if not in_offboard:
            self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        if self.auto_arm and not armed:
            self._send_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info(
            f'Requesting OFFBOARD/arm (nav_state={st.nav_state}, arming_state={st.arming_state}, '
            f'preflight_ok={st.pre_flight_checks_pass}); retrying every {self.handshake_retry:.0f}s '
            'until confirmed.', throttle_duration_sec=self.handshake_retry)

    def _send_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = self._px4_time()
        self.command_pub.publish(msg)

    def _px4_time(self):
        return int(self.get_clock().now().nanoseconds / 1000)


def main(args=None):
    rclpy.init(args=args)
    node = Px4OffboardBridge()
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
