#!/usr/bin/env python3
"""
keyboard_teleop.py

Terminal-based WASD teleop for the quad-X drone. Reads raw keystrokes
from THIS terminal (it must have focus — not Gazebo).

CAVEAT: a terminal only tells us a key was *pressed* (and, while held,
keeps re-sending it via OS key-repeat) — it never tells us a key was
*released*. So "release to stop" is approximated via key_timeout_sec.

MULTI-KEY HANDLING: terminals commonly stop repeating one key the
moment a second key starts repeating, even if both are still physically
held — this is exactly what key_timeout_sec is already for: a key stays
"active" for key_timeout_sec after its last actually-seen repeat, so a
key that's still held but has temporarily stopped re-firing (because
the terminal is busy repeating a different key) doesn't drop out.

DIAGONAL-OSCILLATION BUGFIX: this used to instead TIME-MULTIPLEX whenever
2+ keys were active — each active key got full, exclusive control for
cycle_sec (0.25s), then control round-robin'd to the next one, so
holding w+a alternated full-forward / full-left every 0.25s rather than
blending them into one steady diagonal. That was a workaround for the
same key-repeat-dropout problem key_timeout_sec already solves, and it
came at a real cost: it commanded a literal square wave between two
directions during every diagonal move, which the flight controller
dutifully tracked — showing up as periodic tilting/vibrating exactly
when changing or holding a diagonal direction. Since key_timeout_sec
already keeps a temporarily-silent-but-still-held key marked active,
the round-robin was solving a problem that no longer existed once that
timeout was in place. Summing every currently-active key's contribution
per axis (below) gives a single steady diagonal command instead.

KEYS:
  w         forward     (+x)
  x         backward    (-x)
  a         left        (+y)
  d         right       (-y)
  space     up          (+z)
  z         down        (-z)
  q         yaw left    (+angular.z, ccw)
  e         yaw right   (-angular.z, cw)
  t         toggle autonomous object-tracking mode ('auto') on/off
  m         toggle autonomous mission mode ('mission') on/off
  g         toggle mission type ('waypoints' <-> 'explore'), forwarded
            to mission_manager regardless of current control mode

Roll and pitch aren't commanded directly here — the flight controller's
outer velocity loop generates those automatically as a side effect of
w/a/x/d (tilting to accelerate). q/e command yaw rate directly via
angular.z, scaled by max_yaw_rate_rps.

MODE SWITCHING: this node tracks one of three control modes —
'manual' (this node drives the drone directly via w/a/x/d/space/z/q/e),
'auto' (object_tracker.py drives it via camera-based detection), or
'mission' (mission_manager.py drives it via waypoints or autonomous
frontier exploration). The active mode is published as a std_msgs/String
on /drone/control_mode; flight_controller.py reads whichever command
topic matches it, so this node, object_tracker, and mission_manager
never fight over the same output.

't' and 'm' are each an independent on/off toggle for their own mode,
not a three-way cycle:
  - 't' flips 'auto' on if the current mode isn't already 'auto', or
    back to 'manual' if it is — regardless of whether 'mission' was
    active (pressing t always exits mission straight to auto).
  - 'm' works the same way for 'mission': flips it on from whatever
    mode is current, or back to 'manual' if mission is already active.
Since only one mode is ever active, "toggle auto" and "toggle mission"
naturally also serve as "leave whichever mode I'm in and enter this
one" / "leave this mode back to manual" — there's no way to press one
and land in a mix of both.

While NOT in 'manual', w/a/x/d/space/z/q/e here are still read and
published on /drone/teleop_cmd as always, but flight_controller.py
ignores that topic in favor of object_tracker's or mission_manager's
command topic — so flying manual keys around in 'auto'/'mission' is
harmless, it just isn't obeyed until you toggle back out.

'g' is unrelated to control mode: it only tells mission_manager which
kind of mission to fly (a fixed waypoint list vs. autonomous frontier
exploration) the next time / while 'mission' mode is active. It's sent
on /drone/mission_mode_cmd (std_msgs/String, values 'waypoints' or
'explore') whether or not 'mission' mode is currently engaged, since
mission_manager tracks mission type independently of control mode too
— so you can pre-select exploration before pressing 'm', or change your
mind about the mission type mid-flight without dropping out of it.

Publishes:
  geometry_msgs/Twist  on /drone/teleop_cmd    (manual stick command)
  std_msgs/String      on /drone/control_mode  (current mode)
  std_msgs/String      on /drone/mission_mode_cmd (mission type, on 'g' only)
"""
import atexit
import math
import sys
import termios
import tty
import threading
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# key char -> (axis, sign). 't', 'm', and 'g' are handled separately as
# mode/mission-type toggles, not axes.
_AXIS_MAP = {
    'w': ('x', +1.0),
    'x': ('x', -1.0),
    'a': ('y', +1.0),
    'd': ('y', -1.0),
    ' ': ('z', +1.0),   # space
    'z': ('z', -1.0),
    'q': ('yaw', +1.0),
    'e': ('yaw', -1.0),
}
_AUTO_TOGGLE_KEY = 't'
_MISSION_TOGGLE_KEY = 'm'
_MISSION_TYPE_TOGGLE_KEY = 'g'
_TOGGLE_DEBOUNCE_SEC = 0.5  # ignore repeat-fire while any toggle key is held

_MISSION_TYPES = ('waypoints', 'explore')


class KeyboardTeleop(Node):

    def __init__(self):
        super().__init__('keyboard_teleop')

        self.declare_parameter('teleop_rate_hz', 20.0)
        self.declare_parameter('key_timeout_sec', 0.4)
        self.declare_parameter('max_yaw_rate_rps', 1.5)  # rad/s sent on q/e
        self.rate = self.get_parameter('teleop_rate_hz').value
        self.key_timeout = self.get_parameter('key_timeout_sec').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate_rps').value

        self.pub = self.create_publisher(Twist, '/drone/teleop_cmd', 10)
        self.mode_pub = self.create_publisher(String, '/drone/control_mode', 10)
        self.mission_type_pub = self.create_publisher(String, '/drone/mission_mode_cmd', 10)

        self._lock = threading.Lock()
        self._last_seen = {}   # key char -> monotonic time last seen
        self._stop_reader = False

        # One of 'manual' | 'auto' | 'mission'. See MODE SWITCHING above
        # for how 't' and 'm' each independently toggle their own mode.
        self.mode = 'manual'
        # Which mission mission_manager should fly once/while 'mission'
        # is active. Tracked here purely so 'g' can toggle it and so the
        # log line below can show the current selection; mission_manager
        # is the actual source of truth once it's running.
        self.mission_type = _MISSION_TYPES[0]

        # Separate debounce timestamps per toggle key so holding one down
        # (OS key-repeat) can't also swallow a nearly-simultaneous press
        # of a different toggle key.
        self._last_toggle_time = {
            _AUTO_TOGGLE_KEY: 0.0,
            _MISSION_TOGGLE_KEY: 0.0,
            _MISSION_TYPE_TOGGLE_KEY: 0.0,
        }

        # Put the terminal into cbreak mode HERE, on the main thread, and
        # register the restore with atexit rather than relying on the
        # reader thread's own finally block. The reader thread blocks
        # forever on sys.stdin.read(1); on Ctrl+C it never wakes up to see
        # _stop_reader, so when the process exits it's killed mid-syscall
        # and its finally block never runs — leaving the terminal stuck in
        # cbreak mode (no line editing/history) even after this program
        # quits. atexit fires on normal interpreter shutdown regardless of
        # what the background thread is doing, so it reliably restores the
        # terminal for you.
        self._term_fd = None
        self._old_term_settings = None
        self._setup_terminal()

        self._reader_thread = threading.Thread(target=self._read_keys, daemon=True)
        self._reader_thread.start()

        self.timer = self.create_timer(1.0 / self.rate, self._publish)

        self.get_logger().info(
            'Keyboard teleop ready (terminal mode, blended multi-key). '
            'THIS TERMINAL must have focus, not Gazebo. w/x=fwd/back, '
            'a/d=left/right, space/z=up/down, q/e=yaw left/right, '
            't=toggle auto-tracking, m=toggle mission mode, '
            "g=toggle mission type (currently '" + self.mission_type + "'). "
            'Holding 2+ movement keys blends them into one steady command '
            '(diagonal x/y is magnitude-normalized). '
            'Ctrl+C here to quit.'
        )

    def _setup_terminal(self):
        fd = sys.stdin.fileno()
        try:
            self._old_term_settings = termios.tcgetattr(fd)
        except termios.error:
            self.get_logger().error(
                'Could not access this terminal for raw input — run this '
                'node in an interactive terminal (not piped/redirected).'
            )
            return
        self._term_fd = fd
        tty.setcbreak(fd)
        atexit.register(self._restore_terminal)

    def _restore_terminal(self):
        if self._term_fd is not None and self._old_term_settings is not None:
            try:
                termios.tcsetattr(self._term_fd, termios.TCSADRAIN, self._old_term_settings)
            except termios.error:
                pass
            # Only restore once even if called from both destroy_node() and atexit.
            self._old_term_settings = None

    def _debounced(self, key, now):
        """True (and records the press) if `key`'s last toggle was more
        than _TOGGLE_DEBOUNCE_SEC ago — i.e. this is a fresh press, not
        OS key-repeat from a held-down toggle key."""
        if (now - self._last_toggle_time[key]) < _TOGGLE_DEBOUNCE_SEC:
            return False
        self._last_toggle_time[key] = now
        return True

    def _read_keys(self):
        if self._term_fd is None:
            # _setup_terminal already logged why.
            return
        while not self._stop_reader:
            ch = sys.stdin.read(1)
            if not ch:
                continue
            ch = ch.lower()

            if ch == _AUTO_TOGGLE_KEY:
                now = time.monotonic()
                if self._debounced(_AUTO_TOGGLE_KEY, now):
                    with self._lock:
                        self.mode = 'manual' if self.mode == 'auto' else 'auto'
                        new_mode = self.mode
                    self.get_logger().info(f'mode toggled -> {new_mode}')

            elif ch == _MISSION_TOGGLE_KEY:
                now = time.monotonic()
                if self._debounced(_MISSION_TOGGLE_KEY, now):
                    with self._lock:
                        self.mode = 'manual' if self.mode == 'mission' else 'mission'
                        new_mode = self.mode
                    self.get_logger().info(f'mode toggled -> {new_mode}')

            elif ch == _MISSION_TYPE_TOGGLE_KEY:
                now = time.monotonic()
                if self._debounced(_MISSION_TYPE_TOGGLE_KEY, now):
                    with self._lock:
                        idx = _MISSION_TYPES.index(self.mission_type)
                        self.mission_type = _MISSION_TYPES[(idx + 1) % len(_MISSION_TYPES)]
                        new_type = self.mission_type
                    self.mission_type_pub.publish(String(data=new_type))
                    self.get_logger().info(f'mission type toggled -> {new_type}')

            elif ch in _AXIS_MAP:
                with self._lock:
                    self._last_seen[ch] = time.monotonic()

    def _publish(self):
        now = time.monotonic()
        with self._lock:
            seen = dict(self._last_seen)

        active_keys = sorted(
            ch for ch, last in seen.items()
            if (now - last) <= self.key_timeout
        )

        # Sum every currently-active key's contribution per axis. See the
        # DIAGONAL-OSCILLATION BUGFIX note in the module docstring: this
        # used to round-robin one key at a time when 2+ were held, which
        # commanded a literal square wave between directions on every
        # diagonal move. key_timeout_sec already keeps a temporarily-
        # silent-but-still-held key marked "active" through an OS
        # repeat-dropout, so a plain sum works whether 0, 1, or several
        # keys are down.
        vx = vy = vz = vyaw = 0.0
        for ch in active_keys:
            axis, sign = _AXIS_MAP[ch]
            if axis == 'x':
                vx += sign
            elif axis == 'y':
                vy += sign
            elif axis == 'z':
                vz += sign
            elif axis == 'yaw':
                vyaw += sign

        # Normalize the horizontal pair so a diagonal (e.g. w+a, both
        # summing to +-1 on each axis) commands the same speed magnitude
        # as a single-axis command, instead of sqrt(2)x it.
        horiz_mag = math.hypot(vx, vy)
        if horiz_mag > 1.0:
            vx /= horiz_mag
            vy /= horiz_mag

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz
        msg.angular.z = vyaw * self.max_yaw_rate
        self.pub.publish(msg)

        with self._lock:
            current_mode = self.mode
        self.mode_pub.publish(String(data=current_mode))

    def destroy_node(self):
        self._stop_reader = True
        self._restore_terminal()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardTeleop()
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
