#!/usr/bin/env bash
# drone_marker_diag.sh -- why are there no markers on the GUI map?
#
# Run with Gazebo + PX4 + all ROS nodes + drone_gui already running, in a
# terminal where your workspace is sourced:
#     chmod +x drone_marker_diag.sh && ./drone_marker_diag.sh
# Point the drone camera at a target while it runs (takes ~1 minute).
#
# Overrides (env vars): SEM_NODE, TRK_NODE, WORLD, WAIT (seconds per rate check)
# A copy of the output is saved to /tmp/drone_diag_<timestamp>.txt -- paste it back.

SEM_NODE=${SEM_NODE:-/semantic_map_node}
TRK_NODE=${TRK_NODE:-/object_tracker}
WORLD=${WORLD:-baylands}
WAIT=${WAIT:-4}
REPORT=/tmp/drone_diag_$(date +%Y%m%d_%H%M%S).txt
exec > >(tee "$REPORT") 2>&1

FAILS=()
pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAILS+=("$*"); }
warn() { echo "  [WARN] $*"; }
info() { echo "         $*"; }
hdr()  { echo; echo "=== $* ==="; }

command -v ros2 >/dev/null || { echo "ros2 not found -- source your ROS 2 + workspace setup.bash first"; exit 1; }
echo "ROS_DISTRO=${ROS_DISTRO:-?}  ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}  $(date)"

# ---------------------------------------------------------------- helpers
cap() { local s=$1 f=$2; shift 2; timeout -s INT "$s" "$@" >"$f" 2>&1; }

pubs() { ros2 topic info "$1" 2>/dev/null | awk '/Publisher count/{print $3}'; }
subs() { ros2 topic info "$1" 2>/dev/null | awk '/Subscription count/{print $3}'; }

topic_rate() {   # topic [seconds]
  local f; f=$(mktemp)
  cap "${2:-$WAIT}" "$f" ros2 topic hz "$1"
  grep -o "average rate: [0-9.]*" "$f" | tail -1 | awk '{print $3}'
  rm -f "$f"
}

check_topic() {  # topic label [seconds]
  local t=$1 label=$2 p s r
  p=$(pubs "$t"); s=$(subs "$t"); p=${p:-0}; s=${s:-0}
  if [ "$p" -eq 0 ]; then fail "$t ($label): NO publisher"; return 1; fi
  r=$(topic_rate "$t" "${3:-$WAIT}")
  if [ -z "$r" ]; then fail "$t ($label): $p pub / $s sub but NO messages in ${3:-$WAIT}s"; return 1; fi
  pass "$t ($label): ${r} Hz  (pub=$p sub=$s)"
  return 0
}

echo_once() {    # topic [field]
  local f; f=$(mktemp)
  if [ -n "${2:-}" ]; then
    cap "$WAIT" "$f" ros2 topic echo --once --qos-reliability best_effort --field "$2" "$1"
  else
    cap "$WAIT" "$f" ros2 topic echo --once --qos-reliability best_effort "$1"
  fi
  cat "$f"; rm -f "$f"
}

arr_count() { awk '/^data:/{f=1;next} f&&/^- /{n++} /^---/{f=0} END{print n+0}'; }
arr_rows()  { awk -v w="$1" '/^data:/{f=1;next} f&&/^- /{v[n++]=$2} /^---/{f=0}
              END{for(i=0;i<n;i+=w){s="           ";for(j=0;j<w&&i+j<n;j++)s=s sprintf("%.2f ",v[i+j]);print s}}'; }

has_node() {     # regex label required(1/0)
  if echo "$NODES" | grep -Eqi "$1"; then pass "node running: $2"
  elif [ "$3" = 1 ]; then fail "node NOT running: $2"
  else warn "node not found: $2"; fi
}

# ---------------------------------------------------------------- 0. processes / nodes
hdr "0. Processes and nodes"
pgrep -fa "gz sim|ign gazebo" >/dev/null && pass "Gazebo running" || fail "Gazebo process not found"
pgrep -fa "bin/px4" >/dev/null && pass "PX4 SITL running" || warn "PX4 process not found"
pgrep -fa "MicroXRCEAgent" >/dev/null && pass "Micro XRCE-DDS agent running" || warn "MicroXRCEAgent not found (no PX4 <-> ROS 2 bridge)"
pgrep -fa "parameter_bridge|ros_gz_bridge" >/dev/null && pass "ros_gz_bridge running" || fail "ros_gz_bridge not running (no camera/lidar/imu on the ROS side)"

NODES=$(ros2 node list 2>/dev/null)
has_node "^${SEM_NODE#/}$|${SEM_NODE}\$" "semantic_map_node ($SEM_NODE)" 1
has_node "${TRK_NODE}\$" "object_tracker ($TRK_NODE)" 1
has_node "octomap" "octomap_server" 1
has_node "px4_offboard_bridge|offboard_bridge" "px4_offboard_bridge (republishes /model/drone/odometry)" 1
has_node "odom_tf|tf_broadcaster" "odom_tf_broadcaster (odom -> base_link / lidar TF)" 1
has_node "teleop" "keyboard_teleop (publishes /drone/control_mode)" 0
has_node "gui" "drone_gui" 0
echo "  all nodes:"; echo "$NODES" | sed 's/^/           /'

# ---------------------------------------------------------------- 1. topic liveness
hdr "1. Topic liveness (rate measured for ${WAIT}s each)"
check_topic /clock "Gazebo clock";                           R_CLOCK=$?
check_topic /model/drone/odometry "pose for semantic map";   R_ODOM=$?
check_topic /model/drone/camera "camera image";              R_CAM=$?
check_topic /model/drone/camera_info "intrinsics";           R_INFO=$?
check_topic /model/drone/scan/points "lidar cloud";          R_SCAN=$?
check_topic /map/points "octomap cloud" $((WAIT*2));         R_MAP=$?
check_topic /drone/targets "tracker detections";             R_TGT=$?
check_topic /map/targets_3d "live 3D targets";               R_T3D=$?
check_topic /map/targets_3d_fixed "fixed 3D targets";        R_FIX=$?
check_topic /map/targets_3d_moving "moving 3D targets";      R_MOV=$?
check_topic /drone/control_mode "control mode";              R_MODE=$?

if [ $R_ODOM -ne 0 ]; then
  info "/model/drone/odometry is now produced by px4_offboard_bridge from PX4's EKF2."
  info "PX4 side check:"
  ros2 topic list 2>/dev/null | grep -E "fmu/out/vehicle_(odometry|local_position|status)" | sed 's/^/           /'
  [ -z "$(ros2 topic list 2>/dev/null | grep fmu/out)" ] && info "no /fmu/out/* topics at all -> XRCE agent / PX4 not talking to ROS 2"
fi

# ---------------------------------------------------------------- 2. message content
hdr "2. Message content"
echo "  odometry position:"
ODOM_POS=$(echo_once /model/drone/odometry pose.pose.position | grep -v '^---')
[ -n "$ODOM_POS" ] && echo "$ODOM_POS" | sed 's/^/           /' || info "(no message)"
if echo "$ODOM_POS" | grep -qi "nan"; then fail "odometry contains NaN (semantic_map_node ignores it)"; fi

echo "  camera_info:"
CI_W=$(echo_once /model/drone/camera_info width | head -1)
CI_H=$(echo_once /model/drone/camera_info height | head -1)
CI_K=$(echo_once /model/drone/camera_info k | grep '^- ' | tr -d '\n-' | tr -s ' ')
if [ -n "$CI_W" ] && [ -n "$CI_K" ]; then
  info "size=${CI_W}x${CI_H}  K=[${CI_K}]"
  FX=$(echo "$CI_K" | awk '{print $1}')
  if awk -v f="$FX" 'BEGIN{exit !(f+0 <= 0)}'; then fail "camera_info K has fx=0 -> projection is garbage (gz camera_info not filled?)"; else
    awk -v f="$FX" -v w="$CI_W" 'BEGIN{printf "           implied HFOV = %.1f deg (tracker assumes camera_hfov_deg=80)\n", 2*atan2(w/2,f)*180/3.14159265}'
  fi
else
  fail "camera_info: no message / unreadable -> semantic_map_node returns silently on every detection"
  info "Gazebo camera topics (bridge.yaml maps gz '/model/drone/camera_info'):"
  command -v gz >/dev/null && gz topic -l 2>/dev/null | grep -i camera | sed 's/^/           /'
fi

echo "  frame ids:"
SCAN_FRAME=$(echo_once /model/drone/scan/points header.frame_id | head -1)
CAM_FRAME=$(echo_once /model/drone/camera header.frame_id | head -1)
MAP_FRAME=$(echo_once /map/points header.frame_id | head -1)
info "lidar='${SCAN_FRAME:-?}'  camera='${CAM_FRAME:-?}'  /map/points='${MAP_FRAME:-?}' (octomap frame_id should be odom)"
[ -n "$MAP_FRAME" ] && [ "$MAP_FRAME" != "odom" ] && fail "/map/points frame is '$MAP_FRAME' but semantic_map_node/GUI assume 'odom'"

echo "  TF (needed so octomap can place lidar scans in odom):"
for pair in "odom base_link" "odom ${SCAN_FRAME:-base_link}"; do
  set -- $pair
  f=$(mktemp); cap 4 "$f" ros2 run tf2_ros tf2_echo "$1" "$2"
  if grep -q "At time" "$f"; then pass "TF $1 -> $2 available"
  else fail "TF $1 -> $2 NOT available -> octomap drops every scan, /map/points stays empty"; head -3 "$f" | sed 's/^/           /'; fi
  rm -f "$f"
done

echo "  /map/points size:"
MAPW=$(echo_once /map/points width | head -1)
info "width=${MAPW:-?} points"
if [ -n "$MAPW" ] && [ "$MAPW" -eq 0 ] 2>/dev/null; then fail "/map/points is empty"; fi

echo "  /drone/targets (id,x,y,w,h normalized):"
N=$(echo_once /drone/targets | arr_count)
if [ "$N" -gt 0 ]; then pass "tracker sees $((N/5)) target(s)"; echo_once /drone/targets | arr_rows 5
else fail "/drone/targets is EMPTY -> nothing detected: check camera image / HSV range / tracker (run tracker with debug_detection:=true). Aim the camera at a target and re-run."; fi

for t in /map/targets_3d /map/targets_3d_fixed /map/targets_3d_moving; do
  N=$(echo_once $t | arr_count)
  echo "  $t: $((N/4)) entr(y/ies)   (id, x, y, z)"
  [ "$N" -gt 0 ] && echo_once $t | arr_rows 4
done

echo "  /drone/control_mode:"
MODE=$(echo_once /drone/control_mode data | head -1)
info "mode='${MODE:-<none>}'   (drone_gui draws the LIVE marker + trail only in 'auto'; fixed markers in any mode)"
[ "$MODE" != "auto" ] && warn "not in auto mode -> live tracked-object marker will not be drawn"

# ---------------------------------------------------------------- 3. semantic_map_node internals
hdr "3. semantic_map_node internals ($SEM_NODE)"
for p in gimbal_pitch_rad ground_plane_z target_center_height_m motion_max_tilt_rate_rad_s \
         require_lidar_in_range lidar_visible_range_m fixed_confirm_min_span_sec use_sim_time; do
  printf "  %-30s %s\n" "$p" "$(ros2 param get "$SEM_NODE" "$p" 2>&1 | head -1)"
done

echo "  capturing its DBG status line from /rosout (7s)..."
f=$(mktemp); cap 7 "$f" ros2 topic echo --qos-reliability best_effort --field msg /rosout
DBG=$(grep "DBG odom" "$f" | tail -1)
if [ -z "$DBG" ]; then
  warn "no DBG line seen (node not running, debug_log=false, or rosout not received)"
else
  info "$DBG"
  echo "$DBG" | grep -q "odom=MISSING"     && fail "node: odom=MISSING  -> no pose, nothing localizes (px4_offboard_bridge?)"
  echo "$DBG" | grep -q "cam_info=MISSING" && fail "node: cam_info=MISSING -> no camera intrinsics, every detection is dropped"
  echo "$DBG" | grep -q "map_pts=0 "       && warn "node: map_pts=0 -> no octomap points (lidar-confirm impossible; only camera-ray markers, >35 m)"
  echo "$DBG" | grep -q "disturbed=True"   && warn "node: disturbed=True -> tilt rate over threshold; motion/fixed logic keeps resetting (raise motion_max_tilt_rate_rad_s)"
  echo "$DBG" | grep -q "fixed=0"          && warn "node: fixed=0 -> no permanent markers created yet"
fi
NCONF=$(grep -c "DBG lidar-confirm: no vehicle-sized" "$f")
NREJ=$(grep -c "lidar match rejected" "$f")
NFIX=$(grep -c "Fixed target #" "$f")
info "in last 7s: lidar-confirm-misses=$NCONF  lidar-match-rejected=$NREJ  fixed-marker-events=$NFIX"
[ "$NCONF" -gt 0 ] && warn "octomap has no vehicle-sized cluster where the camera says a target is -> check ground_plane_z vs real spawn/ground height (section 4)"
rm -f "$f"

# ---------------------------------------------------------------- 4. ground-height assumption
hdr "4. Ground height assumption (ground_plane_z = ground_world_z - drone_spawn_world_z)"
info "semantic_map_node comment assumes spawn z=3.2, ground z=2.5 -> -0.7;  octomap_params.yaml says spawn z=5.0."
if command -v gz >/dev/null; then
  for topic in "/world/$WORLD/dynamic_pose/info" "/world/$WORLD/pose/info"; do
    f=$(mktemp); timeout 6 gz topic -e -n 1 -t "$topic" >"$f" 2>&1
    if grep -q "name:" "$f"; then
      info "poses from $topic:"
      grep -E -A8 'name: "(drone|fixed_target_[0-9]|moving_box_[0-9])"' "$f" | grep -E 'name:|z:' | sed 's/^/           /' | head -24
      rm -f "$f"; break
    fi
    rm -f "$f"
  done
  info "Compare the drone's world z (at spawn) with the ground z under a fixed_target; then:"
  info "  ros2 param set $SEM_NODE ground_plane_z <ground_z - spawn_z>"
else
  warn "gz CLI not found; read the drone's spawn z and terrain height from baylands_dynamic.sdf"
fi

# ---------------------------------------------------------------- 5. verdict
hdr "5. Verdict"
if [ ${#FAILS[@]} -eq 0 ]; then
  echo "  No hard failures. If markers still don't show, re-run while the camera is looking at a target"
  echo "  and read the WARN lines above (tilt disturbance, ground height, control mode)."
else
  echo "  ${#FAILS[@]} failure(s), fix in this order (earlier ones cause later ones):"
  i=1; for x in "${FAILS[@]}"; do echo "   $i. $x"; i=$((i+1)); done
fi
echo
echo "  Pipeline order:  odometry -> camera+camera_info -> /drone/targets -> semantic_map_node -> /map/targets_3d* -> drone_gui"
echo "  The first broken link in that chain is your root cause."
echo
echo "Report saved to $REPORT"
