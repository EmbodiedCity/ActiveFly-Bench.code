#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import json
import logging
import math
import os
import sys
import threading
import time
from io import BytesIO
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import requests
from PIL import Image

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion


ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"


def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def color_text(text: str, color: str) -> str:
    if _supports_color(sys.stdout):
        return f"{color}{text}{ANSI_RESET}"
    return text


class ColorFormatter(logging.Formatter):
    def __init__(self, fmt: str, use_color: bool) -> None:
        super().__init__(fmt)
        self.use_color = use_color
        self.level_colors = {
            logging.DEBUG: ANSI_DIM,
            logging.INFO: ANSI_GREEN,
            logging.WARNING: ANSI_YELLOW,
            logging.ERROR: ANSI_RED,
            logging.CRITICAL: f"{ANSI_RED}{ANSI_BOLD}",
        }

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if not self.use_color:
            return base
        color = self.level_colors.get(record.levelno, "")
        if not color:
            return base
        return f"{color}{base}{ANSI_RESET}"


def _import_gimbal_state():
    try:
        from gimbal.msg import GimbalState as ImportedGimbalState
        return ImportedGimbalState
    except ImportError:
        return None


def _import_vla_msgs():
    try:
        from vla_client.msg import VlaStatus as ImportedVlaStatus
        from vla_client.msg import VlaWaypoint as ImportedVlaWaypoint
        return ImportedVlaStatus, ImportedVlaWaypoint
    except ImportError:
        return None, None


GimbalState = _import_gimbal_state()
VlaStatus, VlaWaypoint = _import_vla_msgs()

logger = logging.getLogger(__name__)

IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
DEFAULT_SERVER_PORT = 5007
DEFAULT_SERVER_URL = f"http://127.0.0.1:{DEFAULT_SERVER_PORT}/predict"
DEFAULT_INSTRUCTION = "Describe the UAV task here."
DEFAULT_IMAGE_PATH = os.path.expanduser("~/.ros/vla_client/latest.jpg")
DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/.ros/vla_client_runs")
DEFAULT_ROS_IMAGE_TOPIC = "/gimbal/image_raw"
DEFAULT_PREDICTED_WORLD_POSE_TOPIC = "/predicted_world_pose"
DEFAULT_VLA_WAYPOINT_TOPIC = "/vla_waypoint"
DEFAULT_VLA_STATUS_TOPIC = "/vla_status"

ACTION_SMALL_DELTA_POS = 5.0
ACTION_SMALL_DELTA_YAW = 2.0
ACTION_SMALL_DELTA_PITCH = 2.0
ACTION_SMALL_STEPS = 10


def configure_logger(log_file: str, log_level: str) -> None:
    """Configure this script logger without inheriting rospy/root handlers."""
    resolved_log_level = getattr(logging, log_level.upper(), logging.INFO)
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"

    logger.handlers.clear()
    logger.setLevel(resolved_log_level)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(resolved_log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    stream_stream = sys.stdout
    use_color = _supports_color(stream_stream)
    stream_handler = logging.StreamHandler(stream_stream)
    stream_handler.setLevel(resolved_log_level)
    stream_handler.setFormatter(ColorFormatter(log_format, use_color=use_color))

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


class RealPoseBuffer:
    """Cache UAV odometry and gimbal pitch in a left-handed control frame."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.has_odom = False
        self.has_gimbal = GimbalState is None

        self.odom_sub = rospy.Subscriber(
            "/ekf_quat/ekf_odom",
            Odometry,
            self._odom_callback,
            queue_size=1,
        )

        self.gimbal_sub = None
        if GimbalState is not None:
            self.gimbal_sub = rospy.Subscriber(
                "/gimbal/state",
                GimbalState,
                self._gimbal_callback,
                queue_size=1,
            )
        else:
            logger.warning(
                "Failed to import gimbal.msg.GimbalState, "
                "pose_now_real gimbal_pitch will stay at default value 0.0. "
                "Please source the catkin workspace setup file if needed."
            )

        logger.info("Subscribed ROS odom topic: /ekf_quat/ekf_odom")
        if self.gimbal_sub is not None:
            logger.info("Subscribed ROS gimbal topic: /gimbal/state")

    def _odom_callback(self, msg: Odometry) -> None:
        qx = msg.pose.pose.orientation.x
        qy = msg.pose.pose.orientation.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        _, _, yaw_rad = euler_from_quaternion([qx, qy, qz, qw])
        yaw_deg_right_hand = math.degrees(yaw_rad)

        # x forward positive, y left positive, z up positive, CCW yaw positive.
        # x forward positive, y right positive, z up positive, CW yaw positive.
        # Convert ROS right-handed odometry to the left-handed control frame.
        # The client uses x-forward, y-right, z-up, and clockwise-positive yaw.
        x_left_hand = msg.pose.pose.position.x
        y_left_hand = -msg.pose.pose.position.y
        z_left_hand = msg.pose.pose.position.z
        yaw_deg_left_hand = -yaw_deg_right_hand

        with self.lock:
            self.latest_pose[0] = x_left_hand
            self.latest_pose[1] = y_left_hand
            self.latest_pose[2] = z_left_hand
            self.latest_pose[3] = yaw_deg_left_hand
            self.has_odom = True

    def _gimbal_callback(self, msg) -> None:
        with self.lock:
            self.latest_pose[4] = float(msg.cam_angle_p) / 100.0
            self.has_gimbal = True

    def get_latest_pose(self) -> list:
        with self.lock:
            return self.latest_pose.copy()

    def has_full_pose(self) -> bool:
        with self.lock:
            return self.has_odom and self.has_gimbal

    def wait_for_first_pose(self, timeout: float = 10.0) -> bool:
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        last_report = time.monotonic()
        while not rospy.is_shutdown():
            with self.lock:
                has_odom = self.has_odom
                has_gimbal = self.has_gimbal
            if has_odom and has_gimbal:
                return True
            now = time.monotonic()
            if now - last_report >= 1.0:
                missing = []
                if not has_odom:
                    missing.append("odom(/ekf_quat/ekf_odom)")
                if not has_gimbal:
                    missing.append("gimbal(/gimbal/state)")
                logger.warning("Waiting for first pose: missing %s", ", ".join(missing))
                last_report = now
            if timeout is not None and (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return False


class RosImageBuffer:
    """Cache the latest image received from a ROS image topic as a PIL image."""

    def __init__(self, topic: str):
        self.topic = topic
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_image: Optional[Image.Image] = None
        self.latest_stamp = None

        self.sub = rospy.Subscriber(
            self.topic,
            RosImage,
            self._image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        logger.info("Subscribed ROS image topic: %s", self.topic)

    def _image_callback(self, msg: RosImage) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_img = Image.fromarray(cv_img).convert("RGB")
            with self.lock:
                self.latest_image = pil_img
                self.latest_stamp = msg.header.stamp
        except Exception as exc:
            logger.error("Failed to convert ROS image from %s: %s", self.topic, exc)

    def get_latest_image(self) -> Optional[Image.Image]:
        with self.lock:
            if self.latest_image is None:
                return None

    def wait_for_first_image(self, timeout: float = 10.0) -> bool:
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                if self.latest_image is not None:
                    return True
            if timeout is not None and (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return False


def send_prediction_request(
    image: Image.Image,
    proprio: np.ndarray,
    instr: str,
    server_url: str,
) -> Optional[Dict[str, Any]]:
    """Send a request to the inference service and return JSON response."""
    proprio_list = proprio.tolist()
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    image.save(img_io, format="PNG")
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode("utf-8")

    payload: Dict[str, Any] = {
        "image": img_base64,
        "proprio": proprio_list,
        "instr": instr,
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error("Request failed: %s", exc)
        return None


def maybe_override_proprio(proprio: np.ndarray, enabled: bool) -> np.ndarray:
    """Optionally override proprio right before sending request."""
    if not enabled:
        return proprio

    current = [float(x) for x in proprio.tolist()]
    print(f"Current proprio: {current}")
    user_input = input(
        "[Debug] Input proprio override as x,y,z,yaw,pitch (Enter to keep): "
    ).strip()
    if not user_input:
        return proprio

    try:
        values = [float(x.strip()) for x in user_input.split(',')]
    except ValueError:
        logger.warning("Invalid proprio override input, keep original proprio.")
        return proprio

    if len(values) != len(current):
        logger.warning(
            "Expected %s values, got %s. Keep original proprio.",
            len(current),
            len(values),
        )
        return proprio

    logger.info("Override proprio: %s -> %s", current, values)
    return np.array(values, dtype=np.float32)


def maybe_override_image_path(image_path: str, enabled: bool) -> str:
    """Optionally override image path right before sending request. Only used when --use_local_image is enabled."""
    if not enabled:
        return image_path

    print(f"Current image path: {image_path}")
    user_input = input("[Debug] Input image path override (Enter to keep): ").strip()
    if not user_input:
        return image_path

    if not os.path.exists(user_input):
        logger.warning("Image path does not exist: %s. Keep original path.", user_input)
        return image_path

    logger.info("Override image path: %s -> %s", image_path, user_input)
    return user_input


def _normalize_yaw(delta_yaw: float) -> float:
    delta_yaw = (delta_yaw + 180.0) % 360.0 - 180.0
    return delta_yaw


def _normalize_delta_pos(delta_val: float) -> float:
    abs_val = abs(delta_val)
    if abs_val < 15.0:
        return 0.0
    if abs_val < 20.0:
        return 20.0 if delta_val >= 0 else -20.0
    return delta_val


def _normalize_delta_yaw(delta_val: float) -> float:
    if abs(delta_val) < 3.0:
        return 0.0
    return delta_val


def save_curr_pose_real(
    save_dir: str,
    step_count: int,
    curr_pose_real: list,
    curr_proprio_real: np.ndarray,
) -> None:
    """Save the current world pose and local proprioception debug record."""
    pose_record = {
        "step": step_count,
        "timestamp": datetime.now().isoformat(),
        "curr_pose_real": [float(x) for x in curr_pose_real],
        "curr_proprio_real": [float(x) for x in curr_proprio_real.tolist()],
    }

    step_pose_path = os.path.join(save_dir, f"step_{step_count:04d}_curr_pose_real.json")
    history_pose_path = os.path.join(save_dir, "curr_pose_real_history.jsonl")

    with open(step_pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_record, f, ensure_ascii=False, indent=2)

    with open(history_pose_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pose_record, ensure_ascii=False) + "\n")


def transform_pose_to_init_frame(curr_pose_real: list, init_pose_real: list) -> np.ndarray:
    """Transform a world-frame pose into the initial local frame."""
    dx_world = float(curr_pose_real[0]) - float(init_pose_real[0])
    dy_world = float(curr_pose_real[1]) - float(init_pose_real[1])
    dz_world = float(curr_pose_real[2]) - float(init_pose_real[2])

    init_yaw_deg = float(init_pose_real[3])
    init_yaw_rad = math.radians(init_yaw_deg)
    cos_yaw = math.cos(init_yaw_rad)
    sin_yaw = math.sin(init_yaw_rad)

    local_x = dx_world * cos_yaw + dy_world * sin_yaw
    local_y = -dx_world * sin_yaw + dy_world * cos_yaw
    local_z = dz_world
    local_yaw = _normalize_yaw(float(curr_pose_real[3]) - init_yaw_deg)
    local_pitch = float(curr_pose_real[4]) - float(init_pose_real[4])

    return np.array(
        [local_x, local_y, local_z, local_yaw, local_pitch],
        dtype=np.float32,
    )


def transform_pose_from_init_frame_to_world(local_pose: list, init_pose_real: list) -> np.ndarray:
    """Transform an initial-frame local pose back to the world frame."""
    local_x = float(local_pose[0])
    local_y = float(local_pose[1])
    local_z = float(local_pose[2])
    local_yaw = float(local_pose[3])
    local_pitch = float(local_pose[4])

    init_x = float(init_pose_real[0])
    init_y = float(init_pose_real[1])
    init_z = float(init_pose_real[2])
    init_yaw_deg = float(init_pose_real[3])
    init_pitch = float(init_pose_real[4])

    init_yaw_rad = math.radians(init_yaw_deg)
    cos_yaw = math.cos(init_yaw_rad)
    sin_yaw = math.sin(init_yaw_rad)

    dx_world = local_x * cos_yaw - local_y * sin_yaw
    dy_world = local_x * sin_yaw + local_y * cos_yaw

    world_x = init_x + dx_world
    world_y = init_y + dy_world
    world_z = init_z + local_z
    world_yaw = _normalize_yaw(init_yaw_deg + local_yaw)
    world_pitch = init_pitch + local_pitch

    return np.array(
        [world_x, world_y, world_z, world_yaw, world_pitch],
        dtype=np.float32,
    )


def save_predicted_world_pose(
    save_dir: str,
    step_count: int,
    predicted_local_pose: list,
    predicted_world_pose: np.ndarray,
) -> None:
    """Save the selected local action and resulting world-frame target."""
    pose_record = {
        "step": step_count,
        "timestamp": datetime.now().isoformat(),
        "predicted_local_pose": [float(x) for x in predicted_local_pose],
        "predicted_world_pose": [float(x) for x in predicted_world_pose.tolist()],
    }

    step_pose_path = os.path.join(save_dir, f"step_{step_count:04d}_predicted_world_pose.json")
    history_pose_path = os.path.join(save_dir, "predicted_world_pose_history.jsonl")

    with open(step_pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_record, f, ensure_ascii=False, indent=2)

    with open(history_pose_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(pose_record, ensure_ascii=False) + "\n")


def publish_predicted_world_pose(
    predicted_world_pose_pub: Optional[rospy.Publisher],
    predicted_world_pose: np.ndarray,
) -> None:
    if predicted_world_pose_pub is None:
        return

    msg = Float32MultiArray()
    msg.data = [float(x) for x in predicted_world_pose.tolist()]
    predicted_world_pose_pub.publish(msg)


class VlaStatusBuffer:
    def __init__(self, topic: str):
        if VlaStatus is None:
            raise RuntimeError("vla_client/VlaStatus is unavailable. Please source this catkin workspace setup file first.")
        self.vla_status_cls = VlaStatus
        self.topic = topic
        self.lock = threading.Lock()
        self.latest_msg = None
        self.last_seq_done = -1
        self.sub = rospy.Subscriber(topic, self.vla_status_cls, self._callback, queue_size=10)
        logger.info("Subscribed VLA status topic: %s", self.topic)

    def _callback(self, msg):
        with self.lock:
            self.latest_msg = msg
            if int(msg.status) in (
                self.vla_status_cls.DONE,
                self.vla_status_cls.PLAN_FAILED,
                self.vla_status_cls.HOLD_FAILED,
                self.vla_status_cls.GIMBAL_TIMEOUT,
                self.vla_status_cls.COMMAND_TIMEOUT,
                self.vla_status_cls.ABORTED,
                self.vla_status_cls.REJECTED_BUSY,
            ):
                self.last_seq_done = int(msg.seq_id)
        logger.info(
            "Received VLA status seq=%s status=%s message=%s pos_err=%.3f yaw_err=%.3f pitch_err=%.3f",
            int(msg.seq_id),
            int(msg.status),
            msg.message,
            float(msg.position_error),
            float(msg.yaw_error_deg),
            float(msg.pitch_error_deg),
        )

    def wait_for_ready(self, timeout: float) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                msg = self.latest_msg
                if msg is not None and int(msg.status) in (self.vla_status_cls.IDLE, self.vla_status_cls.DONE):
                    return True
            if rospy.Time.now() > deadline:
                return False
            rate.sleep()
        return False

    def wait_for_seq_result(self, seq_id: int, timeout: float):
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                msg = self.latest_msg
                if msg is not None and int(msg.seq_id) == seq_id and int(msg.status) in (
                    self.vla_status_cls.DONE,
                    self.vla_status_cls.PLAN_FAILED,
                    self.vla_status_cls.HOLD_FAILED,
                    self.vla_status_cls.GIMBAL_TIMEOUT,
                    self.vla_status_cls.COMMAND_TIMEOUT,
                    self.vla_status_cls.ABORTED,
                    self.vla_status_cls.REJECTED_BUSY,
                ):
                    return msg
            if rospy.Time.now() > deadline:
                return None
            rate.sleep()
        return None


def publish_vla_waypoint(
    vla_waypoint_pub: Optional[rospy.Publisher],
    seq_id: int,
    predicted_world_pose: np.ndarray,
    position_tolerance: float,
    yaw_tolerance_deg: float,
    pitch_tolerance_deg: float,
    timeout_sec: float,
    drone_id: int,
    busy_policy: int,
    frame_id: str,
) -> None:
    if vla_waypoint_pub is None:
        return
    if VlaWaypoint is None:
        raise RuntimeError("vla_client/VlaWaypoint is unavailable. Please source this catkin workspace setup file first.")

    vla_waypoint_cls = VlaWaypoint
    msg = vla_waypoint_cls()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.drone_id = drone_id
    msg.seq_id = seq_id
    msg.frame_id = frame_id
    msg.position.x = float(predicted_world_pose[0])
    msg.position.y = float(predicted_world_pose[1])
    msg.position.z = float(predicted_world_pose[2])
    msg.yaw = math.radians(float(predicted_world_pose[3]))
    msg.pitch = float(predicted_world_pose[4])
    msg.position_tolerance = position_tolerance
    msg.yaw_tolerance_deg = yaw_tolerance_deg
    msg.pitch_tolerance_deg = pitch_tolerance_deg
    msg.timeout_sec = timeout_sec
    msg.busy_policy = busy_policy
    vla_waypoint_pub.publish(msg)

    logger.info(
        "Published VLA waypoint seq=%s frame=%s pos=(%.3f, %.3f, %.3f) yaw_deg=%.3f pitch_deg=%.3f",
        seq_id,
        frame_id,
        float(predicted_world_pose[0]),
        float(predicted_world_pose[1]),
        float(predicted_world_pose[2]),
        float(predicted_world_pose[3]),
        float(predicted_world_pose[4]),
    )


def control_loop(
    image_path: str,
    instruction: str,
    server_url: str,
    max_steps: Optional[int],
    manual_proprio: bool,
    manual_image_path: bool,
    use_local_image: bool,
    ros_image_buffer: Optional[RosImageBuffer],
    real_pose_buffer: Optional[RealPoseBuffer],
    predicted_world_pose_pub: Optional[rospy.Publisher],
    vla_waypoint_pub: Optional[rospy.Publisher],
    vla_status_buffer: Optional[VlaStatusBuffer],
    vla_position_tolerance: float,
    vla_yaw_tolerance_deg: float,
    vla_pitch_tolerance_deg: float,
    vla_timeout_sec: float,
    vla_ready_timeout_sec: float,
    drone_id: int,
    vla_frame_id: str,
    save_dir: str,
) -> None:
    if real_pose_buffer is None:
        logger.error("ROS pose mode selected but real_pose_buffer is None.")
        return

    step_count = 0
    small_count = 0

    if not use_local_image:
        if ros_image_buffer is None:
            logger.error("ROS image mode selected but ros_image_buffer is None.")
            return
        logger.info("Waiting for first ROS image from %s ...", ros_image_buffer.topic)
        if not ros_image_buffer.wait_for_first_image(timeout=10.0):
            logger.error("Timeout waiting for ROS image from %s", ros_image_buffer.topic)
            return

    logger.info("Waiting for first real pose ...")
    if not real_pose_buffer.wait_for_first_pose(timeout=10.0):
        logger.error("Timeout waiting for first real pose.")
        return

    if vla_status_buffer is None:
        logger.error("VLA status buffer is required in ROS mode.")
        return

    if not vla_status_buffer.wait_for_ready(timeout=vla_ready_timeout_sec):
        logger.error("Timeout waiting initial IDLE/DONE from /vla_status.")
        return

    init_pose_real = real_pose_buffer.get_latest_pose()
    logger.info("Initial Real Pose: %s", init_pose_real)
    print(f"init pose real: {init_pose_real}")

    logger.info("Starting Control Loop...")
    current_image_path = image_path

    while max_steps is None or step_count < max_steps:
        if rospy.core.is_initialized() and rospy.is_shutdown():
            logger.info("ROS shutdown detected, stopping control loop.")
            break

        separator_line = "=" * 60
        print(color_text(f"\n{separator_line}", ANSI_CYAN))
        logger.info("LOOP ITERATION: %s", step_count)
        print(color_text(f"{separator_line}\n", ANSI_CYAN))

        stop_due_small = False

        try:
            if use_local_image:
                current_image_path = maybe_override_image_path(
                    current_image_path,
                    enabled=manual_image_path,
                )
                if not os.path.exists(current_image_path):
                    logger.error("Image not found at: %s", current_image_path)
                    break
                image = Image.open(current_image_path).convert("RGB")
            else:
                image = ros_image_buffer.get_latest_image() if ros_image_buffer else None
                if image is None:
                    logger.warning("No ROS image received yet, waiting...")
                    rospy.sleep(0.1)
                    continue

            save_path = os.path.join(save_dir, save_filename)
            image.save(save_path)
            # logger.info("Saved input image to: %s", save_path)

        except Exception as exc:
            logger.error("Failed to read or save image: %s", exc)
            break

        # logger.info("Sending request for step %s...", step_count)

        curr_pose_real = real_pose_buffer.get_latest_pose()
        curr_proprio_real = transform_pose_to_init_frame(curr_pose_real, init_pose_real)
        last_proprio_real = curr_proprio_real.copy()
        logger.info("Current Real Pose: %s", curr_pose_real)
        logger.info("Current Proprio Real: %s", curr_proprio_real.tolist())
        save_curr_pose_real(
            save_dir=save_dir,
            step_count=step_count,
            curr_pose_real=curr_pose_real,
            curr_proprio_real=curr_proprio_real,
        )
        proprio_for_request = maybe_override_proprio(
            np.array(curr_proprio_real, dtype=np.float32),
            enabled=manual_proprio,
        )

        response = send_prediction_request(
            image=image,
            proprio=proprio_for_request,
            instr=instruction,
            server_url=server_url,
        )
        if response is None:
            logger.warning("No valid response, stopping.")
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Response 'action' is empty or invalid, stopping.")
            break
        
        action_pose = action_poses[0]
        if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 5):
            logger.warning("Invalid action element: %s", action_pose)
            continue

        logger.info("--- Model Prediction ---")
        logger.info("Model Prediction: %s", action_pose)

        relative_x, relative_y = float(action_pose[0]), float(action_pose[1])
        relative_z = float(action_pose[2])
        relative_yaw = float(np.degrees(action_pose[3]))
        relative_yaw = _normalize_yaw(relative_yaw)
        relative_pit = float(np.degrees(action_pose[4]))

        predicted_world_pose = transform_pose_from_init_frame_to_world(
            [relative_x, relative_y, relative_z, relative_yaw, relative_pit],
            init_pose_real,
        )
        
        predicted_world_pose[1] = -predicted_world_pose[1]
        predicted_world_pose[3] = -predicted_world_pose[3]
        
        # logger.info("Predicted World Pose: %s", predicted_world_pose.tolist())
        save_predicted_world_pose(
            save_dir=save_dir,
            step_count=step_count,
            predicted_local_pose=[relative_x, relative_y, relative_z, relative_yaw, relative_pit],
            predicted_world_pose=predicted_world_pose,
        )

        delta_x = relative_x - curr_proprio_real[0]
        delta_y = relative_y - curr_proprio_real[1]
        delta_z = relative_z - curr_proprio_real[2]
        delta_yaw = _normalize_yaw(relative_yaw - curr_proprio_real[3])
        delta_pit = relative_pit - curr_proprio_real[4]

        # # convert to right-hand coordinates
        # delta_x = delta_x
        # delta_y = -delta_y
        # delta_z = delta_z
        # delta_yaw = -delta_yaw

        logger.info(
            "Calculated Deltas -> x: %.3f, y: %.3f, z: %.3f, yaw: %.3f, pitch: %.3f",
            delta_x,
            delta_y,
            delta_z,
            delta_yaw,
            delta_pit,
        )
        logger.info(
            "Calculated World Pose -> x: %.3f, y: %.3f, z: %.3f, yaw: %.3f, pitch: %.3f",
            predicted_world_pose[0],
            predicted_world_pose[1],
            predicted_world_pose[2],
            predicted_world_pose[3],
            predicted_world_pose[4],
        )

        # print("-" * 30)
        # user_cmd = input(f"[Step {step_count}] Press Space to send pose topic and move, or 'q' to quit: ")
        # if user_cmd.strip().lower() in {"q", "quit", "exit"}:
        #     break

        publish_predicted_world_pose(predicted_world_pose_pub, predicted_world_pose)
        
        seq_id = step_count + 1
        if not vla_status_buffer.wait_for_ready(timeout=vla_ready_timeout_sec):
            logger.error("Planner is not ready to accept seq=%s within %.1f sec.", seq_id, vla_ready_timeout_sec)
            break

        publish_vla_waypoint(
            vla_waypoint_pub=vla_waypoint_pub,
            seq_id=seq_id,
            predicted_world_pose=predicted_world_pose,
            position_tolerance=vla_position_tolerance,
            yaw_tolerance_deg=vla_yaw_tolerance_deg,
            pitch_tolerance_deg=vla_pitch_tolerance_deg,
            timeout_sec=vla_timeout_sec,
            drone_id=drone_id,
            busy_policy=1,
            frame_id=vla_frame_id,
        )

        status_msg = vla_status_buffer.wait_for_seq_result(
            seq_id=seq_id,
            timeout=vla_timeout_sec + vla_ready_timeout_sec,
        )
        if status_msg is None:
            logger.error("Timeout waiting VLA completion feedback for seq=%s.", seq_id)
            break

        if int(status_msg.status) != vla_status_buffer.vla_status_cls.DONE:
            logger.error(
                "VLA seq=%s failed with status=%s message=%s",
                seq_id,
                int(status_msg.status),
                status_msg.message,
            )
            break
        
        print(color_text("-" * 30, ANSI_CYAN))
        user_cmd = input(
            color_text(
                f"[Step {step_count}] Confirm the UAV reaches the target. Press Space to continue, or 'q' to quit: ",
                ANSI_MAGENTA,
            )
        )
        if user_cmd.strip().lower() in {"q", "quit", "exit"}:
            break

        # check if the change is very small for several consecutive steps, if yes, automatically end the task to avoid potential stuck
        curr_pose_real = real_pose_buffer.get_latest_pose()
        curr_proprio_real = transform_pose_to_init_frame(curr_pose_real, init_pose_real)
        
        diffs = [abs(a - b) for a, b in zip(curr_proprio_real, last_proprio_real)]

        logger.info("Pose Updated to: %s", curr_proprio_real)

        if (
            all(d < ACTION_SMALL_DELTA_POS for d in diffs[:3])
            and diffs[3] < ACTION_SMALL_DELTA_YAW
            and diffs[4] < ACTION_SMALL_DELTA_PITCH
        ):
            small_count += 1
        else:
            small_count = 0

        if small_count >= ACTION_SMALL_STEPS:
            logger.info(
                "Detected x,y,z,yaw,pitch continuous %s steps change is very small, "
                "automatically end task.",
                ACTION_SMALL_STEPS,
            )
            stop_due_small = True

        if stop_due_small:
            break

        step_count += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Async control loop with ROS image topic input."
    )
    parser.add_argument("--server_url", type=str, default=DEFAULT_SERVER_URL)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image_path", type=str, default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument(
        "--manual_proprio",
        action="store_true",
        help="Manually override proprio before each request.",
    )
    parser.add_argument(
        "--manual_image_path",
        action="store_true",
        help="Manually override image path before each request. Only works with --use_local_image.",
    )
    parser.add_argument(
        "--use_local_image",
        action="store_true",
        help="Use local image_path instead of ROS image topic. Default is ROS image topic.",
    )
    parser.add_argument(
        "--ros_image_topic",
        type=str,
        default=DEFAULT_ROS_IMAGE_TOPIC,
        help="ROS image topic to subscribe, default: /gimbal/image_raw.",
    )
    parser.add_argument(
        "--predicted_world_pose_topic",
        type=str,
        default=DEFAULT_PREDICTED_WORLD_POSE_TOPIC,
        help="ROS topic used to publish predicted world pose as Float32MultiArray.",
    )
    parser.add_argument("--vla_waypoint_topic", type=str, default=DEFAULT_VLA_WAYPOINT_TOPIC)
    parser.add_argument("--vla_status_topic", type=str, default=DEFAULT_VLA_STATUS_TOPIC)
    parser.add_argument("--drone_id", type=int, default=0)
    parser.add_argument("--vla_frame_id", type=str, default="world")
    parser.add_argument("--vla_position_tolerance", type=float, default=0.25)
    parser.add_argument("--vla_yaw_tolerance_deg", type=float, default=8.0)
    parser.add_argument("--vla_pitch_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--vla_timeout_sec", type=float, default=60.0)
    parser.add_argument("--vla_ready_timeout_sec", type=float, default=60.0)
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for per-run images, JSON records, and run_log.txt.",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="run_log.txt",
        help="Log file name or path. Relative paths are saved inside the run output directory.",
    )
    args = parser.parse_args()

    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    timestamp_str = datetime.now().strftime("openvla_%Y%m%d_%H%M%S")
    save_dir = os.path.join(output_root, timestamp_str)

    try:
        os.makedirs(save_dir, exist_ok=True)
    except OSError as e:
        print(f"Failed to create output directory {save_dir}: {e}")
        return

    if os.path.isabs(args.log_file):
        log_file_path = args.log_file
    else:
        log_file_path = os.path.join(save_dir, args.log_file)

    configure_logger(log_file_path, args.log_level)
    logger.info("Logger initialized. Writing logs to: %s", os.path.abspath(log_file_path))
    logger.info("Run artifacts will be saved under: %s", save_dir)

    ros_image_buffer: Optional[RosImageBuffer] = None
    real_pose_buffer: Optional[RealPoseBuffer] = None
    predicted_world_pose_pub: Optional[rospy.Publisher] = None
    vla_waypoint_pub: Optional[rospy.Publisher] = None
    vla_status_buffer: Optional[VlaStatusBuffer] = None

    if not args.use_local_image:
        rospy.init_node("fast_uav_ros_image_client", anonymous=True)
        if VlaStatus is None or VlaWaypoint is None:
            logger.error("Failed to import vla_client/VlaStatus or VlaWaypoint. Please source this catkin workspace setup file.")
            return
        ros_image_buffer = RosImageBuffer(args.ros_image_topic)
        real_pose_buffer = RealPoseBuffer()
        predicted_world_pose_pub = rospy.Publisher(
            args.predicted_world_pose_topic,
            Float32MultiArray,
            queue_size=1,
        )
        vla_waypoint_pub = rospy.Publisher(
            args.vla_waypoint_topic,
            VlaWaypoint,
            queue_size=1,
        )
        vla_status_buffer = VlaStatusBuffer(args.vla_status_topic)
        logger.info("Publishing predicted world pose to ROS topic: %s", args.predicted_world_pose_topic)
        logger.info("Publishing VLA waypoint to ROS topic: %s", args.vla_waypoint_topic)
    
    print(color_text(f"Server URL: {args.server_url}", ANSI_BLUE))
    control_loop(
        image_path=args.image_path,
        instruction=args.instruction,
        server_url=args.server_url,
        max_steps=args.max_steps,
        manual_proprio=args.manual_proprio,
        manual_image_path=args.manual_image_path,
        use_local_image=args.use_local_image,
        ros_image_buffer=ros_image_buffer,
        real_pose_buffer=real_pose_buffer,
        predicted_world_pose_pub=predicted_world_pose_pub,
        vla_waypoint_pub=vla_waypoint_pub,
        vla_status_buffer=vla_status_buffer,
        vla_position_tolerance=args.vla_position_tolerance,
        vla_yaw_tolerance_deg=args.vla_yaw_tolerance_deg,
        vla_pitch_tolerance_deg=args.vla_pitch_tolerance_deg,
        vla_timeout_sec=args.vla_timeout_sec,
        vla_ready_timeout_sec=args.vla_ready_timeout_sec,
        drone_id=args.drone_id,
        vla_frame_id=args.vla_frame_id,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    main()


