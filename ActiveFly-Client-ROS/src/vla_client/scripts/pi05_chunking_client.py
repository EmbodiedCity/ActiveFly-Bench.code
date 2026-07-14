#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import queue
import json
import logging
import math
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
from PIL import Image

# ----------------- ROS Imports -----------------
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
from tf.transformations import euler_from_quaternion

# ----------------- OpenPI Client Imports -----------------
try:
    from openpi_client import image_tools as openpi_image_tools
    from openpi_client import websocket_client_policy
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Failed to import openpi_client. Please ensure you are in the uv virtual environment "
        "and have installed packages/openpi-client."
    ) from exc

# ----------------- UI / Logging Config -----------------
ANSI_RESET = "\033[0m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_BOLD = "\033[1m"

def _supports_color(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()

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

def configure_logger(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    log_format = "[%(levelname)s] %(asctime)s - %(message)s"
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False
    
    stream_stream = sys.stdout
    use_color = _supports_color(stream_stream)
    stream_handler = logging.StreamHandler(stream_stream)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(ColorFormatter(log_format, use_color=use_color))
    
    logger.handlers.clear()
    logger.addHandler(stream_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# ----------------- Configs -----------------
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
ACTIVEFLY_STATE_DIM: int = 7
DEFAULT_SERVER_IP = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000
DEFAULT_INSTRUCTION = "Describe the UAV task here."
DEFAULT_ROS_IMAGE_TOPIC = "/gimbal/image_raw"
DEFAULT_OUTPUT_ROOT = os.path.expanduser("~/.ros/vla_client_runs")

DEFAULT_OPEN_LOOP_HORIZON = 6
DEFAULT_TEMPORAL_ENSEMBLE_DECAY = 0.3
DEFAULT_TEMPORAL_ENSEMBLE_WINDOW = 2
DEFAULT_VLA_POSITION_TOLERANCE = 0.05
DEFAULT_CLIENT_REACH_TIMEOUT_SEC = 2.0
DEFAULT_SMALL_MOTION_STOP_STEPS = 0
DEFAULT_SMALL_DELTA_POS_M = 0.03
DEFAULT_SMALL_DELTA_YAW_DEG = 2.0
DEFAULT_SMALL_DELTA_PITCH_DEG = 2.0
DEFAULT_ASYNCHRONOUS_QUERY_INTERVAL_STEPS = 3
DEFAULT_MAX_PENDING_QUERIES = 1
DEFAULT_MAX_CHUNK_HISTORY = 2
DEFAULT_QUEUE_HORIZON = 4
DEFAULT_TRAJECTORY_STEP_PERIOD_SEC = 0.1


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
        try:
            from vla_client.msg import VlaWaypointQueue as ImportedVlaWaypointQueue
        except ImportError:
            ImportedVlaWaypointQueue = None
        return ImportedVlaStatus, ImportedVlaWaypoint, ImportedVlaWaypointQueue
    except ImportError:
        return None, None, None


GimbalState = _import_gimbal_state()
VlaStatus, VlaWaypoint, VlaWaypointQueue = _import_vla_msgs()

def color_text(text: str, color: str) -> str:
    return f"{color}{text}{ANSI_RESET}"


def get_policy_client(host: str, port: int) -> websocket_client_policy.WebsocketClientPolicy:
    return websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

def send_prediction_request_pi05(
    policy_client: websocket_client_policy.WebsocketClientPolicy,
    image: Image.Image,
    proprio_5d_local: np.ndarray,
    instr: str,
) -> Optional[np.ndarray]:
    """Send one observation to OpenPi and return its native action chunk."""
    proprio_5d_local = np.asarray(proprio_5d_local, dtype=np.float32).reshape(-1)
    if proprio_5d_local.shape[0] < 5:
        logger.error("OpenPi observation needs 5D local proprio, got shape=%s.", proprio_5d_local.shape)
        return None

    # ActiveFly/OpenPi state: [x, y, z, roll, pitch, yaw, gimbal_pitch].
    # This real-UAV loop controls [x, y, z, yaw, gimbal_pitch], so roll/body-pitch stay 0.
    proprio_7d = np.array([
        proprio_5d_local[0], proprio_5d_local[1], proprio_5d_local[2],
        0.0, 0.0, 
        proprio_5d_local[3], proprio_5d_local[4]
    ], dtype=np.float32)

    image_np = np.asarray(image)
    image_np = openpi_image_tools.resize_with_pad(image_np, *IMG_INPUT_SIZE)
    image_np = openpi_image_tools.convert_to_uint8(image_np)

    observation = {
        "observation/image": image_np,
        "observation/state": proprio_7d,
        "prompt": instr,
    }

    try:
        response = policy_client.infer(observation)
        actions = response.get("actions")
        return actions
    except Exception as exc:
        logger.error(f"WebSocket Request failed: {exc}")
        return None


def validate_action_chunk(action_poses: Any) -> Optional[np.ndarray]:
    """Validate OpenPi `actions` and normalize it to an N x 5 float32 chunk."""
    if action_poses is None:
        return None
    action_chunk = np.asarray(action_poses, dtype=np.float32)
    if action_chunk.ndim == 1:
        action_chunk = action_chunk[None, :]
    if action_chunk.ndim != 2 or action_chunk.shape[0] == 0 or action_chunk.shape[1] < 5:
        return None
    if not np.isfinite(action_chunk[:, :5]).all():
        return None
    return action_chunk[:, :5]


@dataclass
class InferenceRequest:
    query_id: int
    query_step: int
    query_time: float
    image: Image.Image
    proprio_local: np.ndarray
    instruction: str


@dataclass
class InferenceResult:
    query_id: int
    query_step: int
    query_time: float
    receive_time: float
    query_proprio_local: np.ndarray
    raw_chunk: Any = None
    action_chunk: Optional[np.ndarray] = None
    error: Optional[str] = None


@dataclass
class ChunkRecord:
    query_id: int
    query_step: int
    receive_step: int
    query_time: float
    receive_time: float
    query_proprio_local: np.ndarray
    action_chunk: np.ndarray


def prune_chunk_records(step_count: int, chunk_records: List[ChunkRecord]) -> List[ChunkRecord]:
    return [
        record
        for record in chunk_records
        if record.query_step + record.action_chunk.shape[0] > step_count
    ]


def limit_chunk_records(chunk_records: List[ChunkRecord], max_history: int) -> List[ChunkRecord]:
    if max_history <= 0 or len(chunk_records) <= max_history:
        return chunk_records
    return chunk_records[-max_history:]


class AsyncOpenPiWorker:
    """Runs the blocking OpenPi websocket inference call outside the flight loop."""

    def __init__(self, policy_client: websocket_client_policy.WebsocketClientPolicy):
        self.policy_client = policy_client
        self.request_queue: "queue.Queue[Optional[InferenceRequest]]" = queue.Queue()
        self.result_queue: "queue.Queue[InferenceResult]" = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="openpi-inference-worker", daemon=True)
        self.thread.start()

    def submit(self, request: InferenceRequest) -> None:
        self.request_queue.put(request)

    def shutdown(self) -> None:
        self.request_queue.put(None)

    def _run(self) -> None:
        while True:
            request = self.request_queue.get()
            if request is None:
                self.request_queue.task_done()
                break

            try:
                raw_chunk = send_prediction_request_pi05(
                    self.policy_client,
                    request.image,
                    request.proprio_local,
                    request.instruction,
                )
                action_chunk = validate_action_chunk(raw_chunk)
                error = None if action_chunk is not None else "invalid_action_chunk"
                self.result_queue.put(
                    InferenceResult(
                        query_id=request.query_id,
                        query_step=request.query_step,
                        query_time=request.query_time,
                        receive_time=time.monotonic(),
                        query_proprio_local=request.proprio_local.copy(),
                        raw_chunk=raw_chunk,
                        action_chunk=action_chunk,
                        error=error,
                    )
                )
            except Exception as exc:
                self.result_queue.put(
                    InferenceResult(
                        query_id=request.query_id,
                        query_step=request.query_step,
                        query_time=request.query_time,
                        receive_time=time.monotonic(),
                        query_proprio_local=request.proprio_local.copy(),
                        error=str(exc),
                    )
                )
            finally:
                self.request_queue.task_done()


def prune_action_chunk_history(
    step_count: int,
    action_chunk_history: List[Tuple[int, np.ndarray]],
) -> List[Tuple[int, np.ndarray]]:
    return [
        (query_step, chunk)
        for query_step, chunk in action_chunk_history
        if query_step + chunk.shape[0] > step_count
    ]


def _blend_angle_delta_deg(values: np.ndarray, weights: np.ndarray) -> float:
    radians = np.deg2rad(values)
    sin_sum = float(np.sum(np.sin(radians) * weights))
    cos_sum = float(np.sum(np.cos(radians) * weights))
    if abs(sin_sum) < 1e-8 and abs(cos_sum) < 1e-8:
        return float(np.average(values, weights=weights))
    return float(_normalize_yaw(math.degrees(math.atan2(sin_sum, cos_sum))))


def blend_chunk_predictions(
    step_count: int,
    action_chunk_history: List[Tuple[int, np.ndarray]],
    ensemble_decay: float,
    ensemble_window: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for query_step, action_chunk in reversed(action_chunk_history):
        chunk_index = step_count - query_step
        if 0 <= chunk_index < action_chunk.shape[0]:
            candidates.append(
                {
                    "query_step": int(query_step),
                    "chunk_index": int(chunk_index),
                    "chunk_len": int(action_chunk.shape[0]),
                    "action": action_chunk[chunk_index].astype(np.float32, copy=False),
                }
            )
            if len(candidates) >= ensemble_window:
                break

    metadata: Dict[str, Any] = {
        "step": int(step_count),
        "ensemble_decay": float(ensemble_decay),
        "ensemble_window": int(ensemble_window),
        "candidates": [],
    }
    if not candidates:
        return None, metadata

    decay = min(max(float(ensemble_decay), 0.0), 1.0)
    weights = np.asarray([decay**i for i in range(len(candidates))], dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=np.float32)
    weights /= weights.sum()

    for candidate, weight in zip(candidates, weights):
        metadata["candidates"].append(
            {
                "query_step": candidate["query_step"],
                "chunk_index": candidate["chunk_index"],
                "chunk_len": candidate["chunk_len"],
                "weight": float(weight),
                "action_delta": candidate["action"].tolist(),
            }
        )

    if len(candidates) == 1:
        selected = candidates[0]["action"].astype(np.float32, copy=False)
        metadata["selected_action_delta"] = selected.tolist()
        return selected, metadata

    stacked = np.stack([candidate["action"] for candidate in candidates], axis=0)
    blended = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    blended[3] = _blend_angle_delta_deg(stacked[:, 3], weights)
    metadata["selected_action_delta"] = blended.tolist()
    return blended, metadata


def blend_chunk_record_predictions(
    step_count: int,
    chunk_records: List[ChunkRecord],
    ensemble_decay: float,
    ensemble_window: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for record in reversed(chunk_records):
        chunk_index = step_count - record.query_step
        if 0 <= chunk_index < record.action_chunk.shape[0]:
            candidates.append(
                {
                    "query_id": int(record.query_id),
                    "query_step": int(record.query_step),
                    "receive_step": int(record.receive_step),
                    "latency_steps": int(record.receive_step - record.query_step),
                    "chunk_index": int(chunk_index),
                    "chunk_len": int(record.action_chunk.shape[0]),
                    "action": record.action_chunk[chunk_index].astype(np.float32, copy=False),
                }
            )
            if len(candidates) >= ensemble_window:
                break

    metadata: Dict[str, Any] = {
        "step": int(step_count),
        "ensemble_decay": float(ensemble_decay),
        "ensemble_window": int(ensemble_window),
        "candidates": [],
    }
    if not candidates:
        return None, metadata

    decay = min(max(float(ensemble_decay), 0.0), 1.0)
    weights = np.asarray([decay**i for i in range(len(candidates))], dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=np.float32)
    weights /= weights.sum()

    for candidate, weight in zip(candidates, weights):
        metadata["candidates"].append(
            {
                "query_id": candidate["query_id"],
                "query_step": candidate["query_step"],
                "receive_step": candidate["receive_step"],
                "latency_steps": candidate["latency_steps"],
                "chunk_index": candidate["chunk_index"],
                "chunk_len": candidate["chunk_len"],
                "weight": float(weight),
                "action_delta": candidate["action"].tolist(),
            }
        )

    if len(candidates) == 1:
        selected = candidates[0]["action"].astype(np.float32, copy=False)
        metadata["selected_action_delta"] = selected.tolist()
        return selected, metadata

    stacked = np.stack([candidate["action"] for candidate in candidates], axis=0)
    blended = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    blended[3] = _blend_angle_delta_deg(stacked[:, 3], weights)
    metadata["selected_action_delta"] = blended.tolist()
    return blended, metadata


def blend_chunk_record_targets(
    future_step: int,
    chunk_records: List[ChunkRecord],
    ensemble_decay: float,
    ensemble_window: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Blend absolute local targets after anchoring every action at its own query proprio."""
    candidates: List[Dict[str, Any]] = []
    for record in reversed(chunk_records):
        chunk_index = future_step - record.query_step
        if not (0 <= chunk_index < record.action_chunk.shape[0]):
            continue
        action = record.action_chunk[chunk_index].astype(np.float32, copy=False)
        target = np.asarray(record.query_proprio_local, dtype=np.float32).copy()
        target[:3] += action[:3]
        target[3] = _normalize_yaw(float(target[3]) + float(action[3]))
        target[4] += action[4]
        candidates.append(
            {
                "query_id": int(record.query_id),
                "query_step": int(record.query_step),
                "receive_step": int(record.receive_step),
                "latency_steps": int(record.receive_step - record.query_step),
                "chunk_index": int(chunk_index),
                "chunk_len": int(record.action_chunk.shape[0]),
                "action": action,
                "query_proprio_local": record.query_proprio_local,
                "target": target,
            }
        )
        if len(candidates) >= ensemble_window:
            break

    metadata: Dict[str, Any] = {
        "step": int(future_step),
        "ensemble_decay": float(ensemble_decay),
        "ensemble_window": int(ensemble_window),
        "candidates": [],
    }
    if not candidates:
        return None, metadata

    decay = min(max(float(ensemble_decay), 0.0), 1.0)
    weights = np.asarray([decay**i for i in range(len(candidates))], dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=np.float32)
    weights /= weights.sum()

    for candidate, weight in zip(candidates, weights):
        metadata["candidates"].append(
            {
                "query_id": candidate["query_id"],
                "query_step": candidate["query_step"],
                "receive_step": candidate["receive_step"],
                "latency_steps": candidate["latency_steps"],
                "chunk_index": candidate["chunk_index"],
                "chunk_len": candidate["chunk_len"],
                "weight": float(weight),
                "query_proprio_local": candidate["query_proprio_local"].tolist(),
                "action_delta": candidate["action"].tolist(),
                "target_proprio_local": candidate["target"].tolist(),
            }
        )

    stacked = np.stack([candidate["target"] for candidate in candidates], axis=0)
    blended = np.average(stacked, axis=0, weights=weights).astype(np.float32)
    blended[3] = _blend_angle_delta_deg(stacked[:, 3], weights)
    metadata["selected_target_proprio_local"] = blended.tolist()
    return blended, metadata


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, ensure_ascii=False, indent=2)


def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(make_jsonable(payload), ensure_ascii=False) + "\n")


def make_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value

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

class VlaStatusBuffer:
    def __init__(self, topic: str):
        if VlaStatus is None:
            raise RuntimeError("vla_client/VlaStatus is unavailable. Please source this catkin workspace setup file first.")
        self.vla_status_cls = VlaStatus
        self.topic = topic
        self.lock = threading.Lock()
        self.latest_msg = None
        self.last_seq_done = -1
        self.terminal_by_seq: Dict[int, Any] = {}
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
                self.terminal_by_seq[int(msg.seq_id)] = msg
                if len(self.terminal_by_seq) > 64:
                    oldest_seq = next(iter(self.terminal_by_seq))
                    self.terminal_by_seq.pop(oldest_seq, None)
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

    def get_terminal_result(self, seq_id: int):
        with self.lock:
            return self.terminal_by_seq.get(int(seq_id))

    def get_latest(self):
        with self.lock:
            return self.latest_msg

def _normalize_yaw(delta_yaw: float) -> float:
    return (delta_yaw + 180.0) % 360.0 - 180.0

def compute_tracking_error(curr_pose_left_hand: list, target_world_right_hand: np.ndarray) -> Tuple[float, float, float]:
    """Compare current odom pose with a right-hand world-frame target."""
    curr_x = float(curr_pose_left_hand[0])
    curr_y = -float(curr_pose_left_hand[1])
    curr_z = float(curr_pose_left_hand[2])
    curr_yaw = -float(curr_pose_left_hand[3])
    curr_pitch = float(curr_pose_left_hand[4])

    pos_err = math.sqrt(
        (curr_x - float(target_world_right_hand[0])) ** 2
        + (curr_y - float(target_world_right_hand[1])) ** 2
        + (curr_z - float(target_world_right_hand[2])) ** 2
    )
    yaw_err = abs(_normalize_yaw(float(target_world_right_hand[3]) - curr_yaw))
    pitch_err = abs(float(target_world_right_hand[4]) - curr_pitch)
    return pos_err, yaw_err, pitch_err

def wait_for_client_reach(
    real_pose_buffer: RealPoseBuffer,
    target_world_right_hand: np.ndarray,
    position_tolerance: float,
    yaw_tolerance_deg: float,
    pitch_tolerance_deg: float,
    timeout: float,
) -> Tuple[bool, Tuple[float, float, float]]:
    deadline = time.monotonic() + max(0.0, float(timeout))
    last_error = compute_tracking_error(real_pose_buffer.get_latest_pose(), target_world_right_hand)
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        last_error = compute_tracking_error(real_pose_buffer.get_latest_pose(), target_world_right_hand)
        if (
            last_error[0] <= position_tolerance
            and last_error[1] <= yaw_tolerance_deg
            and last_error[2] <= pitch_tolerance_deg
        ):
            return True, last_error
        if time.monotonic() >= deadline:
            return False, last_error
        rate.sleep()
    return False, last_error

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


def build_aligned_waypoint_queue(
    current_step: int,
    queue_horizon: int,
    max_steps: Optional[int],
    chunk_records: List[ChunkRecord],
    ensemble_decay: float,
    ensemble_window: int,
    init_pose_real: list,
    current_proprio_local: np.ndarray,
    action_semantics: str,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Build a contiguous world-frame queue from globally aligned future chunk rows."""
    if action_semantics not in {"incremental", "query_relative"}:
        raise ValueError(f"Unsupported queue action semantics: {action_semantics}")
    world_poses: List[np.ndarray] = []
    point_metadata: List[Dict[str, Any]] = []
    integrated_target = np.asarray(current_proprio_local, dtype=np.float32).copy()
    final_step = current_step + queue_horizon
    if max_steps is not None:
        final_step = min(final_step, max_steps)

    for future_step in range(current_step, final_step):
        if action_semantics == "query_relative":
            target_local, metadata = blend_chunk_record_targets(
                future_step,
                chunk_records,
                ensemble_decay,
                ensemble_window,
            )
            if target_local is None:
                break
        else:
            action_delta, metadata = blend_chunk_record_predictions(
                future_step,
                chunk_records,
                ensemble_decay,
                ensemble_window,
            )
            if action_delta is None:
                break
            integrated_target[:3] += action_delta[:3]
            integrated_target[3] = _normalize_yaw(
                float(integrated_target[3]) + float(action_delta[3])
            )
            integrated_target[4] += action_delta[4]
            target_local = integrated_target.copy()
            metadata["selected_target_proprio_local"] = target_local.tolist()
        metadata["action_semantics"] = action_semantics
        world_pose = transform_pose_from_init_frame_to_world(target_local.tolist(), init_pose_real)
        world_pose[1] = -world_pose[1]
        world_pose[3] = -world_pose[3]
        world_poses.append(world_pose)
        point_metadata.append(metadata)

    if not world_poses:
        return np.empty((0, 5), dtype=np.float32), point_metadata
    return np.stack(world_poses).astype(np.float32), point_metadata


def publish_vla_waypoint_queue(
    vla_waypoint_queue_pub: Optional[rospy.Publisher],
    seq_id: int,
    world_poses: np.ndarray,
    point_metadata: List[Dict[str, Any]],
    position_tolerance: float,
    yaw_tolerance_deg: float,
    pitch_tolerance_deg: float,
    timeout_sec: float,
    drone_id: int,
    busy_policy: str,
    frame_id: str,
) -> Dict[str, int]:
    if vla_waypoint_queue_pub is None or VlaWaypointQueue is None:
        raise RuntimeError(
            "vla_client/VlaWaypointQueue is unavailable. Rebuild and source the ROS workspace first."
        )
    if world_poses.ndim != 2 or world_poses.shape[0] == 0 or world_poses.shape[1] < 5:
        raise ValueError("world_poses must be a non-empty N x 5 waypoint queue")
    if len(point_metadata) != world_poses.shape[0]:
        raise ValueError("point_metadata length must match the waypoint queue")

    policy_by_name = {
        "accept_if_idle": VlaWaypointQueue.ACCEPT_IF_IDLE,
        "reject_if_busy": VlaWaypointQueue.REJECT_IF_BUSY,
        "replace_active": VlaWaypointQueue.REPLACE_ACTIVE,
    }
    if busy_policy not in policy_by_name:
        raise ValueError(f"Unsupported queue busy policy: {busy_policy}")

    first_sources = point_metadata[0].get("candidates", [])
    primary_source = first_sources[0] if first_sources else {}
    source_metadata = {
        "query_id": int(primary_source.get("query_id", 0)),
        "query_step": int(primary_source.get("query_step", 0)),
        "first_chunk_index": int(primary_source.get("chunk_index", 0)),
    }

    msg = VlaWaypointQueue()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.drone_id = drone_id
    msg.seq_id = seq_id
    msg.frame_id = frame_id
    for pose in world_poses:
        point = Point()
        point.x = float(pose[0])
        point.y = float(pose[1])
        point.z = float(pose[2])
        msg.positions.append(point)
        msg.yaws.append(math.radians(float(pose[3])))
        msg.pitches.append(float(pose[4]))
    msg.position_tolerance = position_tolerance
    msg.yaw_tolerance_deg = yaw_tolerance_deg
    msg.pitch_tolerance_deg = pitch_tolerance_deg
    msg.timeout_sec = timeout_sec
    msg.busy_policy = policy_by_name[busy_policy]
    msg.query_id = source_metadata["query_id"]
    msg.query_step = source_metadata["query_step"]
    msg.first_chunk_index = source_metadata["first_chunk_index"]
    vla_waypoint_queue_pub.publish(msg)

    logger.info(
        "Published VLA waypoint queue seq=%s size=%s policy=%s query_id=%s query_step=%s "
        "first_chunk_index=%s final=(%.3f, %.3f, %.3f, yaw=%.3fdeg, pitch=%.3fdeg)",
        seq_id,
        world_poses.shape[0],
        busy_policy,
        source_metadata["query_id"],
        source_metadata["query_step"],
        source_metadata["first_chunk_index"],
        float(world_poses[-1, 0]),
        float(world_poses[-1, 1]),
        float(world_poses[-1, 2]),
        float(world_poses[-1, 3]),
        float(world_poses[-1, 4]),
    )
    return source_metadata

def log_chunk_source_summary(step_count: int, blend_metadata: Dict[str, Any]) -> None:
    chunk_sources = blend_metadata.get("candidates", [])
    if not chunk_sources:
        return

    def _source_text(source: Dict[str, Any]) -> str:
        base = (
            "query_step={query_step} chunk_index={chunk_index}/{last_index} "
            "chunk_len={chunk_len} weight={weight:.3f}"
        ).format(
            query_step=int(source["query_step"]),
            chunk_index=int(source["chunk_index"]),
            last_index=max(int(source["chunk_len"]) - 1, 0),
            chunk_len=int(source["chunk_len"]),
            weight=float(source["weight"]),
        )
        if "query_id" in source:
            base = (
                "query_id={query_id} ".format(query_id=int(source["query_id"]))
                + base
                + " receive_step={receive_step} latency_steps={latency_steps}".format(
                    receive_step=int(source.get("receive_step", -1)),
                    latency_steps=int(source.get("latency_steps", -1)),
                )
            )
        return base

    logger.info(
        "Chunk action source for step=%s: %s",
        step_count,
        "; ".join(_source_text(source) for source in chunk_sources),
    )


def create_inference_request(
    query_id: int,
    query_step: int,
    image: Image.Image,
    proprio_local: np.ndarray,
    instruction: str,
) -> InferenceRequest:
    return InferenceRequest(
        query_id=query_id,
        query_step=query_step,
        query_time=time.monotonic(),
        image=image.copy(),
        proprio_local=np.asarray(proprio_local, dtype=np.float32).copy(),
        instruction=instruction,
    )


def append_inference_result_as_chunk(
    result: InferenceResult,
    receive_step: int,
    reason: str,
) -> Tuple[Optional[ChunkRecord], Dict[str, Any]]:
    event: Dict[str, Any] = {
        "reason": reason,
        "query_id": int(result.query_id),
        "query_step": int(result.query_step),
        "receive_step": int(receive_step),
        "latency_sec": float(result.receive_time - result.query_time),
        "latency_steps": int(receive_step - result.query_step),
        "query_proprio_local": result.query_proprio_local.tolist(),
        "error": result.error,
    }

    if result.action_chunk is None:
        event["accepted"] = False
        event["drop_reason"] = result.error or "invalid_action_chunk"
        event["raw_action_chunk"] = make_jsonable(result.raw_chunk)
        return None, event

    action_chunk = result.action_chunk
    first_usable_index = receive_step - result.query_step
    event.update(
        {
            "raw_action_chunk": make_jsonable(result.raw_chunk),
            "validated_action_chunk": action_chunk.tolist(),
            "chunk_shape": list(action_chunk.shape),
            "first_usable_chunk_index": int(first_usable_index),
            "stale_prefix_actions": int(max(first_usable_index, 0)),
        }
    )

    if first_usable_index >= action_chunk.shape[0]:
        event["accepted"] = False
        event["drop_reason"] = "fully_stale"
        logger.warning(
            "Dropping stale async OpenPi result id=%s query_step=%s receive_step=%s "
            "latency_steps=%s chunk_len=%s.",
            result.query_id,
            result.query_step,
            receive_step,
            receive_step - result.query_step,
            action_chunk.shape[0],
        )
        return None, event

    event["accepted"] = True
    record = ChunkRecord(
        query_id=result.query_id,
        query_step=result.query_step,
        receive_step=receive_step,
        query_time=result.query_time,
        receive_time=result.receive_time,
        query_proprio_local=result.query_proprio_local.copy(),
        action_chunk=action_chunk,
    )
    logger.info(
        "Accepted async OpenPi result id=%s query_step=%s receive_step=%s latency=%.3fs "
        "latency_steps=%s chunk_shape=%s first_usable_chunk_index=%s stale_prefix=%s",
        result.query_id,
        result.query_step,
        receive_step,
        result.receive_time - result.query_time,
        receive_step - result.query_step,
        action_chunk.shape,
        first_usable_index,
        max(first_usable_index, 0),
    )
    return record, event


def control_loop(args, ros_image_buffer, real_pose_buffer, vla_waypoint_pub, vla_status_buffer, policy_client):
    if args.open_loop_horizon < 1:
        raise ValueError("open_loop_horizon must be >= 1")
    if args.temporal_ensemble_window < 1:
        raise ValueError("temporal_ensemble_window must be >= 1")

    step_count = 0
    small_count = 0
    last_executed_local: Optional[np.ndarray] = None
    action_chunk_history: List[Tuple[int, np.ndarray]] = []
    run_dir = getattr(args, "run_dir", None)
    action_history_path = os.path.join(run_dir, "action_chunk_history.jsonl") if run_dir else None
    waypoint_history_path = os.path.join(run_dir, "published_waypoints.jsonl") if run_dir else None

    logger.info("Waiting for first ROS image & Pose ...")
    if not ros_image_buffer.wait_for_first_image(timeout=10.0):
        logger.error("Timeout waiting for first ROS image from %s", ros_image_buffer.topic)
        return
    if not real_pose_buffer.wait_for_first_pose(timeout=10.0):
        logger.error("Timeout waiting for first real pose from /ekf_quat/ekf_odom")
        return
    if not vla_status_buffer.wait_for_ready(timeout=args.vla_ready_timeout_sec):
        logger.error("Timeout waiting for initial IDLE/DONE from %s", args.vla_status_topic)
        return

    init_pose_real = real_pose_buffer.get_latest_pose()
    logger.info(f"Initial Real Pose: {init_pose_real}")
    logger.info(
        "OpenPi motion chunking enabled: open_loop_horizon=%s temporal_ensemble_window=%s "
        "temporal_ensemble_decay=%.3f user_confirm=%s",
        args.open_loop_horizon,
        args.temporal_ensemble_window,
        args.temporal_ensemble_decay,
        not args.no_user_confirm,
    )

    while args.max_steps is None or step_count < args.max_steps:
        if rospy.is_shutdown():
            break

        print(color_text(f"\n{'='*60}\nLOOP ITERATION: {step_count}\n{'='*60}", ANSI_CYAN))

        image = ros_image_buffer.get_latest_image()
        if image is None:
            rospy.sleep(0.1); continue

        if run_dir:
            image.save(os.path.join(run_dir, f"step_{step_count:04d}.png"))

        curr_pose_real = real_pose_buffer.get_latest_pose()
        curr_proprio_local = transform_pose_to_init_frame(curr_pose_real, init_pose_real)
        logger.info(f"Current Proprio Local: {curr_proprio_local.tolist()}")

        query_events: List[Dict[str, Any]] = []
        action_chunk_history = prune_action_chunk_history(step_count, action_chunk_history)
        should_query = not action_chunk_history or step_count % args.open_loop_horizon == 0
        if should_query:
            raw_chunk = send_prediction_request_pi05(
                policy_client, image, curr_proprio_local, args.instruction
            )
            action_chunk = validate_action_chunk(raw_chunk)
            if action_chunk is None:
                logger.warning("Scheduled OpenPi request returned an invalid action chunk. Stopping.")
                break

            action_chunk_history.append((step_count, action_chunk))
            action_chunk_history = prune_action_chunk_history(step_count, action_chunk_history)
            query_events.append(
                {
                    "reason": "scheduled" if step_count % args.open_loop_horizon == 0 else "history_empty",
                    "query_step": int(step_count),
                    "raw_action_chunk": make_jsonable(raw_chunk),
                    "validated_action_chunk": action_chunk.tolist(),
                    "chunk_shape": list(action_chunk.shape),
                }
            )
            logger.info(
                "OpenPi inference query step=%s reason=%s returned action chunk shape=%s",
                step_count,
                query_events[-1]["reason"],
                action_chunk.shape,
            )
            if action_chunk.shape[0] < args.open_loop_horizon:
                logger.warning(
                    "Action chunk length %s is smaller than open_loop_horizon=%s; "
                    "the loop will request a fresh chunk early when cached rows are exhausted.",
                    action_chunk.shape[0],
                    args.open_loop_horizon,
                )

        action_delta, blend_metadata = blend_chunk_predictions(
            step_count, action_chunk_history, 
            args.temporal_ensemble_decay, args.temporal_ensemble_window
        )
        if action_delta is None:
            logger.warning("No usable action in cached chunks at step %s; forcing OpenPi refresh.", step_count)
            raw_chunk = send_prediction_request_pi05(
                policy_client, image, curr_proprio_local, args.instruction
            )
            action_chunk = validate_action_chunk(raw_chunk)
            if action_chunk is None:
                logger.warning("Forced OpenPi refresh returned an invalid action chunk. Stopping.")
                break
            action_chunk_history.append((step_count, action_chunk))
            action_chunk_history = prune_action_chunk_history(step_count, action_chunk_history)
            query_events.append(
                {
                    "reason": "forced_refresh",
                    "query_step": int(step_count),
                    "raw_action_chunk": make_jsonable(raw_chunk),
                    "validated_action_chunk": action_chunk.tolist(),
                    "chunk_shape": list(action_chunk.shape),
                }
            )
            logger.info(
                "OpenPi inference query step=%s reason=%s returned action chunk shape=%s",
                step_count,
                query_events[-1]["reason"],
                action_chunk.shape,
            )
            action_delta, blend_metadata = blend_chunk_predictions(
                step_count,
                action_chunk_history,
                args.temporal_ensemble_decay,
                args.temporal_ensemble_window,
            )
            if action_delta is None:
                logger.error("No valid action delta after forced chunk refresh.")
                break

        log_chunk_source_summary(step_count, blend_metadata)

        logger.info(f"Blended Action Delta (5D): {action_delta.tolist()}")

        # OpenPi/ActiveFly outputs 5D local deltas in degrees: [dx, dy, dz, dyaw, dgimbal_pitch].
        target_proprio_local = np.array([
            curr_proprio_local[0] + action_delta[0],
            curr_proprio_local[1] + action_delta[1],
            curr_proprio_local[2] + action_delta[2],
            _normalize_yaw(curr_proprio_local[3] + action_delta[3]),
            curr_proprio_local[4] + action_delta[4],
        ], dtype=np.float32)

        predicted_world_pose = transform_pose_from_init_frame_to_world(
            target_proprio_local.tolist(), init_pose_real
        )
        
        # The policy/proprio side uses a left-hand convention; Ego Planner expects ROS right-hand targets.
        predicted_world_pose[1] = -predicted_world_pose[1]
        predicted_world_pose[3] = -predicted_world_pose[3]
        
        logger.info(f"Target World Pose (Right-Hand): {predicted_world_pose.tolist()}")
        command_pos_err, command_yaw_err, command_pitch_err = compute_tracking_error(
            curr_pose_real,
            predicted_world_pose,
        )
        logger.info(
            "Command Delta vs Current Odom -> pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg "
            "(planner tolerance pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg)",
            command_pos_err,
            command_yaw_err,
            command_pitch_err,
            args.vla_position_tolerance,
            args.vla_yaw_tolerance_deg,
            args.vla_pitch_tolerance_deg,
        )
        if command_pos_err <= args.vla_position_tolerance and command_yaw_err <= args.vla_yaw_tolerance_deg:
            logger.warning(
                "Target is already inside the planner reach tolerance before sending. "
                "EGO may report DONE without visible UAV motion. Consider lowering "
                "--vla_position_tolerance or increasing policy/action scale."
            )

        seq_id = step_count + 1
        if not vla_status_buffer.wait_for_ready(timeout=args.vla_ready_timeout_sec):
            logger.error("Planner is not ready to accept seq=%s within %.1f sec.", seq_id, args.vla_ready_timeout_sec)
            break

        publish_vla_waypoint(
            vla_waypoint_pub, seq_id, predicted_world_pose,
            args.vla_position_tolerance, args.vla_yaw_tolerance_deg, args.vla_pitch_tolerance_deg,
            args.vla_timeout_sec, args.drone_id, 1, args.vla_frame_id
        )

        status_msg = vla_status_buffer.wait_for_seq_result(seq_id, args.vla_timeout_sec + args.vla_ready_timeout_sec)
        if status_msg is None or int(status_msg.status) != VlaStatus.DONE:
            if status_msg is None:
                logger.error("VLA execution seq=%s timed out.", seq_id)
            else:
                logger.error(
                    "VLA execution seq=%s failed: status=%s message=%s",
                    seq_id,
                    int(status_msg.status),
                    getattr(status_msg, "message", ""),
                )
            break

        reached, client_error = wait_for_client_reach(
            real_pose_buffer=real_pose_buffer,
            target_world_right_hand=predicted_world_pose,
            position_tolerance=args.client_position_tolerance,
            yaw_tolerance_deg=args.client_yaw_tolerance_deg,
            pitch_tolerance_deg=args.client_pitch_tolerance_deg,
            timeout=args.client_reach_timeout_sec,
        )
        logger.info(
            "Client Odom Reach Check -> reached=%s pos_err=%.3fm yaw_err=%.3fdeg pitch_err=%.3fdeg "
            "(client tolerance pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg)",
            reached,
            client_error[0],
            client_error[1],
            client_error[2],
            args.client_position_tolerance,
            args.client_yaw_tolerance_deg,
            args.client_pitch_tolerance_deg,
        )
        if args.require_client_reach and not reached:
            logger.error(
                "Planner reported DONE, but local odom did not reach the target within client tolerance. "
                "This usually means the waypoint tolerance is still too loose or the controller did not execute motion."
            )
            break

        executed_pose_real = real_pose_buffer.get_latest_pose()
        executed_local = transform_pose_to_init_frame(executed_pose_real, init_pose_real)
        if args.small_motion_stop_steps > 0 and last_executed_local is not None:
            pos_diff = np.abs(executed_local[:3] - last_executed_local[:3])
            yaw_diff = abs(_normalize_yaw(float(executed_local[3]) - float(last_executed_local[3])))
            pitch_diff = abs(float(executed_local[4]) - float(last_executed_local[4]))
            if (
                np.all(pos_diff < args.small_delta_pos_m)
                and yaw_diff < args.small_delta_yaw_deg
                and pitch_diff < args.small_delta_pitch_deg
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= args.small_motion_stop_steps:
                logger.info(
                    "Detected small executed motion for %s consecutive steps; ending task.",
                    args.small_motion_stop_steps,
                )
                break
        last_executed_local = executed_local

        step_record = {
            "step": int(step_count),
            "seq_id": int(seq_id),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "instruction": args.instruction,
            "open_loop_horizon": int(args.open_loop_horizon),
            "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
            "temporal_ensemble_window": int(args.temporal_ensemble_window),
            "curr_pose_real_left_hand": make_jsonable(curr_pose_real),
            "curr_proprio_local": curr_proprio_local.tolist(),
            "query_events": query_events,
            "active_chunk_history": [
                {"query_step": int(query_step), "chunk_shape": list(chunk.shape)}
                for query_step, chunk in action_chunk_history
            ],
            "blend": blend_metadata,
            "selected_action_delta": action_delta.tolist(),
            "target_proprio_local": target_proprio_local.tolist(),
            "predicted_world_pose_right_hand": predicted_world_pose.tolist(),
            "planner_done": True,
            "client_reached": bool(reached),
            "client_error": [float(v) for v in client_error],
            "executed_pose_real_left_hand": make_jsonable(executed_pose_real),
            "executed_proprio_local": executed_local.tolist(),
        }
        if run_dir:
            save_json(os.path.join(run_dir, f"step_{step_count:04d}_action_chunk.json"), step_record)
            append_jsonl(action_history_path, step_record)
            append_jsonl(
                waypoint_history_path,
                {
                    "step": int(step_count),
                    "seq_id": int(seq_id),
                    "selected_action_delta": action_delta.tolist(),
                    "target_proprio_local": target_proprio_local.tolist(),
                    "predicted_world_pose_right_hand": predicted_world_pose.tolist(),
                },
            )

        if not args.no_user_confirm:
            try:
                cmd = input(color_text(f"[Step {step_count}] Reached target. Press Enter to continue, 'q' to quit: ", ANSI_MAGENTA))
            except EOFError:
                logger.warning("stdin is closed; stopping before the next waypoint for safety.")
                break
            if cmd.strip().lower() in {"q", "quit", "exit"}:
                break

        step_count += 1


def control_loop_continuous_motion(args, ros_image_buffer, real_pose_buffer, vla_waypoint_pub, vla_status_buffer, policy_client):
    if args.async_query_interval_steps < 1:
        raise ValueError("async_query_interval_steps must be >= 1")
    if args.temporal_ensemble_window < 1:
        raise ValueError("temporal_ensemble_window must be >= 1")
    if args.max_pending_queries < 1:
        raise ValueError("max_pending_queries must be >= 1")
    if args.max_chunk_history < 1:
        raise ValueError("max_chunk_history must be >= 1")

    step_count = 0
    next_query_id = 0
    pending_query_steps: Dict[int, int] = {}
    small_count = 0
    last_executed_local: Optional[np.ndarray] = None
    chunk_records: List[ChunkRecord] = []
    worker = AsyncOpenPiWorker(policy_client)
    run_dir = getattr(args, "run_dir", None)
    action_history_path = os.path.join(run_dir, "action_chunk_history.jsonl") if run_dir else None
    waypoint_history_path = os.path.join(run_dir, "published_waypoints.jsonl") if run_dir else None

    def _submit_async_query(query_step: int, image: Image.Image, proprio_local: np.ndarray, reason: str) -> Dict[str, Any]:
        nonlocal next_query_id
        request = create_inference_request(
            query_id=next_query_id,
            query_step=query_step,
            image=image,
            proprio_local=proprio_local,
            instruction=args.instruction,
        )
        next_query_id += 1
        pending_query_steps[request.query_id] = request.query_step
        worker.submit(request)
        logger.info(
            "Submitted async OpenPi query id=%s query_step=%s reason=%s pending=%s",
            request.query_id,
            request.query_step,
            reason,
            len(pending_query_steps),
        )
        return {
            "event": "submitted",
            "reason": reason,
            "query_id": int(request.query_id),
            "query_step": int(request.query_step),
            "pending_queries": int(len(pending_query_steps)),
        }

    def _drain_results(receive_step: int, reason: str) -> List[Dict[str, Any]]:
        nonlocal chunk_records
        events: List[Dict[str, Any]] = []
        while True:
            try:
                result = worker.result_queue.get_nowait()
            except queue.Empty:
                break
            pending_query_steps.pop(result.query_id, None)
            record, event = append_inference_result_as_chunk(result, receive_step, reason)
            events.append(event)
            if record is not None:
                chunk_records.append(record)
                chunk_records = prune_chunk_records(receive_step, chunk_records)
                chunk_records = limit_chunk_records(chunk_records, args.max_chunk_history)
            worker.result_queue.task_done()
        return events

    logger.info("Waiting for first ROS image & Pose ...")
    if not ros_image_buffer.wait_for_first_image(timeout=10.0):
        logger.error("Timeout waiting for first ROS image from %s", ros_image_buffer.topic)
        worker.shutdown()
        return
    if not real_pose_buffer.wait_for_first_pose(timeout=10.0):
        logger.error("Timeout waiting for first real pose from /ekf_quat/ekf_odom")
        worker.shutdown()
        return
    if not vla_status_buffer.wait_for_ready(timeout=args.vla_ready_timeout_sec):
        logger.error("Timeout waiting for initial IDLE/DONE from %s", args.vla_status_topic)
        worker.shutdown()
        return

    init_pose_real = real_pose_buffer.get_latest_pose()
    logger.info(f"Initial Real Pose: {init_pose_real}")
    logger.info(
        "OpenPi continuous motion enabled: async_query_interval_steps=%s max_pending_queries=%s "
        "max_chunk_history=%s temporal_ensemble_window=%s temporal_ensemble_decay=%.3f user_confirm=%s",
        args.async_query_interval_steps,
        args.max_pending_queries,
        args.max_chunk_history,
        args.temporal_ensemble_window,
        args.temporal_ensemble_decay,
        not args.no_user_confirm,
    )

    try:
        while args.max_steps is None or step_count < args.max_steps:
            if rospy.is_shutdown():
                break

            print(color_text(f"\n{'='*60}\nLOOP ITERATION: {step_count}\n{'='*60}", ANSI_CYAN))

            image = ros_image_buffer.get_latest_image()
            if image is None:
                rospy.sleep(0.1)
                continue

            if run_dir:
                image.save(os.path.join(run_dir, f"step_{step_count:04d}.png"))

            curr_pose_real = real_pose_buffer.get_latest_pose()
            curr_proprio_local = transform_pose_to_init_frame(curr_pose_real, init_pose_real)
            logger.info(f"Current Proprio Local: {curr_proprio_local.tolist()}")

            query_events: List[Dict[str, Any]] = []
            query_events.extend(_drain_results(step_count, "loop_start"))
            chunk_records = prune_chunk_records(step_count, chunk_records)
            chunk_records = limit_chunk_records(chunk_records, args.max_chunk_history)

            if not chunk_records:
                if not pending_query_steps:
                    query_events.append(_submit_async_query(step_count, image, curr_proprio_local, "bootstrap"))
                logger.info("Waiting for bootstrap OpenPi chunk before first motion.")
                while not rospy.is_shutdown() and not chunk_records:
                    query_events.extend(_drain_results(step_count, "bootstrap_wait"))
                    if chunk_records:
                        break
                    if not pending_query_steps:
                        query_events.append(_submit_async_query(step_count, image, curr_proprio_local, "bootstrap_retry"))
                    rospy.sleep(0.02)
                if not chunk_records:
                    logger.error("No bootstrap chunk available; stopping.")
                    break

            should_submit_query = (
                step_count > 0
                and step_count % args.async_query_interval_steps == 0
                and len(pending_query_steps) < args.max_pending_queries
            )
            if should_submit_query:
                query_events.append(_submit_async_query(step_count, image, curr_proprio_local, "fixed_interval"))
            elif (
                step_count > 0
                and step_count % args.async_query_interval_steps == 0
                and len(pending_query_steps) >= args.max_pending_queries
            ):
                logger.info(
                    "Skipping async OpenPi query at step=%s because pending=%s >= max_pending_queries=%s",
                    step_count,
                    len(pending_query_steps),
                    args.max_pending_queries,
                )

            action_delta, blend_metadata = blend_chunk_record_predictions(
                step_count,
                chunk_records,
                args.temporal_ensemble_decay,
                args.temporal_ensemble_window,
            )
            if action_delta is None:
                logger.warning(
                    "No aligned action available at step=%s. Waiting for pending async inference.",
                    step_count,
                )
                wait_start = time.monotonic()
                while not rospy.is_shutdown() and action_delta is None:
                    query_events.extend(_drain_results(step_count, "no_action_wait"))
                    chunk_records = prune_chunk_records(step_count, chunk_records)
                    chunk_records = limit_chunk_records(chunk_records, args.max_chunk_history)
                    action_delta, blend_metadata = blend_chunk_record_predictions(
                        step_count,
                        chunk_records,
                        args.temporal_ensemble_decay,
                        args.temporal_ensemble_window,
                    )
                    if action_delta is not None:
                        break
                    if not pending_query_steps:
                        query_events.append(_submit_async_query(step_count, image, curr_proprio_local, "no_action_retry"))
                    if time.monotonic() - wait_start > args.vla_ready_timeout_sec:
                        logger.error("Timed out waiting for an aligned async action at step=%s.", step_count)
                        break
                    rospy.sleep(0.02)
                if action_delta is None:
                    break

            log_chunk_source_summary(step_count, blend_metadata)
            logger.info(f"Blended Action Delta (5D): {action_delta.tolist()}")

            target_proprio_local = np.array([
                curr_proprio_local[0] + action_delta[0],
                curr_proprio_local[1] + action_delta[1],
                curr_proprio_local[2] + action_delta[2],
                _normalize_yaw(curr_proprio_local[3] + action_delta[3]),
                curr_proprio_local[4] + action_delta[4],
            ], dtype=np.float32)

            predicted_world_pose = transform_pose_from_init_frame_to_world(
                target_proprio_local.tolist(), init_pose_real
            )
            predicted_world_pose[1] = -predicted_world_pose[1]
            predicted_world_pose[3] = -predicted_world_pose[3]

            logger.info(f"Target World Pose (Right-Hand): {predicted_world_pose.tolist()}")
            command_pos_err, command_yaw_err, command_pitch_err = compute_tracking_error(
                curr_pose_real,
                predicted_world_pose,
            )
            logger.info(
                "Command Delta vs Current Odom -> pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg "
                "(planner tolerance pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg)",
                command_pos_err,
                command_yaw_err,
                command_pitch_err,
                args.vla_position_tolerance,
                args.vla_yaw_tolerance_deg,
                args.vla_pitch_tolerance_deg,
            )
            if command_pos_err <= args.vla_position_tolerance and command_yaw_err <= args.vla_yaw_tolerance_deg:
                logger.warning(
                    "Target is already inside the planner reach tolerance before sending. "
                    "EGO may report DONE without visible UAV motion. Consider lowering "
                    "--vla_position_tolerance or increasing policy/action scale."
                )

            seq_id = step_count + 1
            if not vla_status_buffer.wait_for_ready(timeout=args.vla_ready_timeout_sec):
                logger.error("Planner is not ready to accept seq=%s within %.1f sec.", seq_id, args.vla_ready_timeout_sec)
                break

            publish_vla_waypoint(
                vla_waypoint_pub, seq_id, predicted_world_pose,
                args.vla_position_tolerance, args.vla_yaw_tolerance_deg, args.vla_pitch_tolerance_deg,
                args.vla_timeout_sec, args.drone_id, 1, args.vla_frame_id
            )

            status_msg = vla_status_buffer.wait_for_seq_result(seq_id, args.vla_timeout_sec + args.vla_ready_timeout_sec)
            if status_msg is None or int(status_msg.status) != VlaStatus.DONE:
                if status_msg is None:
                    logger.error("VLA execution seq=%s timed out.", seq_id)
                else:
                    logger.error(
                        "VLA execution seq=%s failed: status=%s message=%s",
                        seq_id,
                        int(status_msg.status),
                        getattr(status_msg, "message", ""),
                    )
                break

            reached, client_error = wait_for_client_reach(
                real_pose_buffer=real_pose_buffer,
                target_world_right_hand=predicted_world_pose,
                position_tolerance=args.client_position_tolerance,
                yaw_tolerance_deg=args.client_yaw_tolerance_deg,
                pitch_tolerance_deg=args.client_pitch_tolerance_deg,
                timeout=args.client_reach_timeout_sec,
            )
            logger.info(
                "Client Odom Reach Check -> reached=%s pos_err=%.3fm yaw_err=%.3fdeg pitch_err=%.3fdeg "
                "(client tolerance pos=%.3fm yaw=%.3fdeg pitch=%.3fdeg)",
                reached,
                client_error[0],
                client_error[1],
                client_error[2],
                args.client_position_tolerance,
                args.client_yaw_tolerance_deg,
                args.client_pitch_tolerance_deg,
            )
            if args.require_client_reach and not reached:
                logger.error(
                    "Planner reported DONE, but local odom did not reach the target within client tolerance. "
                    "This usually means the waypoint tolerance is still too loose or the controller did not execute motion."
                )
                break

            executed_pose_real = real_pose_buffer.get_latest_pose()
            executed_local = transform_pose_to_init_frame(executed_pose_real, init_pose_real)
            if args.small_motion_stop_steps > 0 and last_executed_local is not None:
                pos_diff = np.abs(executed_local[:3] - last_executed_local[:3])
                yaw_diff = abs(_normalize_yaw(float(executed_local[3]) - float(last_executed_local[3])))
                pitch_diff = abs(float(executed_local[4]) - float(last_executed_local[4]))
                if (
                    np.all(pos_diff < args.small_delta_pos_m)
                    and yaw_diff < args.small_delta_yaw_deg
                    and pitch_diff < args.small_delta_pitch_deg
                ):
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= args.small_motion_stop_steps:
                    logger.info(
                        "Detected small executed motion for %s consecutive steps; ending task.",
                        args.small_motion_stop_steps,
                    )
                    break
            last_executed_local = executed_local

            step_record = {
                "mode": "continuous_motion",
                "step": int(step_count),
                "seq_id": int(seq_id),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "instruction": args.instruction,
                "async_query_interval_steps": int(args.async_query_interval_steps),
                "max_pending_queries": int(args.max_pending_queries),
                "max_chunk_history": int(args.max_chunk_history),
                "temporal_ensemble_decay": float(args.temporal_ensemble_decay),
                "temporal_ensemble_window": int(args.temporal_ensemble_window),
                "pending_query_steps": {str(k): int(v) for k, v in pending_query_steps.items()},
                "curr_pose_real_left_hand": make_jsonable(curr_pose_real),
                "curr_proprio_local": curr_proprio_local.tolist(),
                "query_events": query_events,
                "active_chunk_history": [
                    {
                        "query_id": int(record.query_id),
                        "query_step": int(record.query_step),
                        "receive_step": int(record.receive_step),
                        "latency_steps": int(record.receive_step - record.query_step),
                        "chunk_shape": list(record.action_chunk.shape),
                        "current_chunk_index": int(step_count - record.query_step),
                    }
                    for record in chunk_records
                ],
                "blend": blend_metadata,
                "selected_action_delta": action_delta.tolist(),
                "target_proprio_local": target_proprio_local.tolist(),
                "predicted_world_pose_right_hand": predicted_world_pose.tolist(),
                "planner_done": True,
                "client_reached": bool(reached),
                "client_error": [float(v) for v in client_error],
                "executed_pose_real_left_hand": make_jsonable(executed_pose_real),
                "executed_proprio_local": executed_local.tolist(),
            }
            if run_dir:
                save_json(os.path.join(run_dir, f"step_{step_count:04d}_action_chunk.json"), step_record)
                append_jsonl(action_history_path, step_record)
                append_jsonl(
                    waypoint_history_path,
                    {
                        "mode": "continuous_motion",
                        "step": int(step_count),
                        "seq_id": int(seq_id),
                        "selected_action_delta": action_delta.tolist(),
                        "target_proprio_local": target_proprio_local.tolist(),
                        "predicted_world_pose_right_hand": predicted_world_pose.tolist(),
                    },
                )

            if not args.no_user_confirm:
                try:
                    cmd = input(color_text(f"[Step {step_count}] Reached target. Press Enter to continue, 'q' to quit: ", ANSI_MAGENTA))
                except EOFError:
                    logger.warning("stdin is closed; stopping before the next waypoint for safety.")
                    break
                if cmd.strip().lower() in {"q", "quit", "exit"}:
                    break

            step_count += 1
    finally:
        worker.shutdown()


def control_loop_trajectory_queue(
    args,
    ros_image_buffer,
    real_pose_buffer,
    vla_waypoint_queue_pub,
    vla_status_buffer,
    policy_client,
):
    """Run async inference and replaceable, globally aligned waypoint-queue execution."""
    if args.async_query_interval_steps < 1:
        raise ValueError("async_query_interval_steps must be >= 1")
    if args.temporal_ensemble_window < 1:
        raise ValueError("temporal_ensemble_window must be >= 1")
    if args.max_pending_queries < 1:
        raise ValueError("max_pending_queries must be >= 1")
    if args.max_chunk_history < 1:
        raise ValueError("max_chunk_history must be >= 1")
    if args.queue_horizon < 1:
        raise ValueError("queue_horizon must be >= 1")
    if args.trajectory_step_period_sec <= 0.0:
        raise ValueError("trajectory_step_period_sec must be > 0")

    step_count = 0
    next_query_id = 0
    next_seq_id = max(1, int(time.time() * 1000) & 0xFFFFFFFF)
    last_query_trigger_step: Optional[int] = None
    pending_query_steps: Dict[int, int] = {}
    chunk_records: List[ChunkRecord] = []
    active_seq_id: Optional[int] = None
    last_published_query_id = -1
    published_queue_end_step = -1
    next_tick_time: Optional[float] = None
    completion_deadline: Optional[float] = None
    stop_after_active_queue = False
    first_queue_confirmed = False
    small_count = 0
    last_tick_local: Optional[np.ndarray] = None
    worker = AsyncOpenPiWorker(policy_client)

    run_dir = getattr(args, "run_dir", None)
    action_history_path = os.path.join(run_dir, "action_chunk_history.jsonl") if run_dir else None
    waypoint_history_path = os.path.join(run_dir, "published_waypoint_queues.jsonl") if run_dir else None

    def _submit_async_query(image: Image.Image, proprio_local: np.ndarray, reason: str) -> Dict[str, Any]:
        nonlocal next_query_id, last_query_trigger_step
        request = create_inference_request(
            query_id=next_query_id,
            query_step=step_count,
            image=image,
            proprio_local=proprio_local,
            instruction=args.instruction,
        )
        next_query_id += 1
        pending_query_steps[request.query_id] = request.query_step
        worker.submit(request)
        last_query_trigger_step = step_count
        logger.info(
            "Submitted trajectory-queue OpenPi query id=%s query_step=%s reason=%s pending=%s",
            request.query_id,
            request.query_step,
            reason,
            len(pending_query_steps),
        )
        return {
            "event": "submitted",
            "reason": reason,
            "query_id": int(request.query_id),
            "query_step": int(request.query_step),
            "pending_queries": int(len(pending_query_steps)),
        }

    def _drain_results(reason: str) -> Tuple[List[Dict[str, Any]], List[int]]:
        nonlocal chunk_records
        events: List[Dict[str, Any]] = []
        accepted_query_ids: List[int] = []
        while True:
            try:
                result = worker.result_queue.get_nowait()
            except queue.Empty:
                break
            pending_query_steps.pop(result.query_id, None)
            record, event = append_inference_result_as_chunk(result, step_count, reason)
            events.append(event)
            if record is not None:
                chunk_records.append(record)
                accepted_query_ids.append(record.query_id)
            worker.result_queue.task_done()
        chunk_records = prune_chunk_records(step_count, chunk_records)
        chunk_records = limit_chunk_records(chunk_records, args.max_chunk_history)
        return events, accepted_query_ids

    logger.info("Waiting for first ROS image, pose, and Ego Planner status ...")
    if not ros_image_buffer.wait_for_first_image(timeout=10.0):
        logger.error("Timeout waiting for first ROS image from %s", ros_image_buffer.topic)
        worker.shutdown()
        return
    if not real_pose_buffer.wait_for_first_pose(timeout=10.0):
        logger.error("Timeout waiting for first real pose from /ekf_quat/ekf_odom")
        worker.shutdown()
        return
    if not vla_status_buffer.wait_for_ready(timeout=args.vla_ready_timeout_sec):
        logger.error("Timeout waiting for initial IDLE/DONE from %s", args.vla_status_topic)
        worker.shutdown()
        return

    init_pose_real = real_pose_buffer.get_latest_pose()
    logger.info(
        "OpenPi trajectory queue enabled: horizon=%s step_period=%.3fs async_interval=%s "
        "max_pending=%s max_history=%s ensemble_window=%s ensemble_decay=%.3f policy=%s semantics=%s",
        args.queue_horizon,
        args.trajectory_step_period_sec,
        args.async_query_interval_steps,
        args.max_pending_queries,
        args.max_chunk_history,
        args.temporal_ensemble_window,
        args.temporal_ensemble_decay,
        args.queue_busy_policy,
        args.queue_action_semantics,
    )

    try:
        while not rospy.is_shutdown():
            loop_now = time.monotonic()

            if next_tick_time is not None:
                ticked = False
                while (
                    loop_now >= next_tick_time
                    and not stop_after_active_queue
                    and (args.max_steps is None or step_count < args.max_steps)
                ):
                    step_count += 1
                    next_tick_time += args.trajectory_step_period_sec
                    ticked = True
                if ticked:
                    current_local = transform_pose_to_init_frame(
                        real_pose_buffer.get_latest_pose(), init_pose_real
                    )
                    if args.small_motion_stop_steps > 0 and last_tick_local is not None:
                        pos_diff = np.abs(current_local[:3] - last_tick_local[:3])
                        yaw_diff = abs(_normalize_yaw(float(current_local[3]) - float(last_tick_local[3])))
                        pitch_diff = abs(float(current_local[4]) - float(last_tick_local[4]))
                        if (
                            np.all(pos_diff < args.small_delta_pos_m)
                            and yaw_diff < args.small_delta_yaw_deg
                            and pitch_diff < args.small_delta_pitch_deg
                        ):
                            small_count += 1
                        else:
                            small_count = 0
                    last_tick_local = current_local
                    if (
                        args.small_motion_stop_steps > 0
                        and small_count >= args.small_motion_stop_steps
                        and not stop_after_active_queue
                    ):
                        stop_after_active_queue = True
                        logger.info(
                            "Detected small motion for %s trajectory ticks; stopping after the active queue.",
                            args.small_motion_stop_steps,
                        )
                    chunk_records = prune_chunk_records(step_count, chunk_records)
                    chunk_records = limit_chunk_records(chunk_records, args.max_chunk_history)

            query_events, accepted_query_ids = _drain_results("trajectory_loop")

            if active_seq_id is not None:
                terminal = vla_status_buffer.get_terminal_result(active_seq_id)
                if terminal is not None:
                    terminal_status = int(terminal.status)
                    if terminal_status == VlaStatus.DONE:
                        logger.info("Trajectory queue seq=%s completed.", active_seq_id)
                        active_seq_id = None
                    else:
                        logger.error(
                            "Active trajectory queue seq=%s failed: status=%s message=%s",
                            active_seq_id,
                            terminal_status,
                            getattr(terminal, "message", ""),
                        )
                        break

            reached_step_limit = (
                (args.max_steps is not None and step_count >= args.max_steps)
                or stop_after_active_queue
            )
            if reached_step_limit:
                if active_seq_id is None:
                    logger.info("Trajectory queue run reached its stop condition and is complete.")
                    break
                if completion_deadline is None:
                    completion_deadline = loop_now + args.vla_timeout_sec + args.vla_ready_timeout_sec
                    logger.info("Stop condition reached; waiting for final queue seq=%s.", active_seq_id)
                elif loop_now > completion_deadline:
                    logger.error("Timed out waiting for final trajectory queue seq=%s.", active_seq_id)
                    break
                rospy.sleep(0.02)
                continue

            image = ros_image_buffer.get_latest_image()
            if image is None:
                rospy.sleep(0.02)
                continue
            curr_pose_real = real_pose_buffer.get_latest_pose()
            curr_proprio_local = transform_pose_to_init_frame(curr_pose_real, init_pose_real)

            if not chunk_records and not pending_query_steps:
                query_events.append(_submit_async_query(image, curr_proprio_local, "bootstrap"))

            fixed_interval_due = (
                step_count > 0
                and step_count % args.async_query_interval_steps == 0
                and last_query_trigger_step != step_count
            )
            if fixed_interval_due and len(pending_query_steps) < args.max_pending_queries:
                query_events.append(_submit_async_query(image, curr_proprio_local, "fixed_interval"))
            elif fixed_interval_due:
                logger.info(
                    "Skipping fixed-interval query at step=%s: pending=%s max=%s",
                    step_count,
                    len(pending_query_steps),
                    args.max_pending_queries,
                )

            queue_exhausted = active_seq_id is not None and step_count >= published_queue_end_step
            should_publish = active_seq_id is None or queue_exhausted
            if accepted_query_ids and max(accepted_query_ids) > last_published_query_id:
                should_publish = True

            if should_publish and chunk_records:
                world_poses, point_metadata = build_aligned_waypoint_queue(
                    current_step=step_count,
                    queue_horizon=args.queue_horizon,
                    max_steps=args.max_steps,
                    chunk_records=chunk_records,
                    ensemble_decay=args.temporal_ensemble_decay,
                    ensemble_window=args.temporal_ensemble_window,
                    init_pose_real=init_pose_real,
                    current_proprio_local=curr_proprio_local,
                    action_semantics=args.queue_action_semantics,
                )
                if world_poses.shape[0] > 0:
                    primary_sources = point_metadata[0].get("candidates", [])
                    primary_query_id = int(primary_sources[0]["query_id"]) if primary_sources else -1
                    is_new_information = primary_query_id > last_published_query_id
                    if active_seq_id is None or queue_exhausted or is_new_information:
                        if not first_queue_confirmed and not args.no_user_confirm:
                            try:
                                cmd = input(
                                    color_text(
                                        f"Publish initial {world_poses.shape[0]}-point trajectory queue? "
                                        "Press Enter to continue, 'q' to quit: ",
                                        ANSI_MAGENTA,
                                    )
                                )
                            except EOFError:
                                logger.warning("stdin is closed; refusing initial queue publication for safety.")
                                break
                            if cmd.strip().lower() in {"q", "quit", "exit"}:
                                break
                            first_queue_confirmed = True

                        seq_id = next_seq_id
                        next_seq_id = (next_seq_id + 1) & 0xFFFFFFFF
                        if next_seq_id == 0:
                            next_seq_id = 1
                        source_metadata = publish_vla_waypoint_queue(
                            vla_waypoint_queue_pub,
                            seq_id,
                            world_poses,
                            point_metadata,
                            args.vla_position_tolerance,
                            args.vla_yaw_tolerance_deg,
                            args.vla_pitch_tolerance_deg,
                            args.vla_timeout_sec,
                            args.drone_id,
                            args.queue_busy_policy,
                            args.vla_frame_id,
                        )
                        active_seq_id = seq_id
                        last_published_query_id = max(last_published_query_id, source_metadata["query_id"])
                        published_queue_end_step = step_count + int(world_poses.shape[0])
                        if next_tick_time is None:
                            next_tick_time = time.monotonic() + args.trajectory_step_period_sec
                            last_tick_local = curr_proprio_local.copy()

                        queue_record = {
                            "mode": "trajectory_queue_motion",
                            "seq_id": int(seq_id),
                            "publish_step": int(step_count),
                            "queue_end_step": int(published_queue_end_step),
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "instruction": args.instruction,
                            "queue_action_semantics": args.queue_action_semantics,
                            "source": source_metadata,
                            "query_events": query_events,
                            "pending_query_steps": {str(k): int(v) for k, v in pending_query_steps.items()},
                            "world_poses_right_hand": world_poses.tolist(),
                            "point_alignment": point_metadata,
                        }
                        if run_dir:
                            save_json(os.path.join(run_dir, f"queue_{seq_id:04d}.json"), queue_record)
                            append_jsonl(action_history_path, queue_record)
                            append_jsonl(waypoint_history_path, queue_record)
                elif active_seq_id is None and not pending_query_steps:
                    query_events.append(_submit_async_query(image, curr_proprio_local, "queue_empty_retry"))

            if active_seq_id is None and not chunk_records and not pending_query_steps:
                query_events.append(_submit_async_query(image, curr_proprio_local, "no_aligned_action_retry"))

            rospy.sleep(0.02)
    finally:
        worker.shutdown()

def main():
    parser = argparse.ArgumentParser(description="OpenPi pi0.5 ROS real-drone client with client-side motion chunking")
    parser.add_argument("--server_ip", type=str, default=DEFAULT_SERVER_IP, help="OpenPi websocket policy server IP / hostname.")
    parser.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT, help="OpenPi websocket policy server port.")
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION, help="Language instruction sent to OpenPi.")
    parser.add_argument("--max_steps", type=int, default=80, help="Maximum number of Ego Planner waypoints to execute.")
    parser.add_argument(
        "--open_loop_horizon",
        type=int,
        default=DEFAULT_OPEN_LOOP_HORIZON,
        help=(
            "How many executed waypoint steps to consume before requesting a fresh OpenPi action chunk. "
            "Set to 1 for most conservative visual feedback, or 4/8 to consume more of each chunk open-loop."
        ),
    )
    parser.add_argument(
        "--temporal_ensemble_decay",
        type=float,
        default=DEFAULT_TEMPORAL_ENSEMBLE_DECAY,
        help="Exponential decay for blending overlapping chunk predictions; newer chunks get higher weight.",
    )
    parser.add_argument(
        "--temporal_ensemble_window",
        type=int,
        default=DEFAULT_TEMPORAL_ENSEMBLE_WINDOW,
        help="Maximum number of overlapping chunk rows to blend for the current logical step.",
    )
    parser.add_argument(
        "--continuous_motion",
        action="store_true",
        help="Run OpenPi inference in a background worker while executing aligned cached chunk actions.",
    )
    parser.add_argument(
        "--trajectory_queue_motion",
        action="store_true",
        help=(
            "Publish aligned future action rows as replaceable /vla_waypoint_queue trajectories; "
            "inference and planner-status handling remain non-blocking."
        ),
    )
    parser.add_argument(
        "--async_query_interval_steps",
        type=int,
        default=DEFAULT_ASYNCHRONOUS_QUERY_INTERVAL_STEPS,
        help=(
            "Continuous-motion mode only: launch a new async OpenPi query every N executed waypoint steps. "
            "This fixed-interval trigger is separated from future remaining-horizon prefetch logic."
        ),
    )
    parser.add_argument(
        "--max_pending_queries",
        type=int,
        default=DEFAULT_MAX_PENDING_QUERIES,
        help="Continuous-motion mode only: maximum in-flight async OpenPi queries.",
    )
    parser.add_argument(
        "--max_chunk_history",
        type=int,
        default=DEFAULT_MAX_CHUNK_HISTORY,
        help="Continuous-motion mode only: keep at most this many recent valid chunks for alignment/ensembling.",
    )
    parser.add_argument(
        "--queue_horizon",
        type=int,
        default=DEFAULT_QUEUE_HORIZON,
        help="Trajectory-queue mode only: maximum number of aligned future waypoints per publication.",
    )
    parser.add_argument(
        "--trajectory_step_period_sec",
        type=float,
        default=DEFAULT_TRAJECTORY_STEP_PERIOD_SEC,
        help=(
            "Trajectory-queue mode only: wall-clock duration of one logical action row. "
            "Set this to the action sampling period used by the deployed policy."
        ),
    )
    parser.add_argument(
        "--queue_busy_policy",
        choices=["accept_if_idle", "reject_if_busy", "replace_active"],
        default="replace_active",
        help="Trajectory-queue mode only: Ego Planner policy when a VLA task is already active.",
    )
    parser.add_argument(
        "--queue_action_semantics",
        choices=["incremental", "query_relative"],
        default="incremental",
        help=(
            "How action rows become queue targets. 'incremental' cumulatively applies aligned deltas "
            "from the current state (matching the legacy client); 'query_relative' anchors every row "
            "at its originating query proprio (matching the handoff plan)."
        ),
    )
    parser.add_argument("--ros_image_topic", type=str, default=DEFAULT_ROS_IMAGE_TOPIC, help="ROS image topic to subscribe.")
    parser.add_argument("--vla_waypoint_topic", type=str, default="/vla_waypoint", help="Ego Planner VLA waypoint topic.")
    parser.add_argument(
        "--vla_waypoint_queue_topic",
        type=str,
        default="/vla_waypoint_queue",
        help="Ego Planner VLA multi-waypoint queue topic.",
    )
    parser.add_argument("--vla_status_topic", type=str, default="/vla_status", help="Ego Planner VLA status topic.")
    parser.add_argument("--drone_id", type=int, default=0)
    parser.add_argument("--vla_frame_id", type=str, default="world")
    parser.add_argument("--vla_position_tolerance", type=float, default=DEFAULT_VLA_POSITION_TOLERANCE)
    parser.add_argument("--vla_yaw_tolerance_deg", type=float, default=8.0)
    parser.add_argument("--vla_pitch_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--vla_timeout_sec", type=float, default=60.0)
    parser.add_argument("--vla_ready_timeout_sec", type=float, default=60.0)
    parser.add_argument("--client_position_tolerance", type=float, default=DEFAULT_VLA_POSITION_TOLERANCE)
    parser.add_argument("--client_yaw_tolerance_deg", type=float, default=8.0)
    parser.add_argument("--client_pitch_tolerance_deg", type=float, default=5.0)
    parser.add_argument("--client_reach_timeout_sec", type=float, default=DEFAULT_CLIENT_REACH_TIMEOUT_SEC)
    parser.add_argument("--require_client_reach", dest="require_client_reach", action="store_true", default=True)
    parser.add_argument("--no_require_client_reach", dest="require_client_reach", action="store_false")
    parser.add_argument(
        "--no_user_confirm",
        action="store_true",
        help="Do not pause for keyboard confirmation after each executed waypoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for per-run images, chunk JSON records, and run_log.txt.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument(
        "--small_motion_stop_steps",
        type=int,
        default=DEFAULT_SMALL_MOTION_STOP_STEPS,
        help="Stop after this many consecutive small executed motions. 0 disables this safety heuristic.",
    )
    parser.add_argument("--small_delta_pos_m", type=float, default=DEFAULT_SMALL_DELTA_POS_M)
    parser.add_argument("--small_delta_yaw_deg", type=float, default=DEFAULT_SMALL_DELTA_YAW_DEG)
    parser.add_argument("--small_delta_pitch_deg", type=float, default=DEFAULT_SMALL_DELTA_PITCH_DEG)
    args = parser.parse_args()

    if args.continuous_motion and args.trajectory_queue_motion:
        parser.error("--continuous_motion and --trajectory_queue_motion are mutually exclusive")

    run_dir = None
    if args.output_dir:
        output_root = os.path.abspath(os.path.expanduser(args.output_dir))
        run_dir = os.path.join(output_root, datetime.now().strftime("pi05_chunking_%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
    args.run_dir = run_dir

    configure_logger(args.log_level, os.path.join(run_dir, "run_log.txt") if run_dir else None)

    if VlaStatus is None:
        logger.error("Failed to import vla_client/VlaStatus. Please source this catkin workspace setup file.")
        return
    if args.trajectory_queue_motion and VlaWaypointQueue is None:
        logger.error(
            "Failed to import vla_client/VlaWaypointQueue. Rebuild and source the workspace containing the new message."
        )
        return
    if not args.trajectory_queue_motion and VlaWaypoint is None:
        logger.error("Failed to import vla_client/VlaWaypoint. Please source this catkin workspace setup file.")
        return
    if run_dir:
        logger.info("Saving OpenPi chunking debug records to %s", run_dir)

    rospy.init_node("pi05_real_drone_client", anonymous=True)
    
    ros_image_buffer = RosImageBuffer(args.ros_image_topic)
    real_pose_buffer = RealPoseBuffer()
    vla_status_buffer = VlaStatusBuffer(args.vla_status_topic)
    vla_waypoint_pub = None
    vla_waypoint_queue_pub = None
    if args.trajectory_queue_motion:
        vla_waypoint_queue_pub = rospy.Publisher(
            args.vla_waypoint_queue_topic, VlaWaypointQueue, queue_size=1
        )
        logger.info("Publishing trajectory queues to ROS topic: %s", args.vla_waypoint_queue_topic)
    else:
        vla_waypoint_pub = rospy.Publisher(args.vla_waypoint_topic, VlaWaypoint, queue_size=1)

    logger.info(f"Connecting to pi05 Server at ws://{args.server_ip}:{args.server_port}")
    policy_client = get_policy_client(args.server_ip, args.server_port)

    if args.trajectory_queue_motion:
        control_loop_trajectory_queue(
            args,
            ros_image_buffer,
            real_pose_buffer,
            vla_waypoint_queue_pub,
            vla_status_buffer,
            policy_client,
        )
    elif args.continuous_motion:
        control_loop_continuous_motion(
            args,
            ros_image_buffer,
            real_pose_buffer,
            vla_waypoint_pub,
            vla_status_buffer,
            policy_client,
        )
    else:
        control_loop(args, ros_image_buffer, real_pose_buffer, vla_waypoint_pub, vla_status_buffer, policy_client)

if __name__ == "__main__":
    main()

