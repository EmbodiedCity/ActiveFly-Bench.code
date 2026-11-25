import argparse
import glob
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import parse as urlparse

import airsim
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENPI_CLIENT_SRC = REPO_ROOT / "run" / "packages" / "openpi-client" / "src"
if OPENPI_CLIENT_SRC.exists():
    sys.path.insert(0, str(OPENPI_CLIENT_SRC))

try:
    from openpi_client import image_tools as openpi_image_tools
    from openpi_client import websocket_client_policy
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        f"Failed to import openpi_client dependency '{exc.name}'. "
        "Run this script from the UE/AirSim environment and install the missing "
        "package(s) there."
    ) from exc

logger = logging.getLogger(__name__)

IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_DELTA_PITCH: float = 1.0
ACTION_SMALL_STEPS: int = 5
ACTION_SCALE: float = 0.01
ACTIVEFLY_STATE_DIM: int = 7
ACTIVEFLY_DATASET_FPS: float = 10.0
DEFAULT_OPEN_LOOP_HORIZON: int = 2
DEFAULT_TEMPORAL_ENSEMBLE_DECAY: float = 0.5
DEFAULT_TEMPORAL_ENSEMBLE_WINDOW: int = 4

# Activefly deployment contract in this script:
# - observation/state is 7D: [x, y, z, roll, pitch, yaw, gimbal_pitch].
#   This script only actively controls 5 DoF in the loop: [x, y, z, yaw, gimbal_pitch].
#   Roll and body pitch are therefore kept at 0.0 when packing the 7D state for the policy.
# - action is 5D delta: [dx, dy, dz, dyaw_deg, dgimbal_pitch_deg].
#
# The four deployment fixes below are intentional and should not be "simplified" away:
# 1) Do not feed a 5D action vector into the 7D observation/state slot.
# 2) Activefly angle labels are already in degrees, so do not call math.degrees() on policy output.
# 3) Policy output is a local delta relative to the current pose, not an absolute world pose.
# 4) The policy returns action chunks, so we consume them open-loop and blend overlaps to reduce chunk-boundary jumps.

_POLICY_CLIENTS: Dict[str, websocket_client_policy.WebsocketClientPolicy] = {}


def get_policy_client(server_url: str) -> websocket_client_policy.WebsocketClientPolicy:
    policy_client = _POLICY_CLIENTS.get(server_url)
    if policy_client is not None:
        return policy_client

    parsed = urlparse.urlparse(server_url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Invalid server url for openpi websocket client: {server_url}")

    policy_client = websocket_client_policy.WebsocketClientPolicy(host=parsed.hostname, port=parsed.port)
    _POLICY_CLIENTS[server_url] = policy_client
    return policy_client


def send_prediction_request(
    image: Image.Image,
    proprio: List[float],
    instr: str,
    server_url: str,
) -> Optional[Dict[str, Any]]:
    # Fix 1: the policy was trained with a 7D observation/state schema. Sending a 5D action-like vector here
    # will misalign the learned features immediately.
    if len(proprio) != ACTIVEFLY_STATE_DIM:
        raise ValueError(
            f"Activefly policy expects observation/state dim={ACTIVEFLY_STATE_DIM}, got {len(proprio)}."
        )
    image_np = np.asarray(image)
    image_np = openpi_image_tools.resize_with_pad(image_np, *IMG_INPUT_SIZE)
    image_np = openpi_image_tools.convert_to_uint8(image_np)
    observation: Dict[str, Any] = {
        "observation/image": image_np,
        "observation/state": np.asarray(proprio, dtype=np.float32),
        "prompt": instr,
    }
    try:
        policy_client = get_policy_client(server_url)
        response = policy_client.infer(observation)
        return {
            "action": response.get("actions"),
            "policy_timing": response.get("policy_timing"),
            "server_timing": response.get("server_timing"),
        }
    except Exception as exc:
        logger.error(f"Request failed: {exc}")
        return None


def reset_model(server_url: str) -> None:
    try:
        policy_client = get_policy_client(server_url)
        policy_client.reset()
        logger.info("Policy reset completed.")
    except Exception as exc:
        logger.error(f"Model reset failed: {exc}")


def normalize_angle_deg(angle: float) -> float:
    angle = angle % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


def set_vehicle_pose(
    client: airsim.VehicleClient,
    pos_xyz: List[float],
    yaw_deg: float,
) -> None:
    pose = airsim.Pose(
        airsim.Vector3r(pos_xyz[0], pos_xyz[1], pos_xyz[2]),
        airsim.to_quaternion(0.0, 0.0, math.radians(yaw_deg)),
    )
    client.simSetVehiclePose(pose, ignore_collision=True)


def world_to_local_pose(
    pose_world: List[float],
    init_pose_world: List[float],
    init_yaw_deg: float,
    init_pitch_deg: float,
) -> List[float]:
    dx = pose_world[0] - init_pose_world[0]
    dy = pose_world[1] - init_pose_world[1]
    dz = pose_world[2] - init_pose_world[2]
    theta = -math.radians(init_yaw_deg)
    x_rel = dx * math.cos(theta) - dy * math.sin(theta)
    y_rel = dx * math.sin(theta) + dy * math.cos(theta)
    yaw_rel = normalize_angle_deg(pose_world[3] - init_yaw_deg)
    pitch_rel = normalize_angle_deg(pose_world[4] - init_pitch_deg)
    return [float(x_rel), float(y_rel), float(dz), float(yaw_rel), float(pitch_rel)]


def local_delta_to_world(dx: float, dy: float, yaw0_deg: float) -> Tuple[float, float]:
    theta = math.radians(yaw0_deg)
    wx = dx * math.cos(theta) - dy * math.sin(theta)
    wy = dx * math.sin(theta) + dy * math.cos(theta)
    return wx, wy


def angular_distance_deg(angle_a: float, angle_b: float) -> float:
    return abs(normalize_angle_deg(angle_a - angle_b))


def make_activefly_state(local_pose: List[float]) -> List[float]:
    """Pack the 5-DoF local control state into Activefly's 7D state schema.

    Activefly uses [x, y, z, roll, pitch, yaw, gimbal_pitch].
    In this UE5 deployment loop we only control [x, y, z, yaw, gimbal_pitch], so roll/body-pitch stay 0.0.
    This is a schema adapter, not a new physical state estimator.
    """
    return [
        float(local_pose[0]),
        float(local_pose[1]),
        float(local_pose[2]),
        0.0,
        0.0,
        float(local_pose[3]),
        float(local_pose[4]),
    ]


def apply_action_delta(local_pose: List[float], action_delta: np.ndarray) -> List[float]:
    # Fix 2 + 3: Activefly action angles are already in degrees, and the 5D action is a delta relative to the
    # current local pose. We add it directly instead of converting units or treating it as an absolute pose.
    dx, dy, dz, dyaw_deg, dpitch_deg = [float(v) for v in action_delta[:5]]
    return [
        local_pose[0] + dx,
        local_pose[1] + dy,
        local_pose[2] + dz,
        normalize_angle_deg(local_pose[3] + dyaw_deg),
        normalize_angle_deg(local_pose[4] + dpitch_deg),
    ]


def local_pose_to_world_pose(
    local_pose: List[float],
    init_pose_world: List[float],
    init_yaw_deg: float,
    init_pitch_deg: float,
    pos_scale: float,
) -> Tuple[List[float], float, float]:
    global_x, global_y = local_delta_to_world(local_pose[0], local_pose[1], init_yaw_deg)
    pos_world = [
        init_pose_world[0] + global_x * pos_scale,
        init_pose_world[1] + global_y * pos_scale,
        init_pose_world[2] - local_pose[2] * pos_scale,
    ]
    global_yaw = normalize_angle_deg(init_yaw_deg + local_pose[3])
    global_pitch = normalize_angle_deg(init_pitch_deg + local_pose[4])
    return pos_world, global_yaw, global_pitch


def validate_action_chunk(action_poses: Any) -> Optional[np.ndarray]:
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


def blend_chunk_predictions(
    step_count: int,
    action_chunk_history: List[Tuple[int, np.ndarray]],
    ensemble_decay: float,
    ensemble_window: int,
) -> Optional[np.ndarray]:
    candidates: List[np.ndarray] = []
    for query_step, action_chunk in reversed(action_chunk_history):
        chunk_index = step_count - query_step
        if 0 <= chunk_index < action_chunk.shape[0]:
            candidates.append(action_chunk[chunk_index])
            if len(candidates) >= ensemble_window:
                break

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0].astype(np.float32, copy=False)

    decay = min(max(float(ensemble_decay), 0.0), 1.0)
    weights = np.asarray([decay**i for i in range(len(candidates))], dtype=np.float32)
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(candidates), dtype=np.float32)
    weights /= weights.sum()
    return np.average(np.stack(candidates, axis=0), axis=0, weights=weights).astype(np.float32)


def get_image_rgb(
    client: airsim.VehicleClient,
    camera_name: str,
    image_type: airsim.ImageType,
) -> Optional[np.ndarray]:
    responses = client.simGetImages(
        [airsim.ImageRequest(camera_name, image_type, False, False)],
    )
    if not responses:
        return None
    resp = responses[0]
    if resp.width == 0 or resp.height == 0 or not resp.image_data_uint8:
        return None
    img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
    channels = int(len(img1d) / (resp.width * resp.height))
    if channels < 3:
        return None
    img = img1d.reshape(resp.height, resp.width, channels)
    return img[:, :, :3][:, :, ::-1]


def set_camera_pitch(client: airsim.VehicleClient, camera_name: str, pitch_deg: float) -> None:
    cam_pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        airsim.to_quaternion(math.radians(pitch_deg), 0.0, 0.0),
    )
    client.simSetCameraPose(camera_name, cam_pose)


def create_obj_if_needed(client: airsim.VehicleClient, obj_info: Optional[Dict[str, Any]]) -> None:
    if obj_info is None:
        return
    use_obj = obj_info.get("use_obj", None)
    obj_pos = obj_info.get("obj_pos", None)
    obj_rot = obj_info.get("obj_rot", None)
    try:
        if use_obj == 1 and obj_pos is not None:
            pose = airsim.Pose(
                airsim.Vector3r(obj_pos[0], obj_pos[1], obj_pos[2]),
                airsim.to_quaternion(
                    math.radians(obj_rot[0]), math.radians(obj_rot[1]), math.radians(obj_rot[2])
                ),
            )
            client.simSetObjectPose("BP_Character_21", pose, teleport=True)
        elif use_obj == 2 and obj_pos is not None:
            pose = airsim.Pose(
                airsim.Vector3r(obj_pos[0], obj_pos[1], 0.0),
                airsim.to_quaternion(
                    math.radians(obj_rot[0]), math.radians(obj_rot[1]), math.radians(obj_rot[2])
                ),
            )
            client.simSetObjectPose("BP_Character_22", pose, teleport=True)
    except Exception as exc:
        logger.warning(f"Failed to set object pose: {exc}")
    time.sleep(SLEEP_SHORT_S)


def control_loop_vla(
    client: airsim.VehicleClient,
    init_pose_world: List[float],
    init_yaw_deg: float,
    init_pitch_deg: float,
    camera_name: str,
    image_type: airsim.ImageType,
    pos_scale: float,
    instruction: str,
    server_url: str,
    max_steps: int,
    frame_dir: str,
    frame_sleep: float,
    open_loop_horizon: int,
    temporal_ensemble_decay: float,
    temporal_ensemble_window: int,
    trajectory_log: List[Dict[str, Any]],
) -> bool:
    if open_loop_horizon < 1:
        raise ValueError("open_loop_horizon must be >= 1")
    if temporal_ensemble_window < 1:
        raise ValueError("temporal_ensemble_window must be >= 1")

    os.makedirs(frame_dir, exist_ok=True)
    policy_local_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_local_pose_world: Optional[List[float]] = None
    small_count = 0
    step_count = 0
    action_chunk_history: List[Tuple[int, np.ndarray]] = []
    logged_chunk_shape = False

    while step_count < max_steps:
        image = get_image_rgb(client, camera_name, image_type)
        if image is None:
            logger.warning("Empty image; skipping step.")
            time.sleep(frame_sleep)
            continue

        should_query_policy = not action_chunk_history or step_count % open_loop_horizon == 0
        if should_query_policy:
            response = send_prediction_request(
                image=Image.fromarray(image),
                proprio=make_activefly_state(policy_local_pose),
                instr=instruction,
                server_url=server_url,
            )
            if response is None:
                logger.warning("No valid response, ending control loop.")
                return False

            action_chunk = validate_action_chunk(response.get("action"))
            if action_chunk is None:
                logger.warning("Response 'action' is empty or invalid, stopping.")
                return False

            action_chunk_history.append((step_count, action_chunk))
            action_chunk_history = [
                (query_step, chunk)
                for query_step, chunk in action_chunk_history
                if query_step + chunk.shape[0] > step_count
            ]
            if not logged_chunk_shape:
                logger.info(f"Policy returned action chunk shape: {action_chunk.shape}")
                logged_chunk_shape = True
            if action_chunk.shape[0] < open_loop_horizon:
                logger.warning(
                    "Action chunk length %s is smaller than open_loop_horizon=%s; "
                    "the loop will fall back to querying early when needed.",
                    action_chunk.shape[0],
                    open_loop_horizon,
                )

        # Fix 4: the policy predicts an action chunk, not a single perfectly smooth command stream.
        # We therefore consume it in open loop and blend overlapping chunk predictions at the same logical step
        # to reduce hard switches at chunk boundaries.
        action_delta = blend_chunk_predictions(
            step_count,
            action_chunk_history,
            temporal_ensemble_decay,
            temporal_ensemble_window,
        )
        if action_delta is None:
            response = send_prediction_request(
                image=Image.fromarray(image),
                proprio=make_activefly_state(policy_local_pose),
                instr=instruction,
                server_url=server_url,
            )
            if response is None:
                logger.warning("No valid response for forced chunk refresh, ending control loop.")
                return False

            action_chunk = validate_action_chunk(response.get("action"))
            if action_chunk is None:
                logger.warning("Forced chunk refresh returned an invalid action chunk, stopping.")
                return False

            action_chunk_history.append((step_count, action_chunk))
            action_chunk_history = [
                (query_step, chunk)
                for query_step, chunk in action_chunk_history
                if query_step + chunk.shape[0] > step_count
            ]
            action_delta = blend_chunk_predictions(
                step_count,
                action_chunk_history,
                temporal_ensemble_decay,
                temporal_ensemble_window,
            )
            if action_delta is None:
                logger.warning("No usable action after forced chunk refresh, stopping.")
                return False

        policy_local_pose = apply_action_delta(policy_local_pose, action_delta)
        pos_world, global_yaw, global_pit = local_pose_to_world_pose(
            policy_local_pose,
            init_pose_world,
            init_yaw_deg,
            init_pitch_deg,
            pos_scale,
        )
        set_vehicle_pose(client, pos_world, global_yaw)
        set_camera_pitch(client, camera_name, global_pit)
        time.sleep(max(frame_sleep, 0.0))

        pose_world = [pos_world[0], pos_world[1], pos_world[2], global_yaw, global_pit]
        local_pose_world = world_to_local_pose(pose_world, init_pose_world, init_yaw_deg, init_pitch_deg)
        trajectory_log.append({"state": [local_pose_world[:3], [local_pose_world[4], local_pose_world[3], 0]]})
        step_count += 1

        image_after = get_image_rgb(client, camera_name, image_type)
        if image_after is not None:
            img_path = os.path.join(frame_dir, f"{step_count:06d}.png")
            Image.fromarray(image_after).save(img_path)

        if last_local_pose_world is not None:
            diffs_pos = [abs(a - b) for a, b in zip(local_pose_world[:3], last_local_pose_world[:3])]
            yaw_diff = angular_distance_deg(local_pose_world[3], last_local_pose_world[3])
            pitch_diff = angular_distance_deg(local_pose_world[4], last_local_pose_world[4])
            if (
                all(d < ACTION_SMALL_DELTA_POS * ACTION_SCALE for d in diffs_pos)
                and yaw_diff < ACTION_SMALL_DELTA_YAW
                and pitch_diff < ACTION_SMALL_DELTA_PITCH
            ):
                small_count += 1
            else:
                small_count = 0
            if small_count >= ACTION_SMALL_STEPS:
                logger.info(f"Detected small change for {ACTION_SMALL_STEPS} steps, ending task.")
                return True
        last_local_pose_world = local_pose_world
    return True


def ensure_no_proxy(hosts: Optional[List[str]] = None) -> None:
    no_proxy_hosts = ["127.0.0.1", "localhost"]
    if hosts:
        no_proxy_hosts.extend([host for host in hosts if host])
    no_proxy_value = ",".join(dict.fromkeys(no_proxy_hosts))
    os.environ["no_proxy"] = no_proxy_value
    os.environ["NO_PROXY"] = no_proxy_value


if __name__ == "__main__":
    ACTIVEFLY_ROOT = REPO_ROOT.parent
    DEFAULT_DATA_ROOT = Path(
        os.environ.get("ACTIVEFLY_FLUC_SIM_DIR")
        or (ACTIVEFLY_ROOT / "datasets")
    )
    DEFAULT_LOG_ROOT = [
        str(DEFAULT_DATA_ROOT / "ActiveFly_Test"),
    ]
    DEFAULT_IMAGES_DIR = str(DEFAULT_DATA_ROOT / "outputs" / "openpi" / "pi05_activefly_eval")
    DEFAULT_SERVER_IP = "127.0.0.1"
    DEFAULT_SERVER_PORT = 8000
    DEFAULT_MAX_STEPS = 80
    DEFAULT_POS_SCALE = 1
    DEFAULT_FRAME_SLEEP = 1.0 / ACTIVEFLY_DATASET_FPS

    parser = argparse.ArgumentParser(description="Batch run UAV control in UE5 via AirSim without plotting.")
    parser.add_argument("-r", "--log_root", action='append', default=None, help="Root dir containing log.json files")
    parser.add_argument("-o", "--images_dir", default=DEFAULT_IMAGES_DIR, help="Directory to save outputs")
    parser.add_argument("--require_complete", action="store_true",
                        help="Fail if the policy stops returning usable actions during the rollout.")
    parser.add_argument("-i", "--instruction_type", default="instruction", choices=["instruction", "instruction_unified"])
    parser.add_argument("--server_ip", default=DEFAULT_SERVER_IP, help="OpenPI policy server IP / hostname")
    parser.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT, help="OpenPI policy server port")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per task")
    parser.add_argument("--camera_name", default="0", help="AirSim camera name")
    parser.add_argument("--image_type", default="Scene", help="AirSim ImageType, e.g. Scene")
    parser.add_argument("--pos_scale", type=float, default=DEFAULT_POS_SCALE, help="Scale json positions into AirSim meters")
    parser.add_argument(
        "--frame_sleep",
        type=float,
        default=DEFAULT_FRAME_SLEEP,
        help="Sleep seconds per executed action; default matches the activefly dataset FPS (10 Hz).",
    )
    parser.add_argument(
        "--open_loop_horizon",
        type=int,
        default=DEFAULT_OPEN_LOOP_HORIZON,
        help="How many executed actions to consume before requesting a fresh action chunk.",
    )
    parser.add_argument(
        "--temporal_ensemble_decay",
        type=float,
        default=DEFAULT_TEMPORAL_ENSEMBLE_DECAY,
        help="Exponential decay for blending overlapping action chunks. Newer chunk predictions get higher weight.",
    )
    parser.add_argument(
        "--temporal_ensemble_window",
        type=int,
        default=DEFAULT_TEMPORAL_ENSEMBLE_WINDOW,
        help="Maximum number of overlapping chunk predictions to blend for each executed action.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    if args.log_root is None:
        args.log_root = DEFAULT_LOG_ROOT

    ensure_no_proxy([args.server_ip])

    client = airsim.VehicleClient()
    client.confirmConnection()

    image_type = getattr(airsim.ImageType, args.image_type, airsim.ImageType.Scene)
    server_url = f"http://{args.server_ip}:{args.server_port}/predict"
    logger.info(f"Connected to OpenPI policy server metadata: {get_policy_client(server_url).get_server_metadata()}")

    log_files = []
    for log_root in args.log_root:
        log_files.extend(sorted(glob.glob(os.path.join(log_root, "**", "log.json"), recursive=True)))
    log_files = [path for path in log_files if "(copy)" not in path]
    logger.info(f"Detected {len(log_files)} log.json files")
    os.makedirs(args.images_dir, exist_ok=True)

    for idx, log_file in enumerate(log_files):
        base_name = os.path.basename(os.path.dirname(log_file))
        out_dir = os.path.join(args.images_dir, base_name)
        frame_dir = os.path.join(out_dir, "frames")
        traj_json_path = os.path.join(out_dir, base_name + ".json")
        if os.path.exists(traj_json_path):
            logger.info(f"{base_name} json already exists, skipping.")
            continue

        logger.info(f"\n===== Start task {idx+1}/{len(log_files)} file: {log_file} =====")
        reset_model(server_url)
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if not isinstance(log_data, dict):
            logger.warning(f"Unsupported json format, skipping: {log_file}")
            continue

        if args.instruction_type == "instruction_unified":
            instruction = log_data.get("instruction_unified", log_data.get("instruction", ""))
        else:
            instruction = log_data.get("instruction", log_data.get("instruction_unified", ""))

        raw_logs = log_data.get("raw_logs", None)
        if not raw_logs:
            logger.error("Missing 'raw_logs' in log.json; skipping.")
            continue

        init_raw = raw_logs[0]
        if len(init_raw) < 7:
            logger.error("Invalid first raw_logs pose; skipping.")
            continue

        init_world = [
            float(init_raw[0]) * args.pos_scale,
            float(init_raw[1]) * args.pos_scale,
            float(init_raw[2]) * args.pos_scale,
            float(init_raw[5]),
            float(init_raw[6]),
        ]
        set_vehicle_pose(client, init_world[:3], init_world[3])
        set_camera_pitch(client, args.camera_name, float(init_world[4]))
        time.sleep(SLEEP_SHORT_S)

        obj_info = None
        if "obj_id" in log_data and "use_obj" in log_data:
            if "target_pos" in log_data and isinstance(log_data["target_pos"], list) and len(log_data["target_pos"]) == 6:
                obj_pos = [v * args.pos_scale for v in log_data["target_pos"][:3]]
                obj_rot = log_data["target_pos"][3:]
            else:
                obj_pos = log_data.get("obj_pos", None)
                if obj_pos is not None:
                    obj_pos = [v * args.pos_scale for v in obj_pos]
                obj_rot = log_data.get("obj_rot", [0, 0, 0])
            if obj_pos is not None:
                obj_info = {
                    "use_obj": log_data["use_obj"],
                    "obj_id": log_data["obj_id"],
                    "obj_pos": obj_pos,
                    "obj_rot": obj_rot,
                }
        create_obj_if_needed(client, obj_info)

        logger.info(f"instruction: {instruction}")
        trajectory_log: List[Dict[str, Any]] = []
        completed = control_loop_vla(
            client=client,
            trajectory_log=trajectory_log,
            init_pose_world=init_world,
            init_yaw_deg=init_world[3],
            init_pitch_deg=init_world[4],
            camera_name=args.camera_name,
            image_type=image_type,
            pos_scale=args.pos_scale,
            instruction=instruction,
            server_url=server_url,
            max_steps=args.max_steps,
            frame_dir=frame_dir,
            frame_sleep=args.frame_sleep,
            open_loop_horizon=args.open_loop_horizon,
            temporal_ensemble_decay=args.temporal_ensemble_decay,
            temporal_ensemble_window=args.temporal_ensemble_window,
        )
        if args.require_complete and not completed:
            raise RuntimeError(f"Policy interrupted rollout for {base_name}")

        os.makedirs(out_dir, exist_ok=True)
        with open(traj_json_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_log, f, indent=4)

        logger.info(f"Trajectory json saved: {traj_json_path}")
        logger.info(f"===== Task {idx+1} finished =====\n")
