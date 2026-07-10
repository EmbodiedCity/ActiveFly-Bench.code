import argparse
import base64
import glob
import json
import logging
import math
import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import airsim
import numpy as np
import requests
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


logger = logging.getLogger(__name__)

# ====== Constants ======
IMG_INPUT_SIZE: Tuple[int, int] = (224, 224)
SLEEP_SHORT_S: float = 1.0
ACTION_SMALL_DELTA_POS: float = 3.0
ACTION_SMALL_DELTA_YAW: float = 1.0
ACTION_SMALL_DELTA_PITCH: float = 1.0
ACTION_SMALL_STEPS: int = 5
# ACTION_CM_TO_M: float = 0.01
ACTION_SCALE: float = 0.01

TRAJ_IMG_SIZE_2D: Tuple[int, int] = (10, 10)
TRAJ_IMG_SIZE_3D: Tuple[int, int] = (12, 10)
PLOT_YAW_ARROW_MIN_LEN_2D: int = 10
PLOT_YAW_ARROW_MIN_LEN_3D: int = 12


def draw_2d_trajectory_from_log(
    log_path: str,
    save_path: str,
    instruction: str,
    target: Optional[List[float]],
    tick_x: Optional[float] = None,
    tick_y: Optional[float] = None,
    min_span_x: Optional[float] = None,
    min_span_y: Optional[float] = None,
) -> None:
    with open(log_path, "r", encoding="utf-8") as file_obj:
        log = json.load(file_obj)
    if not isinstance(log, list) or len(log) == 0:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + "\n" + log_path + " (no data)")
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        ax = plt.gca()
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return

    trajectory = np.array([item["state"][0] + item["state"][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        plt.figure(figsize=TRAJ_IMG_SIZE_2D)
        plt.title(instruction + "\n" + log_path + " (invalid data)")
        plt.xlabel("Y (right)")
        plt.ylabel("X (forward)")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        return

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_2D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length

    plt.figure(figsize=TRAJ_IMG_SIZE_2D)
    if target is not None:
        plt.scatter(target[1], target[0], color="red", label="target")
    plt.plot(y, x, color="blue", label="trajectory")
    plt.quiver(
        y,
        x,
        dy,
        dx,
        angles="xy",
        scale_units="xy",
        scale=1,
        color="green",
        width=0.003,
        label="yaw",
    )

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    cur_xmin, cur_xmax = ax.get_xlim()
    cur_ymin, cur_ymax = ax.get_ylim()
    span_y_axis = cur_xmax - cur_xmin
    span_x_axis = cur_ymax - cur_ymin
    if min_span_y is not None and min_span_y > 0 and span_y_axis < min_span_y:
        center = (cur_xmin + cur_xmax) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(center - half, center + half)
    if min_span_x is not None and min_span_x > 0 and span_x_axis < min_span_x:
        center = (cur_ymin + cur_ymax) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(center - half, center + half)

    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))

    plt.legend()
    plt.title(instruction + "\n" + log_path)
    plt.xlabel("Y (right)")
    plt.ylabel("X (forward)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def draw_3d_trajectory_from_log(
    log_path: str,
    save_path: str,
    instruction: str,
    target: Optional[List[float]],
    tick_x: Optional[float] = None,
    tick_y: Optional[float] = None,
    tick_z: Optional[float] = None,
    min_span_x: Optional[float] = None,
    min_span_y: Optional[float] = None,
    min_span_z: Optional[float] = None,
) -> None:
    with open(log_path, "r", encoding="utf-8") as file_obj:
        log = json.load(file_obj)
    if not isinstance(log, list) or len(log) == 0:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (no data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        if tick_y is not None and tick_y > 0:
            ax.xaxis.set_major_locator(MultipleLocator(tick_y))
        if tick_x is not None and tick_x > 0:
            ax.yaxis.set_major_locator(MultipleLocator(tick_x))
        if tick_z is not None and tick_z > 0:
            ax.zaxis.set_major_locator(MultipleLocator(tick_z))
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return

    trajectory = np.array([item["state"][0] + item["state"][1] for item in log])
    if trajectory.ndim != 2 or trajectory.shape[1] < 5:
        fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction} (invalid data)")
        ax.set_xlabel("Y (right)")
        ax.set_ylabel("X (forward)")
        ax.set_zlabel("Z (up)")
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    yaw = np.deg2rad(trajectory[:, 4])
    arrow_length = max(PLOT_YAW_ARROW_MIN_LEN_3D, np.sqrt(x.var() + y.var()) // 4)
    dx = np.cos(yaw) * arrow_length
    dy = np.sin(yaw) * arrow_length
    dz = np.zeros_like(dx)

    fig = plt.figure(figsize=TRAJ_IMG_SIZE_3D)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(y, x, z, color="blue", label="Trajectory")
    ax.quiver(y, x, z, dy, dx, dz, color="green", length=1.0, normalize=False, linewidth=0.5, label="Yaw direction")
    ax.scatter(y[0], x[0], z[0], color="blue", s=50)
    ax.text(y[0], x[0], z[0], "Start", color="blue", fontsize=12)
    if target is not None:
        ax.scatter(target[1], target[0], target[2], color="red", s=50)
        ax.text(target[1], target[0], target[2], "Target", color="red", fontsize=12)

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    cur_zlim = ax.get_zlim()
    span_y = cur_xlim[1] - cur_xlim[0]
    span_x = cur_ylim[1] - cur_ylim[0]
    span_z = cur_zlim[1] - cur_zlim[0]
    if min_span_y is not None and min_span_y > 0 and span_y < min_span_y:
        center = (cur_xlim[0] + cur_xlim[1]) / 2.0
        half = min_span_y / 2.0
        ax.set_xlim(center - half, center + half)
    if min_span_x is not None and min_span_x > 0 and span_x < min_span_x:
        center = (cur_ylim[0] + cur_ylim[1]) / 2.0
        half = min_span_x / 2.0
        ax.set_ylim(center - half, center + half)
    if min_span_z is not None and min_span_z > 0 and span_z < min_span_z:
        center = (cur_zlim[0] + cur_zlim[1]) / 2.0
        half = min_span_z / 2.0
        ax.set_zlim(center - half, center + half)

    if tick_y is not None and tick_y > 0:
        ax.xaxis.set_major_locator(MultipleLocator(tick_y))
    if tick_x is not None and tick_x > 0:
        ax.yaxis.set_major_locator(MultipleLocator(tick_x))
    if tick_z is not None and tick_z > 0:
        ax.zaxis.set_major_locator(MultipleLocator(tick_z))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    ax.set_box_aspect((max(xlim[1] - xlim[0], 1e-6), max(ylim[1] - ylim[0], 1e-6), max(zlim[1] - zlim[0], 1e-6)))
    ax.set_title(f"3D Trajectory with Yaw Direction\nInstruction: {instruction}")
    ax.set_xlabel("Y (right)")
    ax.set_ylabel("X (forward)")
    ax.set_zlabel("Z (up)")
    ax.legend()
    ax.grid(True)
    ax.view_init(elev=30, azim=-100)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def world_to_local(target: List[float], init_pos: List[float]) -> List[float]:
    x0, y0, z0 = init_pos[0:3]
    yaw0 = init_pos[4]
    dx = target[0] - x0
    dy = target[1] - y0
    dz = target[2] - z0
    theta = -np.radians(yaw0)
    x_rel = dx * np.cos(theta) - dy * np.sin(theta)
    y_rel = dx * np.sin(theta) + dy * np.cos(theta)
    return [x_rel, y_rel, dz]


# Call VLA server /predict with image + proprio + instruction.
# Input contract (client -> server):
# - image: base64 PNG (RGB), resized to 224x224
# - proprio: [x, y, z, yaw_deg, pitch_deg] in local first-frame coordinates
# - instr: natural-language instruction
# Output contract (server -> client):
# - action: list of poses; each pose has at least 5 values [dx, dy, dz, dyaw_rad, dpitch_rad]
def send_prediction_request(image: Image.Image, proprio: List[float], instr: str, server_url: str) -> Optional[Dict[str, Any]]:
    img_io = BytesIO()
    if image.size != IMG_INPUT_SIZE:
        image = image.resize(IMG_INPUT_SIZE)
    image.save(img_io, format="PNG")
    img_data = img_io.getvalue()
    img_base64 = base64.b64encode(img_data).decode("utf-8")
    payload: Dict[str, Any] = {
        "image": img_base64,
        "proprio": proprio,
        "instr": instr,
    }
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(
            server_url,
            data=json.dumps(payload),
            headers=headers,
            # proxies={"http": None, "https": None},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.error(f"Request failed: {exc}")
        return None


def reset_model(server_url: str) -> None:
    try:
        resp = requests.post(
            server_url.replace("/predict", "/reset"),
            proxies={"http": None, "https": None},
            timeout=10,
        )
        logger.info(f"Model reset response: {resp.status_code}")
    except Exception as exc:
        logger.error(f"Model reset failed: {exc}")


# Normalize degrees to [-180, 180] for yaw math.
def normalize_angle_deg(angle: float) -> float:
    angle = angle % 360.0
    if angle > 180.0:
        angle -= 360.0
    return angle


# Set AirSim vehicle pose directly (Computer Vision / kinematic control).
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


# Read AirSim vehicle pose (world frame, yaw in degrees).
def get_vehicle_pose(client: airsim.VehicleClient) -> List[float]:
    pose = client.simGetVehiclePose()
    pitch, roll, yaw = airsim.to_eularian_angles(pose.orientation)
    return [pose.position.x_val, pose.position.y_val, pose.position.z_val, math.degrees(yaw)]


# Convert world pose to local (init-frame) pose.
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


# Rotate local XY delta into world frame using initial yaw.
def local_delta_to_world(dx: float, dy: float, yaw0_deg: float) -> Tuple[float, float]:
    theta = math.radians(yaw0_deg)
    wx = dx * math.cos(theta) - dy * math.sin(theta)
    wy = dx * math.sin(theta) + dy * math.cos(theta)
    return wx, wy


# Fetch an RGB frame from AirSim camera.
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
    # AirSim returns BGR; convert to RGB for consistent saving.
    return img[:, :, :3][:, :, ::-1]


# Set camera pitch (degrees) relative to vehicle.
def set_camera_pitch(client: airsim.VehicleClient, camera_name: str, pitch_deg: float) -> None:
    cam_pose = airsim.Pose(
        airsim.Vector3r(0.0, 0.0, 0.0),
        airsim.to_quaternion(math.radians(pitch_deg), 0.0, 0.0),
    )
    client.simSetCameraPose(camera_name, cam_pose)


# Optional: place scene objects if log has object metadata.
def create_obj_if_needed(client: airsim.VehicleClient, obj_info: Optional[Dict[str, Any]]) -> None:
    if obj_info is None:
        return
    use_obj = obj_info.get("use_obj", None)
    obj_id = obj_info.get("obj_id", None)
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


# Replay a raw trajectory: set pose each step, capture image, and log local pose.
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
    trajectory_log: List[Dict[str, Any]],
) -> None:
    os.makedirs(frame_dir, exist_ok=True)
    # local frame: x, y, z, yaw (deg), pitch (deg) relative to the first frame
    current_pose = [0.0, 0.0, 0.0, 0.0, 0.0]
    last_pose: Optional[List[float]] = None
    small_count = 0
    step_count = 0

    while step_count < max_steps:
        image = get_image_rgb(client, camera_name, image_type)
        if image is None:
            logger.warning("Empty image; skipping step.")
            time.sleep(frame_sleep)
            continue

        response = send_prediction_request(
            image=Image.fromarray(image),
            proprio=current_pose,
            instr=instruction,
            server_url=server_url,
        )
        if response is None:
            logger.warning("No valid response, ending control loop.")
            break

        action_poses = response.get("action")
        if not isinstance(action_poses, list) or len(action_poses) == 0:
            logger.warning("Response 'action' is empty or invalid, stopping.")
            break

        for i, action_pose in enumerate(action_poses):
            if not (isinstance(action_pose, (list, tuple)) and len(action_pose) >= 5):
                logger.warning(f"Invalid action element at {i}: {action_pose}")
                continue

            # Model output is first-local-frame delta; yaw is in radians.
            rel_x = float(action_pose[0])
            rel_y = float(action_pose[1])
            rel_z = float(action_pose[2])  
            rel_yaw_deg = normalize_angle_deg(math.degrees(float(action_pose[3])))
            rel_pit_deg = normalize_angle_deg(math.degrees(float(action_pose[4])))

            # VLA outputs relative displacement in cm; convert to meters for AirSim.
            rel_x_m = rel_x # * ACTION_SCALE
            rel_y_m = rel_y # * ACTION_SCALE
            rel_z_m = rel_z # * ACTION_SCALE

            # Convert relative-to-start action into world pose.
            global_x, global_y = local_delta_to_world(rel_x_m, rel_y_m, init_yaw_deg)
            global_yaw = normalize_angle_deg(rel_yaw_deg + init_yaw_deg)
            global_pit = normalize_angle_deg(rel_pit_deg + init_pitch_deg)
            
            pos_world = [
                init_pose_world[0] + global_x * pos_scale,
                init_pose_world[1] + global_y * pos_scale,
                init_pose_world[2] - rel_z_m * pos_scale,   # 将模型的左手系z轴转换为NED的z轴
            ]
            set_vehicle_pose(client, pos_world, global_yaw)
            set_camera_pitch(client, camera_name, global_pit)
            time.sleep(max(frame_sleep, 0.0))

            current_pose = [rel_x, rel_y, rel_z, rel_yaw_deg, rel_pit_deg]  # current pose 是基于第一帧的左手系
            # print(f"current pose: {current_pose}")
            pose_world = [pos_world[0], pos_world[1], pos_world[2], global_yaw, global_pit]
            local_pose = world_to_local_pose(pose_world, init_pose_world, init_yaw_deg, init_pitch_deg)
            trajectory_log.append({"state": [local_pose[:3], [local_pose[4], local_pose[3], 0]]})
            step_count += 1

            # Save observation frame after each applied action.
            image = get_image_rgb(client, camera_name, image_type)
            if image is not None:
                img_path = os.path.join(frame_dir, f"{step_count:06d}.png")
                Image.fromarray(image).save(img_path)

            if last_pose is not None:
                diffs = [abs(a - b) for a, b in zip(current_pose, last_pose)]
                if (
                    all(d < ACTION_SMALL_DELTA_POS * ACTION_SCALE for d in diffs[:3])
                    and diffs[3] < ACTION_SMALL_DELTA_YAW
                    and diffs[4] < ACTION_SMALL_DELTA_PITCH
                ):
                    small_count += 1
                else:
                    small_count = 0
                if small_count >= ACTION_SMALL_STEPS:
                    logger.info(f"Detected small change for {ACTION_SMALL_STEPS} steps, ending task.")
                    return
            last_pose = current_pose

            if step_count >= max_steps:
                return

        if response.get("done") is True:
            logger.info("Server returned done=True. Ending control loop.")
            return


def ensure_no_proxy() -> None:
    os.environ["no_proxy"] = "127.0.0.1"
    os.environ["N_PROXY"] = "127.0.0.1"


if __name__ == "__main__":
    # ---- ensure no proxy -------
    ensure_no_proxy()

    # ---- CLI defaults ----
    DEFAULT_SERVER_IP = "127.0.0.1"
    DEFAULT_SERVER_PORT = 5007
    DEFAULT_MAX_STEPS = 80
    DEFAULT_TICK_X = 100
    DEFAULT_TICK_Y = 100
    DEFAULT_TICK_Z = 100
    DEFAULT_MIN_SPAN_X = 400
    DEFAULT_MIN_SPAN_Y = 400
    DEFAULT_MIN_SPAN_Z = 400
    DEFAULT_POS_SCALE = 1
    DEFAULT_Z_FLIP = False
    DEFAULT_FRAME_SLEEP = 0.02

    # ---- CLI ----
    parser = argparse.ArgumentParser(description="Batch run UAV control in UE5 via AirSim (CV mode, no gym).")
    parser.add_argument("-r", "--log_root", required=True, help="Root dir containing log.json files")
    parser.add_argument("-o", "--images_dir", required=True, help="Directory to save outputs")
    parser.add_argument("-i", "--instruction_type", default="instruction", choices=["instruction", "instruction_unified"])
    parser.add_argument("--server_ip", default=DEFAULT_SERVER_IP, help="VLA server IP")
    parser.add_argument("--server_port", type=int, default=DEFAULT_SERVER_PORT, help="VLA server port")
    parser.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS, help="Max steps per task")
    parser.add_argument("--camera_name", default="0", help="AirSim camera name")
    parser.add_argument("--image_type", default="Scene", help="AirSim ImageType, e.g. Scene")
    parser.add_argument("--pos_scale", type=float, default=DEFAULT_POS_SCALE, help="Scale json positions into AirSim meters")
    parser.add_argument("--frame_sleep", type=float, default=DEFAULT_FRAME_SLEEP, help="Sleep seconds per frame")
    parser.add_argument("--tick_x", type=float, default=None, help="Tick step for X (forward) axis")
    parser.add_argument("--tick_y", type=float, default=None, help="Tick step for Y (right) axis")
    parser.add_argument("--tick_z", type=float, default=None, help="Tick step for Z (up) axis")
    parser.add_argument("--min_span_x", type=float, default=None, help="Minimum span for X (forward) axis")
    parser.add_argument("--min_span_y", type=float, default=None, help="Minimum span for Y (right) axis")
    parser.add_argument("--min_span_z", type=float, default=None, help="Minimum span for Z (up) axis")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s")

    tick_x = args.tick_x if args.tick_x is not None else DEFAULT_TICK_X
    tick_y = args.tick_y if args.tick_y is not None else DEFAULT_TICK_Y
    tick_z = args.tick_z if args.tick_z is not None else DEFAULT_TICK_Z

    min_span_x = args.min_span_x if args.min_span_x is not None else DEFAULT_MIN_SPAN_X
    min_span_y = args.min_span_y if args.min_span_y is not None else DEFAULT_MIN_SPAN_Y
    min_span_z = args.min_span_z if args.min_span_z is not None else DEFAULT_MIN_SPAN_Z

    # Z axis handling: UE is Z-up; AirSim default is NED (Z-down).
    # Connect to AirSim CV-mode client.
    client = airsim.VehicleClient()
    client.confirmConnection()

    image_type = getattr(airsim.ImageType, args.image_type, airsim.ImageType.Scene)
    server_url = f"http://{args.server_ip}:{args.server_port}/predict"

    # Discover all log.json files under log_root.
    log_files = sorted(glob.glob(os.path.join(args.log_root, "**", "log.json"), recursive=True))
    logger.info(f"Detected {len(log_files)} log.json files")
    os.makedirs(args.images_dir, exist_ok=True)

    # Process each trajectory: set initial pose, replay raw_logs, save frames + plots.
    for idx, log_file in enumerate(log_files):
        base_name = os.path.basename(os.path.dirname(log_file))
        out_dir = os.path.join(args.images_dir, base_name)
        frame_dir = os.path.join(out_dir, "frames")
        img2d_path = os.path.join(out_dir, base_name + "_2d.png")
        img3d_path = os.path.join(out_dir, base_name + "_3d.png")
        if os.path.exists(img2d_path) and os.path.exists(img3d_path):
            logger.info(f"{base_name} images already exist, skipping.")
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
            float(init_raw[6])
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
                obj_info = {"use_obj": log_data["use_obj"], "obj_id": log_data["obj_id"], "obj_pos": obj_pos, "obj_rot": obj_rot}
        create_obj_if_needed(client, obj_info)

        logger.info(f"instruction: {instruction}")
        trajectory_log: List[Dict[str, Any]] = []
        control_loop_vla(
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
        )

        traj_json_path = os.path.join(out_dir, base_name + ".json")
        with open(traj_json_path, "w", encoding="utf-8") as f:
            json.dump(trajectory_log, f, indent=4)

        target_local = None
        target_pos = log_data.get("target_pos", None)
        if target_pos is not None and init_raw is not None and len(init_raw) >= 6:
            init_for_target = [init_raw[0], init_raw[1], init_raw[2], 0.0, init_raw[5]]
            target_local = world_to_local(target_pos, init_for_target)

        try:
            draw_2d_trajectory_from_log(traj_json_path, img2d_path, instruction, target_local, tick_x=tick_x, tick_y=tick_y, min_span_x=min_span_x, min_span_y=min_span_y)
            draw_3d_trajectory_from_log(traj_json_path, img3d_path, instruction, target_local, tick_x=tick_x, tick_y=tick_y, tick_z=tick_z, min_span_x=min_span_x, min_span_y=min_span_y, min_span_z=min_span_z)
            logger.info(f"Trajectory and images saved: {base_name}")
        except Exception as exc:
            logger.error(f"Plotting failed: {exc}")

        logger.info(f"===== Task {idx+1} finished =====\n")

    # ComputerVision mode doesn't require arming/api control.
