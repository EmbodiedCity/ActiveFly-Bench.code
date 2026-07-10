"""
Convert custom ActiveFly-style trajectory folders to LeRobot format.

Input layout:
ROOT_DIR/
  20260128_195255/
    0.png
    1.png
    ...
    log.json
  20260128_200001/
    ...

Example:
python dataset_tools/prepare_data.py \
  --root_dir ../datasets/ActiveFly_Train \
  --repo_name activefly_all_mix \
  --action_horizon 4
"""

import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR


def _numeric_png_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (0, f"{int(path.stem):012d}")
    except ValueError:
        return (1, path.stem)


def _load_episode_dirs(root_dir: Path) -> list[Path]:
    episode_dirs = sorted({path.parent for path in root_dir.rglob("log.json") if path.is_file()})
    if not episode_dirs:
        raise ValueError(f"No episode folders with log.json found under: {root_dir}")
    return episode_dirs


def _load_pngs(episode_dir: Path) -> list[Path]:
    image_paths = list(episode_dir.glob("*.png")) + list(episode_dir.glob("*.jpg")) + list(episode_dir.glob("*.jpeg"))
    pngs = sorted(image_paths, key=_numeric_png_sort_key)
    if not pngs:
        raise ValueError(f"No PNG/JPG frames found in: {episode_dir}")
    return pngs


def _validate_image_size(image_height: int | None, image_width: int | None) -> None:
    if (image_height is None) != (image_width is None):
        raise ValueError("image_height and image_width must both be set, or both be None.")
    if image_height is not None and (image_height <= 0 or image_width <= 0):
        raise ValueError("image_height and image_width must be positive.")


def _resize_with_pad(image: Image.Image, image_height: int | None, image_width: int | None) -> Image.Image:
    if image_height is None or image_width is None:
        return image

    cur_width, cur_height = image.size
    ratio = max(cur_width / image_width, cur_height / image_height)
    resized_width = max(1, int(cur_width / ratio))
    resized_height = max(1, int(cur_height / ratio))

    resized = image.resize((resized_width, resized_height), _BILINEAR)
    padded = Image.new("RGB", (image_width, image_height), color=(0, 0, 0))
    pad_left = (image_width - resized_width) // 2
    pad_top = (image_height - resized_height) // 2
    padded.paste(resized, (pad_left, pad_top))
    return padded


def _read_rgb_image(path: Path, image_height: int | None, image_width: int | None) -> np.ndarray:
    with Image.open(path) as image:
        image = _resize_with_pad(image.convert("RGB"), image_height, image_width)
        return np.asarray(image, dtype=np.uint8)


# ADDED: 只读 shape，不真正加载整张图到 numpy
def _read_image_shape(path: Path, image_height: int | None, image_width: int | None) -> tuple[int, int, int]:
    with Image.open(path) as image:
        if image_height is not None and image_width is not None:
            return (image_height, image_width, 3)
        w, h = image.size
        return (h, w, 3)


def _validate_episode_integrity(
    episode_dir: Path,
    png_count: int,
    state_count: int,
    action_count: int,
    declared_length: int | None,
) -> None:
    if png_count == 0:
        raise ValueError(f"{episode_dir}: empty episode is not allowed.")

    if not (png_count == state_count == action_count):
        raise ValueError(
            f"{episode_dir}: inconsistent lengths (images={png_count}, states={state_count}, actions={action_count}). "
            "Episodes must be complete and aligned; conversion is aborted."
        )

    if declared_length is not None and declared_length != png_count:
        raise ValueError(
            f"{episode_dir}: log.json length={declared_length}, but images/states/actions={png_count}. "
            "Episode is not self-consistent; conversion is aborted."
        )


def main(
    root_dir: str,
    *,
    repo_name: str = "your_hf_username/activefly",
    output_dir_name: str | None = None,
    fps: int = 10,
    robot_type: str = "panda",
    state_key: str = "preprocessed_logs",
    action_key: str = "relative_action",
    action_field: str = "actions",
    action_horizon: int = 4,
    image_height: int | None = 224,
    image_width: int | None = 224,
    bad_cases_json_name: str = "bad_cases.json",
    push_to_hub: bool = False,
):
    _validate_image_size(image_height, image_width)

    root = Path(root_dir)
    episode_dirs = _load_episode_dirs(root)

    output_repo_id = output_dir_name or repo_name

    first_episode = episode_dirs[0]
    first_pngs = _load_pngs(first_episode)
    first_image = _read_rgb_image(first_pngs[0], image_height, image_width)
    with (first_episode / "log.json").open("r", encoding="utf-8") as f:
        first_log = json.load(f)

    if state_key not in first_log:
        raise KeyError(f"'{state_key}' not found in {first_episode / 'log.json'}")
    if action_key not in first_log:
        raise KeyError(f"'{action_key}' not found in {first_episode / 'log.json'}")

    first_states = np.asarray(first_log[state_key], dtype=np.float32)
    first_actions = np.asarray(first_log[action_key], dtype=np.float32)
    if first_states.ndim != 2 or first_actions.ndim != 2:
        raise ValueError("state/action logs must be 2D arrays shaped as [T, D].")

    state_dim = int(first_states.shape[1])
    action_dim = int(first_actions.shape[1])
    image_shape = tuple(first_image.shape)

    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type=robot_type,
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": ["state"],
            },
            action_field: {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": [action_field],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_frames = 0
    converted_episodes = 0
    bad_cases = []

    for episode_idx, episode_dir in enumerate(episode_dirs):
        log_path = episode_dir / "log.json"
        pngs = _load_pngs(episode_dir)
        with log_path.open("r", encoding="utf-8") as f:
            log_data = json.load(f)

        if state_key not in log_data or action_key not in log_data:
            raise KeyError(
                f"Missing '{state_key}' or '{action_key}' in {log_path}"
            )

        states = np.asarray(log_data[state_key], dtype=np.float32)
        actions = np.asarray(log_data[action_key], dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError(f"{log_path} has invalid state/action shape.")

        declared_length = int(log_data["length"]) if "length" in log_data else None
        _validate_episode_integrity(
            episode_dir=episode_dir,
            png_count=len(pngs),
            state_count=states.shape[0],
            action_count=actions.shape[0],
            declared_length=declared_length,
        )

        # OpenPI action_horizon 就是 action chunk size。
        # 这里只把长度不足 action_horizon 的 episode 统一记录为 bad case。
        if len(pngs) < action_horizon:
            bad_cases.append(
                {
                    "episode_index": episode_idx,
                    "episode_name": episode_dir.name,
                    "episode_path": str(episode_dir),
                    "episode_length": len(pngs),
                    "action_horizon": action_horizon,
                    "case_type": "too_short_for_action_horizon",
                    "reason": f"episode length {len(pngs)} < action_horizon {action_horizon}",
                }
            )
            print(f"[BAD] Episode {episode_idx + 1}/{len(episode_dirs)}: {episode_dir.name} | too_short")
            continue

        # ADDED: 图片 shape 不一致也算 BAD，避免 dataset.add_frame 直接崩
        shape_bad = None
        for img_path in pngs:
            actual_shape = _read_image_shape(img_path, image_height, image_width)
            if actual_shape != image_shape:
                shape_bad = {
                    "image_path": str(img_path),
                    "actual_shape": list(actual_shape),
                    "expected_shape": list(image_shape),
                }
                break

        if shape_bad is not None:
            bad_cases.append(
                {
                    "episode_index": episode_idx,
                    "episode_name": episode_dir.name,
                    "episode_path": str(episode_dir),
                    "episode_length": len(pngs),
                    "case_type": "image_shape_mismatch",
                    "reason": "image shape does not match LeRobot dataset feature shape",
                    **shape_bad,
                }
            )
            print(
                f"[BAD] Episode {episode_idx + 1}/{len(episode_dirs)}: "
                f"{episode_dir.name} | image_shape_mismatch "
                f"{shape_bad['actual_shape']} != {shape_bad['expected_shape']}"
            )
            continue

        task = str(log_data.get("instruction", ""))

        # ADDED: add_frame/save_episode 失败也记 BAD，不中断全流程
        try:
            for i in range(len(pngs)):
                dataset.add_frame(
                    {
                        "image": _read_rgb_image(pngs[i], image_height, image_width),
                        "state": states[i],
                        action_field: actions[i],
                        "task": task,
                    }
                )
            dataset.save_episode()
        except Exception as exc:
            if hasattr(dataset, "clear_episode_buffer"):
                dataset.clear_episode_buffer()

            bad_cases.append(
                {
                    "episode_index": episode_idx,
                    "episode_name": episode_dir.name,
                    "episode_path": str(episode_dir),
                    "episode_length": len(pngs),
                    "case_type": "add_frame_or_save_failed",
                    "reason": str(exc),
                    "exception_type": type(exc).__name__,
                    "expected_image_shape": list(image_shape),
                }
            )
            print(
                f"[BAD] Episode {episode_idx + 1}/{len(episode_dirs)}: "
                f"{episode_dir.name} | add_frame_or_save_failed: {exc}"
            )
            continue

        total_frames += len(pngs)
        converted_episodes += 1
        print(f"[OK] Episode {episode_idx + 1}/{len(episode_dirs)}: {episode_dir.name}")

    bad_cases_json_path = output_path / bad_cases_json_name
    with bad_cases_json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "root_dir": str(root),
                "repo_name": repo_name,
                "output_repo_id": output_repo_id,
                "image_shape": list(image_shape),
                "image_height": image_height,
                "image_width": image_width,
                "state_dim": state_dim,
                "action_dim": action_dim,
                "action_horizon": action_horizon,
                "total_episodes": len(episode_dirs),
                "converted_episodes": converted_episodes,
                "bad_case_count": len(bad_cases),
                "bad_cases": bad_cases,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    if converted_episodes == 0:
        raise ValueError(
            f"No valid episodes left after bad case filtering. "
            f"Bad case report saved to: {bad_cases_json_path}"
        )

    if push_to_hub:
        dataset.push_to_hub(
            tags=["custom", "activefly", "lerobot"],
            private=False,
            push_videos=True,
        )

    print(
        f"[Done] Converted dataset saved to: {output_path} "
        f"(episodes={converted_episodes}/{len(episode_dirs)}, "
        f"total_frames={total_frames}, bad_cases={len(bad_cases)})"
    )
    print(f"[Done] Bad case report saved to: {bad_cases_json_path}")


if __name__ == "__main__":
    tyro.cli(main)
