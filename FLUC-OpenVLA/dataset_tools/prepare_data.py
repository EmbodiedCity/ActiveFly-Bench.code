import argparse
import json
from pathlib import Path

import tqdm
from datasets import load_dataset


def process_parquet_to_folder_dataset(parquet_path: Path, output_folder: Path) -> None:
    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train", streaming=True)
    trajectory_dict: dict[str, dict] = {}

    for row in tqdm.tqdm(dataset, desc=f"processing {parquet_path.name}"):
        trajectory_id = row["id"]
        frame_idx = row["frame_idx"]
        log = json.loads(row["log"])
        if trajectory_id not in trajectory_dict:
            trajectory_dict[trajectory_id] = {
                "id": trajectory_id,
                "raw_logs": log["raw_logs"],
                "preprocessed_logs": log["preprocessed_logs"],
                "instruction": log["instruction"],
                "instruction_unified": log["instruction_unified"],
                "length": len(log["preprocessed_logs"]),
                "images": {},
            }

        trajectory_dict[trajectory_id]["images"][frame_idx] = row["image"]
        if len(trajectory_dict[trajectory_id]["images"]) != trajectory_dict[trajectory_id]["length"]:
            continue

        imgs = [
            trajectory_dict[trajectory_id]["images"][key]
            for key in sorted(trajectory_dict[trajectory_id]["images"].keys())
        ]
        trajectory_dict[trajectory_id].pop("images", None)
        episode_dir = output_folder / trajectory_id
        episode_dir.mkdir(parents=True, exist_ok=True)

        for i, image in enumerate(imgs):
            img_path = episode_dir / f"{str(i).zfill(6)}.jpg"
            with img_path.open("wb") as file_obj:
                image.save(file_obj, format="JPEG")

        with (episode_dir / "log.json").open("w", encoding="utf-8") as file_obj:
            json.dump(trajectory_dict[trajectory_id], file_obj)

        trajectory_dict.pop(trajectory_id, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert UAV-Flow parquet data into trajectory folders.")
    parser.add_argument(
        "--parquet-dir",
        type=Path,
        default=None,
        help="Directory containing one or more .parquet files.",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        action="append",
        default=None,
        help="Path to a .parquet file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write trajectory folders into.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_paths: list[Path] = []

    if args.parquet_dir is not None:
        parquet_paths.extend(sorted(args.parquet_dir.glob("*.parquet")))
    if args.parquet_path is not None:
        parquet_paths.extend(args.parquet_path)

    parquet_paths = sorted({path.resolve() for path in parquet_paths})
    if not parquet_paths:
        raise ValueError("No parquet files found. Pass --parquet-dir or at least one --parquet-path.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for parquet_path in parquet_paths:
        process_parquet_to_folder_dataset(parquet_path, args.output_dir)


if __name__ == "__main__":
    main()
