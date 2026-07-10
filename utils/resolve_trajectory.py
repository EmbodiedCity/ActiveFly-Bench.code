#!/usr/bin/env python3
"""Resolve ActiveFly data_id values to rollout/eval paths.

The rollout scripts need log roots and output roots, while the eval scripts
consume the metric_qa.py directory layout. This helper keeps that mapping in
one place so later data-format changes do not leak into run_pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path
from typing import Any, Iterable

from activefly_eval_data import DatasetProvider
from paths import FLUC_SIM_DIR, OPENVLA_DIR, PI0_DIR, FLUC_REAL_DIR


ROOT = Path(__file__).resolve().parents[1]


def load_all_items() -> list[dict[str, Any]]:
    provider = DatasetProvider()
    return provider.load("EQA") + provider.load("OBP")


def find_item(data_id: str) -> dict[str, Any]:
    for item in load_all_items():
        if str(item.get("data_id", "")) == str(data_id):
            return item
    raise SystemExit(f"data_id not found in EQA/OBP metadata: {data_id}")


def item_by_index(dataset: str, idx: int) -> dict[str, Any]:
    dataset = dataset.upper()
    if dataset not in {"EQA", "OBP"}:
        raise SystemExit(f"unsupported dataset for --idx: {dataset}")
    data = DatasetProvider().load(dataset)
    if idx < 0 or idx >= len(data):
        raise SystemExit(f"{dataset} idx out of range: {idx} (total {len(data)})")
    return data[idx]


def real_data_map() -> dict[str, Path]:
    if not FLUC_REAL_DIR.is_dir():
        return {}
    return {p.parent.name: p.parent for p in FLUC_REAL_DIR.rglob("000000.png")}


def action_candidates(base: Path, action: str, data_id: str | None = None) -> list[Path]:
    names = [action]
    underscored = action.replace(" ", "_")
    if underscored not in names:
        names.append(underscored)
    paths = [base / name for name in names]
    if data_id:
        paths = [path / data_id for path in paths]
    return paths


def first_existing(paths: Iterable[Path], fallback: Path) -> Path:
    return next((path for path in paths if path.exists()), fallback)


def gt_log_root(item: dict[str, Any]) -> Path:
    data_id = str(item.get("data_id", ""))
    if item.get("data_simorreal") == "real":
        return real_data_map().get(data_id, FLUC_REAL_DIR / data_id)

    category = str(item.get("data_category", ""))
    scene = str(item.get("environment", ""))
    if category == "action":
        action = str(item.get("data_subcategory", ""))
        base = FLUC_SIM_DIR / "action" / "action_raw" / "action_all_mix_test" / scene
        candidates = action_candidates(base, action, data_id)
        return first_existing(candidates, candidates[0])
    return FLUC_SIM_DIR / category / "test" / scene / data_id


def gt_group_root(item: dict[str, Any]) -> Path:
    if item.get("data_simorreal") == "real":
        return FLUC_REAL_DIR
    category = str(item.get("data_category", ""))
    scene = str(item.get("environment", ""))
    if category == "action":
        action = str(item.get("data_subcategory", ""))
        base = FLUC_SIM_DIR / "action" / "action_raw" / "action_all_mix_test" / scene
        candidates = action_candidates(base, action)
        return first_existing(candidates, candidates[0])
    return FLUC_SIM_DIR / category / "test" / scene


def openvla_images_dir(item: dict[str, Any]) -> Path:
    category = str(item.get("data_category", ""))
    scene = str(item.get("environment", ""))
    if category == "action":
        action = str(item.get("data_subcategory", ""))
        base = OPENVLA_DIR / category / "cotraining_tpz" / scene
        candidates = action_candidates(base, action)
        return first_existing(candidates, candidates[0])
    return OPENVLA_DIR / category / "cotraining_tpz" / scene


def trajectory_info_from_item(item: dict[str, Any]) -> dict[str, Any]:
    data_id = str(item.get("data_id", ""))
    return {
        "data_id": data_id,
        "scene": item.get("environment", ""),
        "category": item.get("data_category", ""),
        "subcategory": item.get("data_subcategory", ""),
        "data_simorreal": item.get("data_simorreal", ""),
        "log_root": str(gt_log_root(item)),
        "openvla_images_dir": str(openvla_images_dir(item)),
        "pi0_images_dir": str(PI0_DIR),
    }


def trajectory_info(data_id: str) -> dict[str, Any]:
    return trajectory_info_from_item(find_item(data_id))


def rollout_roots(scene: str | None = None) -> list[dict[str, str]]:
    roots: dict[tuple[str, str, str], dict[str, str]] = {}
    for item in load_all_items():
        item_scene = str(item.get("environment", ""))
        if scene and item_scene != scene:
            continue
        log_root = gt_group_root(item)
        key = (item_scene, str(log_root), str(openvla_images_dir(item)))
        roots[key] = {
            "scene": item_scene,
            "category": str(item.get("data_category", "")),
            "subcategory": str(item.get("data_subcategory", "")),
            "log_root": str(log_root),
            "openvla_images_dir": str(openvla_images_dir(item)),
            "pi0_images_dir": str(PI0_DIR),
        }
    return sorted(roots.values(), key=lambda row: (row["scene"], row["category"], row["subcategory"], row["log_root"]))


def emit_shell(info: dict[str, Any]) -> None:
    key_map = {
        "data_id": "TRAJ_DATA_ID",
        "scene": "TRAJ_SCENE",
        "category": "TRAJ_CATEGORY",
        "subcategory": "TRAJ_SUBCATEGORY",
        "data_simorreal": "TRAJ_SIMORREAL",
        "log_root": "TRAJ_LOG_ROOT",
        "openvla_images_dir": "TRAJ_OPENVLA_IMAGES_DIR",
        "pi0_images_dir": "TRAJ_PI0_IMAGES_DIR",
    }
    for key, env_name in key_map.items():
        print(f"{env_name}={shlex.quote(str(info.get(key, '')))}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-id", default="", help="resolve one trajectory")
    parser.add_argument("--idx", type=int, default=None, help="resolve trajectory by dataset index")
    parser.add_argument("--dataset", choices=["EQA", "OBP"], default="EQA", help="dataset for --idx")
    parser.add_argument("--roots", action="store_true", help="list rollout roots for all matching trajectories")
    parser.add_argument("--scene", default="", help="optional scene filter for --roots")
    parser.add_argument("--format", choices=["json", "shell", "tsv"], default="json")
    args = parser.parse_args()

    if args.idx is not None:
        info = trajectory_info_from_item(item_by_index(args.dataset, args.idx))
        if args.format == "shell":
            emit_shell(info)
        elif args.format == "tsv":
            print("\t".join([info["scene"], info["log_root"], info["openvla_images_dir"], info["pi0_images_dir"]]))
        else:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.data_id:
        info = trajectory_info(args.data_id)
        if args.format == "shell":
            emit_shell(info)
        elif args.format == "tsv":
            print("\t".join([info["scene"], info["log_root"], info["openvla_images_dir"], info["pi0_images_dir"]]))
        else:
            print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0

    if args.roots:
        rows = rollout_roots(args.scene or None)
        if args.format == "tsv":
            for row in rows:
                print("\t".join([row["scene"], row["log_root"], row["openvla_images_dir"], row["pi0_images_dir"]]))
        else:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    parser.error("pass --data-id, --idx, or --roots")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
