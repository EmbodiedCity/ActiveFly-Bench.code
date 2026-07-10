#!/usr/bin/env python3
"""OpenPI policy server entrypoint for ActiveFly."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACTIVEFLY_ROOT = REPO_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ActiveFly OpenPI policy server.")
    parser.add_argument("--policy-dir", default=None, help="Checkpoint directory to serve.")
    parser.add_argument("--policy-config", default="pi05_activefly_train_lora", help="OpenPI training config name.")
    parser.add_argument("--exp-name", default="activefly_pi05_lora", help="Experiment name used for default checkpoints.")
    parser.add_argument("--checkpoint-step", default="29999", help="Checkpoint step used when --policy-dir is omitted.")
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument("--record", action="store_true", help="Record policy requests for debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_root = Path(
        os.environ.get("ACTIVEFLY_FLUC_SIM_DIR")
        or (ACTIVEFLY_ROOT / "datasets")
    ).resolve()
    os.environ.setdefault("ACTIVEFLY_FLUC_SIM_DIR", str(data_root))
    os.environ.setdefault("HF_LEROBOT_HOME", str(data_root))
    os.environ.setdefault("OPENPI_DATA_HOME", str(data_root / ".cache" / "openpi"))

    openpi_src = REPO_ROOT / "run" / "src"
    openpi_client_src = REPO_ROOT / "run" / "packages" / "openpi-client" / "src"
    sys.path[:0] = [str(openpi_src), str(openpi_client_src)]

    policy_dir = args.policy_dir
    if policy_dir is None:
        policy_dir = str(data_root / "a_checkpoints" / "openpi" / args.policy_config / args.exp_name / args.checkpoint_step)

    from serve_policy import Args, Checkpoint, main as serve_main

    serve_main(
        Args(
            port=args.port,
            record=args.record,
            policy=Checkpoint(config=args.policy_config, dir=policy_dir),
        )
    )


if __name__ == "__main__":
    main()
