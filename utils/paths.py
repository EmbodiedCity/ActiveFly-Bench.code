"""Shared ActiveFly path resolution from environment variables."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].rstrip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


_load_env_file(ROOT / ".env")


def _clean_env(*names: str) -> Path | None:
    for name in names:
        raw = os.environ.get(name, "").strip().strip('"').strip("'")
        if raw:
            return Path(raw)
    return None


def env_path(*names: str, default: str | Path) -> Path:
    value = _clean_env(*names)
    return value if value is not None else Path(default)


QA_DIR = env_path("ACTIVEFLY_QA_DIR", default=ROOT / "Air-EQA")
FLUC_SIM_DIR = env_path("ACTIVEFLY_FLUC_SIM_DIR", default=ROOT / "datasets")
FLUC_REAL_DIR = env_path("ACTIVEFLY_FLUC_REAL_DIR", default=FLUC_SIM_DIR / "realworld_indoor")

_rollout_dir = _clean_env("ACTIVEFLY_ROLLOUT_DIR")
_pi0_dir = _clean_env("ACTIVEFLY_PI0_DIR")
_openvla_dir = _clean_env("ACTIVEFLY_OPENVLA_DIR")

if _rollout_dir is not None:
    ROLLOUT_DIR = _rollout_dir
    PI0_DIR = _pi0_dir or (_rollout_dir / "pi")
    OPENVLA_DIR = _openvla_dir or (_rollout_dir / "openvla")
else:
    _default_rollout = FLUC_SIM_DIR / "z_model_output" / "Embodied-City"
    ROLLOUT_DIR = _default_rollout
    PI0_DIR = _pi0_dir or (_default_rollout / "pi")
    OPENVLA_DIR = _openvla_dir or (_default_rollout / "openvla")
