#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)
ACTIVEFLY_ROOT=$(CDPATH= cd -- "${REPO_ROOT}/.." && pwd)

# Default: repo datasets/. Override with ACTIVEFLY_FLUC_SIM_DIR.
ACTIVEFLY_FLUC_SIM_DIR="${ACTIVEFLY_FLUC_SIM_DIR:-${ACTIVEFLY_ROOT}/datasets}"
TRAIN_ROOT="${OPENPI_ACTIVEFLY_TRAIN_ROOT:-${ACTIVEFLY_FLUC_SIM_DIR}/ActiveFly_Train}"
REPO_NAME="${OPENPI_ACTIVEFLY_REPO_NAME:-activefly_all_mix}"
CONFIG_NAME="${OPENPI_ACTIVEFLY_CONFIG:-pi05_activefly_train_lora}"
EXP_NAME="${OPENPI_ACTIVEFLY_EXP_NAME:-activefly_pi05_lora}"
CHECKPOINT_ROOT="${OPENPI_ACTIVEFLY_CHECKPOINT_ROOT:-${ACTIVEFLY_FLUC_SIM_DIR}/a_checkpoints/openpi}"
PREPARE_IMAGE_HEIGHT="${OPENPI_ACTIVEFLY_IMAGE_HEIGHT:-224}"
PREPARE_IMAGE_WIDTH="${OPENPI_ACTIVEFLY_IMAGE_WIDTH:-224}"
PYTHON_CMD="${PYTHON_BIN:-python}"

usage() {
  cat <<'EOF'
Usage:
  sh run/scripts/finetune_pi.sh [finetune_pi.py options]

Environment:
  ACTIVEFLY_FLUC_SIM_DIR         Dataset root (same as README §2).
  OPENPI_ACTIVEFLY_TRAIN_ROOT     Override the training trajectory root.
  OPENPI_ACTIVEFLY_REPO_NAME      Override the converted LeRobot repo name.
  OPENPI_ACTIVEFLY_IMAGE_HEIGHT   Converted image height. Defaults to 224.
  OPENPI_ACTIVEFLY_IMAGE_WIDTH    Converted image width. Defaults to 224.
  HF_DATASETS_CACHE               HuggingFace datasets cache. Defaults to OPENPI_DATA_HOME/hf_datasets.
  OPENPI_SKIP_PREPARE=1           Skip dataset conversion and start training.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

export ACTIVEFLY_FLUC_SIM_DIR
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${ACTIVEFLY_FLUC_SIM_DIR}}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${ACTIVEFLY_FLUC_SIM_DIR}/.cache/openpi}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${OPENPI_DATA_HOME}/hf_datasets}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"

cd "${REPO_ROOT}"

OPENPI_SRC="${REPO_ROOT}/run/src"
OPENPI_CLIENT_SRC="${REPO_ROOT}/run/packages/openpi-client/src"
export PYTHONPATH="${OPENPI_SRC}:${OPENPI_CLIENT_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${CHECKPOINT_ROOT}" "${OPENPI_DATA_HOME}" "${HF_DATASETS_CACHE}"

if [ "${OPENPI_SKIP_PREPARE:-0}" != "1" ]; then
  ${PYTHON_CMD} dataset_tools/prepare_data.py \
    --root_dir "${TRAIN_ROOT}" \
    --repo_name "${REPO_NAME}" \
    --image-height "${PREPARE_IMAGE_HEIGHT}" \
    --image-width "${PREPARE_IMAGE_WIDTH}"
fi

${PYTHON_CMD} run/scripts/finetune_pi.py "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --checkpoint-base-dir "${CHECKPOINT_ROOT}" \
  "$@"
