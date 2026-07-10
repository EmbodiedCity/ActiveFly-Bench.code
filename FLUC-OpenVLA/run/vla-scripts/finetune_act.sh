#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
ACTIVEFLY_ROOT=$(CDPATH= cd -- "${REPO_ROOT}/../.." && pwd)
SITE_PACKAGES=$(python - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)

export LD_LIBRARY_PATH="${SITE_PACKAGES}/nvidia/cudnn/lib:${SITE_PACKAGES}/nvidia/cublas/lib:${SITE_PACKAGES}/torch/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# Default: repo datasets/. Override with ACTIVEFLY_FLUC_SIM_DIR.
ACTIVEFLY_FLUC_SIM_DIR="${ACTIVEFLY_FLUC_SIM_DIR:-${ACTIVEFLY_ROOT}/datasets}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${ACTIVEFLY_FLUC_SIM_DIR}/a_checkpoints}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
TRAIN_ROOT="${ACTIVEFLY_FLUC_SIM_DIR}/ActiveFly_Train"

torchrun --standalone --nnodes 1 --nproc-per-node 1 "${SCRIPT_DIR}/finetune_act.py" \
  --vla_path wangxiangyu0814/OpenVLA-UAV \
  --data_root_dir "${TRAIN_ROOT}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --adapter_tmp_dir "${REPO_ROOT}/task_output/adapter_tmp_ckpt/" \
  --lora_rank 32 \
  --batch_size 8 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla_training \
  --wandb_entity "${WANDB_ENTITY}" \
  --save_steps 1000 \
  --max_steps 30000 \
  --num_epochs 1
