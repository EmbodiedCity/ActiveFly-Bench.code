#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="${EVAL_DIR:-$ROOT/eval}"
LOG_DIR="${LOG_DIR:-$ROOT/test_logs/pipeline}"
ACTIVEFLY_RESULTS_DIR="${ACTIVEFLY_RESULTS_DIR:-$EVAL_DIR/api_results}"
export ACTIVEFLY_RESULTS_DIR
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

usage() {
  cat <<'EOF'
Usage:
  ./run_pipeline.sh
  ./run_pipeline.sh --idx 0
  ACTIVEFLY_IDX=0 ./run_pipeline.sh
  ./run_pipeline.sh --data-id 20260217_154148
  DRY_RUN=1 ./run_pipeline.sh --idx 0

run_pipeline.sh starts the selected UE simulator, OpenVLA server, Pi0 server,
rollouts, OBP, EQA, FLUC, then eval collection. Use run_eval.sh when you only
want to summarize existing outputs.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

CLI_DATA_ID=""
CLI_IDX=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    all|"")
      CLI_DATA_ID=""
      shift
      ;;
    --data-id)
      CLI_DATA_ID="${2:-}"
      shift 2
      ;;
    --idx)
      CLI_IDX="${2:-}"
      shift 2
      ;;
    --dataset)
      ACTIVEFLY_IDX_DATASET="${2:-EQA}"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    -*)
      msg "unknown argument: $1"
      usage
      exit 2
      ;;
    *)
      CLI_DATA_ID="$1"
      shift
      ;;
  esac
done

msg() { printf '%s\n' "$*"; }
is_enabled() { [ "${1:-0}" = "1" ]; }

default_var() {
  local name="$1"
  local fallback="${2:-}"
  if [ -z "${!name:-}" ] && [ -n "$fallback" ]; then
    export "$name=$fallback"
  else
    export "$name=${!name:-}"
  fi
}

load_env_file() {
  local path="$1"
  [ -f "$path" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    case "$line" in
      ""|\#*) continue ;;
    esac
    local key="${line%%=*}"
    local value="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [ -z "${!key:-}" ]; then
      value="${value%%#*}"
      value="${value%"${value##*[![:space:]]}"}"
      value="${value#"${value%%[![:space:]]*}"}"
      value="${value%\"}"; value="${value#\"}"
      value="${value%\'}"; value="${value#\'}"
      export "$key=$value"
    fi
  done < "$path"
}

load_env_file "$ROOT/.env"

export ACTIVEFLY_QA_DIR="${ACTIVEFLY_QA_DIR:-$ROOT/Air-EQA}"
export ACTIVEFLY_FLUC_SIM_DIR="${ACTIVEFLY_FLUC_SIM_DIR:-$ROOT/datasets}"
export ACTIVEFLY_FLUC_REAL_DIR="${ACTIVEFLY_FLUC_REAL_DIR:-$ACTIVEFLY_FLUC_SIM_DIR/realworld_indoor}"
QA_DIR="$ACTIVEFLY_QA_DIR"
if [ -z "${ACTIVEFLY_ROLLOUT_DIR:-}" ]; then
  export ACTIVEFLY_ROLLOUT_DIR="${ACTIVEFLY_FLUC_SIM_DIR}/z_model_output/Embodied-City"
fi

# Keep API keys in .env or environment variables. Do not hard-code them here.
default_var OPENAI_API_KEY ''
default_var OPENAI_BASE_URL ''
default_var GEMINI_API_KEY ''
default_var GOOGLE_GEMINI_BASE_URL ''
default_var DASHSCOPE_API_KEY ''
default_var DASHSCOPE_BASE_URL ''
default_var ACTIVEFLY_GPT_MODEL gpt-5.4
default_var ACTIVEFLY_GEMINI_MODEL gemini-2.5-pro
default_var ACTIVEFLY_QWEN_MODEL qwen3-vl-plus
if [ -z "${GEMINI_API_KEY:-}" ] && [ -n "${GOOGLE_API_KEY:-}" ]; then
  export GEMINI_API_KEY="$GOOGLE_API_KEY"
fi
unset GOOGLE_API_KEY

DRY_RUN="${DRY_RUN:-0}"
HOST_PYTHON_BIN="${HOST_PYTHON_BIN:-python3}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
SKIP_CONDA="${SKIP_CONDA:-0}"

API_ENV="${API_ENV:-api}"
OPENVLA_SERVER_ENV="${OPENVLA_SERVER_ENV:-openvla}"
OPENVLA_ROLLOUT_ENV="${OPENVLA_ROLLOUT_ENV:-ue_sim}"
PI0_SERVER_ENV="${PI0_SERVER_ENV:-}"
PI0_ROLLOUT_ENV="${PI0_ROLLOUT_ENV:-}"

RUN_SIMULATOR="${RUN_SIMULATOR:-1}"
RUN_OPENVLA_SERVER="${RUN_OPENVLA_SERVER:-1}"
RUN_PI0_SERVER="${RUN_PI0_SERVER:-1}"
RUN_OPENVLA_ROLLOUT="${RUN_OPENVLA_ROLLOUT:-1}"
RUN_PI0_ROLLOUT="${RUN_PI0_ROLLOUT:-1}"
RUN_OBP="${RUN_OBP:-1}"
RUN_EQA="${RUN_EQA:-1}"
RUN_FLUC="${RUN_FLUC:-1}"
RUN_COLLECT_EVAL="${RUN_COLLECT_EVAL:-1}"

ACTIVEFLY_DATA_ID="${CLI_DATA_ID:-${ACTIVEFLY_DATA_ID:-}}"
ACTIVEFLY_IDX="${CLI_IDX:-${ACTIVEFLY_IDX:-}}"
ACTIVEFLY_IDX_DATASET="${ACTIVEFLY_IDX_DATASET:-EQA}"
ACTIVEFLY_SCENES="${ACTIVEFLY_SCENES:-beijing citypark}"

EQA_MODELS="${EQA_MODELS:-gemini:gt gpt:gt qwen:gt gemini:pi gpt:pi qwen:pi gemini:openvla gpt:openvla qwen:openvla}"
OBP_MODELS="${OBP_MODELS:-gemini:gt gpt:gt qwen:gt}"

OPENVLA_SERVER_IP="${OPENVLA_SERVER_IP:-127.0.0.1}"
OPENVLA_SERVER_PORT="${OPENVLA_SERVER_PORT:-5007}"
PI0_SERVER_IP="${PI0_SERVER_IP:-127.0.0.1}"
PI0_SERVER_PORT="${PI0_SERVER_PORT:-8000}"

BEIJING_SIM_CWD="${BEIJING_SIM_CWD:-$ROOT/simulators/TrafficSimulation}"
BEIJING_SIM_CMD="${BEIJING_SIM_CMD:-chmod +x TrafficSimulation/Binaries/Linux/TrafficSimulation && exec TrafficSimulation/Binaries/Linux/TrafficSimulation TrafficSimulation}"
CITYPARK_SIM_CWD="${CITYPARK_SIM_CWD:-$ROOT/simulators/AirVLN}"
CITYPARK_SIM_CMD="${CITYPARK_SIM_CMD:-chmod +x AirVLN/Binaries/Linux/AirVLN-Linux-Shipping && exec AirVLN/Binaries/Linux/AirVLN-Linux-Shipping AirVLN}"
SIMULATOR_START_SLEEP="${SIMULATOR_START_SLEEP:-5}"
AIRSIM_RPC_HOST="${AIRSIM_RPC_HOST:-127.0.0.1}"
AIRSIM_RPC_PORT="${AIRSIM_RPC_PORT:-41451}"
AIRSIM_RPC_TIMEOUT="${AIRSIM_RPC_TIMEOUT:-180}"
SIMULATOR_DISPLAY="${SIMULATOR_DISPLAY:-${DISPLAY:-}}"
if [ -z "$SIMULATOR_DISPLAY" ]; then
  if [ -S /tmp/.X11-unix/X11 ]; then
    SIMULATOR_DISPLAY=":11"
  elif [ -S /tmp/.X11-unix/X10 ]; then
    SIMULATOR_DISPLAY=":10"
  elif [ -S /tmp/.X11-unix/X0 ]; then
    SIMULATOR_DISPLAY=":0"
  fi
fi
SIMULATOR_XAUTHORITY="${SIMULATOR_XAUTHORITY:-}"
if [ -z "$SIMULATOR_XAUTHORITY" ] && [ -n "$SIMULATOR_DISPLAY" ]; then
  _display_id="${SIMULATOR_DISPLAY#:}"
  _display_id="${_display_id%%.*}"
  _sim_user="${USER:-${LOGNAME:-}}"
  if [ -n "$_sim_user" ] && [ -f "/var/opt/thinlinc/sessions/${_sim_user}/${_display_id}/Xauthority" ]; then
    SIMULATOR_XAUTHORITY="/var/opt/thinlinc/sessions/${_sim_user}/${_display_id}/Xauthority"
  elif [ -f "$HOME/.Xauthority" ]; then
    SIMULATOR_XAUTHORITY="$HOME/.Xauthority"
  fi
fi

OPENVLA_SERVER_CWD="${OPENVLA_SERVER_CWD:-$ROOT/UAV-Flow-main/OpenVLA-UAV}"
OPENVLA_SERVER_CMD="${OPENVLA_SERVER_CMD:-$PYTHON_BIN vla-scripts/run_act_server.py}"
OPENVLA_ROLLOUT_CWD="${OPENVLA_ROLLOUT_CWD:-$ROOT/UAV-Flow-main/UAV-Flow-Eval}"
OPENVLA_ROLLOUT_ARGS="${OPENVLA_ROLLOUT_ARGS:-}"

if [ -z "${PI0_SERVER_CWD:-}" ]; then
  PI0_SERVER_CWD="$ROOT/FLUC-OpenPi"
fi
PI0_POLICY_CONFIG="${PI0_POLICY_CONFIG:-pi05_activefly_lora}"
PI0_POLICY_DIR="${PI0_POLICY_DIR:-$ROOT/FLUC-OpenPi/checkpoints/pi05_activefly_lora/fly_experiment/29999}"
PI0_SERVER_CMD="${PI0_SERVER_CMD:-$UV_BIN run --no-sync scripts/serve_policy.py --port $PI0_SERVER_PORT policy:checkpoint --policy.config=$PI0_POLICY_CONFIG --policy.dir=$PI0_POLICY_DIR}"
PI0_ROLLOUT_CWD="${PI0_ROLLOUT_CWD:-$ROOT/FLUC-OpenPi}"
PI0_ROLLOUT_ARGS="${PI0_ROLLOUT_ARGS:-}"
PI0_ROLLOUT_PYTHON="${PI0_ROLLOUT_PYTHON:-$PYTHON_BIN}"
METRIC_ARGS="${METRIC_ARGS:-}"
RESUME_ARGS="${RESUME_ARGS:-}"

if ! is_enabled "$DRY_RUN"; then
  mkdir -p "$EVAL_DIR" "$LOG_DIR"
  if is_enabled "$RUN_OBP" || is_enabled "$RUN_EQA"; then
    mkdir -p "$ACTIVEFLY_RESULTS_DIR"
  fi
fi

BG_PIDS=()
BG_LABELS=()
LAST_BG_PID=""

cleanup() {
  local i pid label
  for i in "${!BG_PIDS[@]}"; do
    pid="${BG_PIDS[$i]}"
    label="${BG_LABELS[$i]}"
    if kill -0 "$pid" 2>/dev/null; then
      msg "[cleanup] stopping $label pid=$pid"
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

shell_quote() { printf '%q' "$1"; }

prepare_env() {
  local env_name="${1:-}"
  if [ "$SKIP_CONDA" = "1" ] || [ -z "$env_name" ]; then
    return 0
  fi
  if [ ! -f "$CONDA_SH" ]; then
    msg "[warn] conda.sh not found: $CONDA_SH; running without conda env $env_name"
    return 0
  fi
  # shellcheck disable=SC1090
  . "$CONDA_SH"
  if ! conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    msg "[warn] conda env not found: $env_name; running without conda activation"
    return 0
  fi
  conda activate "$env_name"
}

run_in_env() {
  local label="$1"
  local env_name="$2"
  local cwd="$3"
  local cmd="$4"
  msg "[$label] $cmd"
  if is_enabled "$DRY_RUN"; then
    msg "[dry-run] cwd=$cwd env=${env_name:-none}"
    return 0
  fi
  if [ ! -d "$cwd" ]; then
    msg "[missing cwd] $cwd"
    return 1
  fi
  (
    cd "$cwd"
    prepare_env "$env_name"
    bash -c "$cmd"
  )
}

start_bg() {
  local label="$1"
  local env_name="$2"
  local cwd="$3"
  local cmd="$4"
  local log="$LOG_DIR/${label// /_}.log"
  msg "[$label] start: $cmd"
  if is_enabled "$DRY_RUN"; then
    msg "[dry-run] cwd=$cwd env=${env_name:-none} log=$log"
    LAST_BG_PID=""
    return 0
  fi
  if [ ! -d "$cwd" ]; then
    msg "[missing cwd] $cwd"
    return 1
  fi
  (
    cd "$cwd"
    prepare_env "$env_name"
    exec setsid bash -c "$cmd"
  ) >"$log" 2>&1 &
  LAST_BG_PID="$!"
  BG_PIDS+=("$LAST_BG_PID")
  BG_LABELS+=("$label")
  msg "[$label] pid=$LAST_BG_PID log=$log"
}

stop_pid() {
  local pid="${1:-}"
  local label="${2:-process}"
  [ -n "$pid" ] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    msg "[$label] stop pid=$pid"
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
  fi
}

port_open() {
  "$HOST_PYTHON_BIN" - "$1" "$2" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1)
try:
    sock.connect((host, port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
}

wait_port() {
  local host="$1"
  local port="$2"
  local timeout="${3:-120}"
  if is_enabled "$DRY_RUN"; then
    return 0
  fi
  "$HOST_PYTHON_BIN" - "$host" "$port" "$timeout" <<'PY'
import socket, sys, time
host, port, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
deadline = time.time() + timeout
while time.time() < deadline:
    sock = socket.socket()
    sock.settimeout(1)
    try:
        sock.connect((host, port))
        sock.close()
        raise SystemExit(0)
    except OSError:
        sock.close()
        time.sleep(1)
raise SystemExit(f"timeout waiting for {host}:{port}")
PY
}

wait_airsim_rpc() {
  msg "[AirSim] waiting for RPC $AIRSIM_RPC_HOST:$AIRSIM_RPC_PORT"
  wait_port "$AIRSIM_RPC_HOST" "$AIRSIM_RPC_PORT" "$AIRSIM_RPC_TIMEOUT"
}

start_openvla_server() {
  is_enabled "$RUN_OPENVLA_SERVER" || return 0
  if port_open "$OPENVLA_SERVER_IP" "$OPENVLA_SERVER_PORT" 2>/dev/null; then
    msg "[OpenVLA server] already reachable at $OPENVLA_SERVER_IP:$OPENVLA_SERVER_PORT"
    return 0
  fi
  start_bg "OpenVLA server" "$OPENVLA_SERVER_ENV" "$OPENVLA_SERVER_CWD" "$OPENVLA_SERVER_CMD"
  wait_port "$OPENVLA_SERVER_IP" "$OPENVLA_SERVER_PORT" "${OPENVLA_SERVER_TIMEOUT:-180}"
}

start_pi0_server() {
  is_enabled "$RUN_PI0_SERVER" || return 0
  if port_open "$PI0_SERVER_IP" "$PI0_SERVER_PORT" 2>/dev/null; then
    msg "[Pi0 server] already reachable at $PI0_SERVER_IP:$PI0_SERVER_PORT"
    return 0
  fi
  start_bg "Pi0 server" "$PI0_SERVER_ENV" "$PI0_SERVER_CWD" "$PI0_SERVER_CMD"
  wait_port "$PI0_SERVER_IP" "$PI0_SERVER_PORT" "${PI0_SERVER_TIMEOUT:-180}"
}

start_simulator() {
  local scene="$1"
  local cwd cmd label
  is_enabled "$RUN_SIMULATOR" || return 0
  if ! is_enabled "$RUN_OPENVLA_ROLLOUT" && ! is_enabled "$RUN_PI0_ROLLOUT"; then
    return 0
  fi
  case "$scene" in
    beijing)
      cwd="$BEIJING_SIM_CWD"; cmd="$BEIJING_SIM_CMD"; label="UE beijing"
      ;;
    citypark)
      cwd="$CITYPARK_SIM_CWD"; cmd="$CITYPARK_SIM_CMD"; label="UE citypark"
      ;;
    *)
      msg "[simulator] unknown scene '$scene'; skip"
      LAST_BG_PID=""
      return 0
      ;;
  esac
  local sim_cmd="$cmd"
  if [ -n "$SIMULATOR_DISPLAY" ]; then
    sim_cmd="export DISPLAY=$SIMULATOR_DISPLAY; ${sim_cmd}"
  fi
  if [ -n "$SIMULATOR_XAUTHORITY" ]; then
    sim_cmd="export XAUTHORITY=$SIMULATOR_XAUTHORITY; ${sim_cmd}"
  fi
  start_bg "$label" "" "$cwd" "$sim_cmd"
  if ! is_enabled "$DRY_RUN"; then
    sleep "$SIMULATOR_START_SLEEP"
    wait_airsim_rpc
  fi
}

run_openvla_rollout_root() {
  local log_root="$1"
  local images_dir="$2"
  local cmd
  cmd="$PYTHON_BIN run_act_client.py --log_root $(shell_quote "$log_root") --images_dir $(shell_quote "$images_dir") --server_ip $(shell_quote "$OPENVLA_SERVER_IP") --server_port $(shell_quote "$OPENVLA_SERVER_PORT") $OPENVLA_ROLLOUT_ARGS"
  run_in_env "OpenVLA rollout" "$OPENVLA_ROLLOUT_ENV" "$OPENVLA_ROLLOUT_CWD" "$cmd"
}

run_pi0_rollout_root() {
  local log_root="$1"
  local images_dir="$2"
  local cmd
  cmd="$PI0_ROLLOUT_PYTHON batch_run_pi05_all_ue5_cv_pitch.py --log_root $(shell_quote "$log_root") --images_dir $(shell_quote "$images_dir") --server_ip $(shell_quote "$PI0_SERVER_IP") --server_port $(shell_quote "$PI0_SERVER_PORT") $PI0_ROLLOUT_ARGS"
  run_in_env "Pi0 rollout" "$PI0_ROLLOUT_ENV" "$PI0_ROLLOUT_CWD" "$cmd"
}

resolve_one_trajectory() {
  if [ -n "$ACTIVEFLY_IDX" ]; then
    eval "$("$HOST_PYTHON_BIN" "$ROOT/utils/resolve_trajectory.py" --dataset "$ACTIVEFLY_IDX_DATASET" --idx "$ACTIVEFLY_IDX" --format shell)"
    ACTIVEFLY_DATA_ID="$TRAJ_DATA_ID"
    ACTIVEFLY_SCENES="$TRAJ_SCENE"
    return 0
  fi
  if [ -z "$ACTIVEFLY_DATA_ID" ]; then
    return 0
  fi
  eval "$("$HOST_PYTHON_BIN" "$ROOT/utils/resolve_trajectory.py" --data-id "$ACTIVEFLY_DATA_ID" --format shell)"
  ACTIVEFLY_SCENES="$TRAJ_SCENE"
}

run_rollouts_for_scene() {
  local scene="$1"
  local sim_pid=""
  local line root_scene log_root openvla_images pi0_images

  start_simulator "$scene"
  sim_pid="$LAST_BG_PID"

  if [ -n "$ACTIVEFLY_DATA_ID" ]; then
    if is_enabled "$RUN_OPENVLA_ROLLOUT"; then
      run_openvla_rollout_root "$TRAJ_LOG_ROOT" "$TRAJ_OPENVLA_IMAGES_DIR"
    fi
    if is_enabled "$RUN_PI0_ROLLOUT"; then
      run_pi0_rollout_root "$TRAJ_LOG_ROOT" "$TRAJ_PI0_IMAGES_DIR"
    fi
    stop_pid "$sim_pid" "UE $scene"
    return 0
  fi

  while IFS=$'\t' read -r root_scene log_root openvla_images pi0_images; do
    [ -n "$log_root" ] || continue
    if is_enabled "$RUN_OPENVLA_ROLLOUT"; then
      run_openvla_rollout_root "$log_root" "$openvla_images"
    fi
    if is_enabled "$RUN_PI0_ROLLOUT"; then
      run_pi0_rollout_root "$log_root" "$pi0_images"
    fi
  done < <("$HOST_PYTHON_BIN" "$ROOT/utils/resolve_trajectory.py" --roots --scene "$scene" --format tsv)

  stop_pid "$sim_pid" "UE $scene"
}

run_rollout_pipeline() {
  if ! is_enabled "$RUN_OPENVLA_ROLLOUT" && ! is_enabled "$RUN_PI0_ROLLOUT"; then
    msg "== Rollout stages skipped =="
    return 0
  fi
  msg "== Rollout pipeline =="
  if is_enabled "$RUN_OPENVLA_ROLLOUT"; then
    start_openvla_server
  fi
  if is_enabled "$RUN_PI0_ROLLOUT"; then
    start_pi0_server
  fi
  for scene in $ACTIVEFLY_SCENES; do
    run_rollouts_for_scene "$scene"
  done
}

run_answer_stage() {
  local dataset="$1"
  local models="$2"
  local model_spec model traj idx qid cmd
  if [ -n "$ACTIVEFLY_DATA_ID" ]; then
    while IFS=$'	' read -r idx qid; do
      [ -n "$idx" ] || continue
      for model_spec in $models; do
        model="${model_spec%%:*}"
        traj="${model_spec#*:}"
        [ "$traj" = "$model_spec" ] && traj="gt"
        cmd="$PYTHON_BIN test_qa.py --model $(shell_quote "$model") --traj $(shell_quote "$traj") --dataset $(shell_quote "$dataset") --idx $(shell_quote "$idx")"
        run_in_env "$dataset $model:$traj $qid" "$API_ENV" "$QA_DIR" "$cmd"
      done
    done < <("$HOST_PYTHON_BIN" "$ROOT/utils/trajectory_items.py" --dataset "$dataset" --data-id "$ACTIVEFLY_DATA_ID")
  else
    for model_spec in $models; do
      model="${model_spec%%:*}"
      traj="${model_spec#*:}"
      [ "$traj" = "$model_spec" ] && traj="gt"
      cmd="$PYTHON_BIN test_qa.py --model $(shell_quote "$model") --traj $(shell_quote "$traj") --dataset $(shell_quote "$dataset") $RESUME_ARGS"
      run_in_env "$dataset $model:$traj" "$API_ENV" "$QA_DIR" "$cmd"
    done
  fi
}

run_fluc_stage() {
  local cmd
  if ! is_enabled "$DRY_RUN"; then
    mkdir -p "$EVAL_DIR"
  fi
  if [ -n "$ACTIVEFLY_DATA_ID" ]; then
    cmd="$PYTHON_BIN utils/run_trajectory_fluc.py --data-id $(shell_quote "$ACTIVEFLY_DATA_ID") --out $(shell_quote "$EVAL_DIR/trajectory_fluc.jsonl")"
    run_in_env "FLUC trajectory" "$API_ENV" "$ROOT" "$cmd"
  else
    cmd="$PYTHON_BIN metric_qa.py --json $METRIC_ARGS > $(shell_quote "$EVAL_DIR/eval_metric_qa.json")"
    run_in_env "FLUC metrics" "$API_ENV" "$QA_DIR" "$cmd"
  fi
}

collect_eval() {
  local cmd="$HOST_PYTHON_BIN utils/collect_eval.py --out $(shell_quote "$EVAL_DIR")"
  if [ -n "$ACTIVEFLY_DATA_ID" ]; then
    cmd="$cmd --data-id $(shell_quote "$ACTIVEFLY_DATA_ID")"
  fi
  run_in_env "collect eval" "" "$ROOT" "$cmd"
}

resolve_one_trajectory

msg "ActiveFly pipeline"
msg "  root: $ROOT"
msg "  data_id: ${ACTIVEFLY_DATA_ID:-all}"
msg "  idx: ${ACTIVEFLY_IDX:-}"
msg "  scenes: $ACTIVEFLY_SCENES"
msg "  eval: $EVAL_DIR"

run_rollout_pipeline

if is_enabled "$RUN_OBP"; then
  msg "== OBP =="
  run_answer_stage OBP "$OBP_MODELS"
fi

if is_enabled "$RUN_EQA"; then
  msg "== EQA =="
  run_answer_stage EQA "$EQA_MODELS"
fi

if is_enabled "$RUN_FLUC"; then
  msg "== FLUC =="
  run_fluc_stage
fi

if is_enabled "$RUN_COLLECT_EVAL"; then
  msg "== collect eval =="
  collect_eval
fi

msg "Done."
