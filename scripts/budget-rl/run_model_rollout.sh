#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

TASK="${1:-${TASK:-sokoban}}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="$(printf '%s' "$ROLLOUT_MODEL" | tr '/:.' '---' | tr -cs 'A-Za-z0-9_-' '-')"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl}"
RUN_NAME="${RUN_NAME:-${TASK}_${MODEL_SLUG}_rollout}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/$RUN_NAME}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUTPUT_DIR/rollouts.jsonl}"
if [[ "$OUTPUT_JSONL" != /* ]]; then
  OUTPUT_JSONL="$PROJECT_ROOT/$OUTPUT_JSONL"
fi

NUM_TRAJECTORIES="${NUM_TRAJECTORIES:-128}"
VAL_GROUP_SIZE="${VAL_GROUP_SIZE:-1}"
VAL_GROUPS="${VAL_GROUPS:-}"
if [[ -z "$VAL_GROUPS" ]]; then
  VAL_GROUPS=$(((NUM_TRAJECTORIES + VAL_GROUP_SIZE - 1) / VAL_GROUP_SIZE))
fi
ACTUAL_TRAJECTORIES=$((VAL_GROUPS * VAL_GROUP_SIZE))

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
RESPONSE_LENGTH="${RESPONSE_LENGTH:-800}"
TEMPERATURE="${TEMPERATURE:-0.5}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
DO_SAMPLE="${DO_SAMPLE:-true}"
MAX_TURN="${MAX_TURN:-10}"
if [[ -z "${MAX_ACTIONS_PER_TURN+x}" ]]; then
  case "$TASK" in
    searchr1|search|searchqa) MAX_ACTIONS_PER_TURN=1 ;;
    *) MAX_ACTIONS_PER_TURN=3 ;;
  esac
fi
if [[ -z "${MAX_ACTIONS_PER_TRAJ+x}" ]]; then
  case "$TASK" in
    searchr1|search|searchqa) MAX_ACTIONS_PER_TRAJ=10 ;;
    *) MAX_ACTIONS_PER_TRAJ=30 ;;
  esac
fi
ENV_MAX_TOKENS="${ENV_MAX_TOKENS:-$RESPONSE_LENGTH}"
DRY_RUN="${DRY_RUN:-0}"

activate_runtime() {
  if [[ "${SKIP_ENV_ACTIVATE:-0}" = "1" ]]; then
    return 0
  fi
  if [[ -n "${VENV_PATH:-}" && -f "$VENV_PATH/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH/bin/activate"
    return 0
  fi
  if [[ -n "${CONDA_ENV_NAME:-ragenv2}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [[ -n "${CONDA_BASE:-}" && -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$CONDA_BASE/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [[ -f "/sw/external/python/anaconda3/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1091
      source "/sw/external/python/anaconda3/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    fi
  fi
}

bool_hydra() {
  case "${1,,}" in
    1|true|yes|on) printf 'True' ;;
    *) printf 'False' ;;
  esac
}

quote_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

mkdir -p "$(dirname "$OUTPUT_JSONL")"
cd "$PROJECT_ROOT"
activate_runtime
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

output_dir="$(dirname "$OUTPUT_JSONL")"
output_file="$(basename "$OUTPUT_JSONL")"

declare -a cmd=(
  python -m ragen.llm_agent.agent_proxy
  --config-name eval
  "system.CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  "model_path=${ROLLOUT_MODEL}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.response_length=${RESPONSE_LENGTH}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=$(bool_hydra "$DO_SAMPLE")"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${TOP_P}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K}"
  "agent_proxy.max_turn=${MAX_TURN}"
  "agent_proxy.max_actions_per_turn=${MAX_ACTIONS_PER_TURN}"
  "es_manager.val.env_groups=${VAL_GROUPS}"
  "es_manager.val.group_size=${VAL_GROUP_SIZE}"
  "output.dir=${output_dir}"
  "output.filename=${output_file}"
  "output.format=jsonl"
  "output.append_timestamp=false"
  "output.save_pkl_backup=$(bool_hydra "${SAVE_PKL_BACKUP:-0}")"
)

case "$TASK" in
  sokoban|coord_sokoban)
    ENV_TAG="${ENV_TAG:-CoordSokoban}"
    SOKOBAN_DIM_X="${SOKOBAN_DIM_X:-6}"
    SOKOBAN_DIM_Y="${SOKOBAN_DIM_Y:-6}"
    SOKOBAN_NUM_BOXES="${SOKOBAN_NUM_BOXES:-1}"
    SOKOBAN_SEARCH_DEPTH="${SOKOBAN_SEARCH_DEPTH:-30}"
    SOKOBAN_OBSERVATION_FORMAT="${SOKOBAN_OBSERVATION_FORMAT:-grid_coord}"
    cmd+=(
      "es_manager.val.env_configs.tags=[${ENV_TAG}]"
      "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]"
      "custom_envs.${ENV_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}"
      "custom_envs.${ENV_TAG}.max_tokens=${ENV_MAX_TOKENS}"
      "custom_envs.${ENV_TAG}.env_config.dim_x=${SOKOBAN_DIM_X}"
      "custom_envs.${ENV_TAG}.env_config.dim_y=${SOKOBAN_DIM_Y}"
      "custom_envs.${ENV_TAG}.env_config.num_boxes=${SOKOBAN_NUM_BOXES}"
      "custom_envs.${ENV_TAG}.env_config.search_depth=${SOKOBAN_SEARCH_DEPTH}"
      "custom_envs.${ENV_TAG}.env_config.observation_format=${SOKOBAN_OBSERVATION_FORMAT}"
    )
    ;;
  searchr1|search|searchqa)
    ENV_TAG="${ENV_TAG:-SearchQA}"
    SEARCH_DATA_PATH="${SEARCH_DATA_PATH:-/projects/bflz/searchr1_data/data/search/train.parquet}"
    RETRIEVAL_SERVER_URL="${RETRIEVAL_SERVER_URL:-http://127.0.0.1:8000}"
    SEARCH_MOCK_MODE="${SEARCH_MOCK_MODE:-false}"
    cmd+=(
      "es_manager.val.env_configs.tags=[${ENV_TAG}]"
      "es_manager.val.env_configs.n_groups=[${VAL_GROUPS}]"
      "custom_envs.${ENV_TAG}.max_actions_per_traj=${MAX_ACTIONS_PER_TRAJ}"
      "custom_envs.${ENV_TAG}.max_tokens=${ENV_MAX_TOKENS}"
      "custom_envs.${ENV_TAG}.env_config.train_path=${SEARCH_DATA_PATH}"
      "custom_envs.${ENV_TAG}.env_config.retrieval_server_url=${RETRIEVAL_SERVER_URL}"
      "custom_envs.${ENV_TAG}.env_config.mock_mode=$(bool_hydra "$SEARCH_MOCK_MODE")"
    )
    ;;
  *)
    echo "Unsupported TASK=$TASK. Supported: sokoban, searchr1." >&2
    exit 1
    ;;
esac

if [[ -n "${EXTRA_HYDRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${EXTRA_HYDRA_ARGS})
  cmd+=("${extra_args[@]}")
fi

echo "Running local-model rollout"
echo "  task: $TASK"
echo "  model: $ROLLOUT_MODEL"
echo "  trajectories: $ACTUAL_TRAJECTORIES (${VAL_GROUPS} groups x ${VAL_GROUP_SIZE})"
echo "  output: $OUTPUT_JSONL"
quote_cmd "${cmd[@]}"

if [[ "$DRY_RUN" = "1" ]]; then
  exit 0
fi

"${cmd[@]}"

if [[ ! -f "$OUTPUT_JSONL" ]]; then
  echo "Expected rollout JSONL was not written: $OUTPUT_JSONL" >&2
  exit 2
fi
