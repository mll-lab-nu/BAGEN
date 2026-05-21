#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/budget-rl}"

TASK="${TASK:-sokoban}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen/Qwen3-8B}"
LEARNER_MODEL="${LEARNER_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER="${TOKENIZER:-$LEARNER_MODEL}"
SFT_ABLATION="${SFT_ABLATION:-sft_interval_pct30}"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-5}"
RL_KL="${RL_KL:-0.05}"

slug() {
  printf '%s' "$1" | tr '/:.' '---' | tr -cs 'A-Za-z0-9_-' '-'
}

ROLLOUT_SLUG="$(slug "$ROLLOUT_MODEL")"
LEARNER_SLUG="$(slug "$LEARNER_MODEL")"
EXP_NAME="${EXP_NAME:-${TASK}_${ROLLOUT_SLUG}_to_${LEARNER_SLUG}}"
EXP_BASE="${EXP_BASE:-$DATA_ROOT/$EXP_NAME}"
if [[ "$EXP_BASE" != /* ]]; then
  EXP_BASE="$PROJECT_ROOT/$EXP_BASE"
fi
ROLLOUT_JSONL="${ROLLOUT_JSONL:-$EXP_BASE/rollouts.jsonl}"
if [[ "$ROLLOUT_JSONL" != /* ]]; then
  ROLLOUT_JSONL="$PROJECT_ROOT/$ROLLOUT_JSONL"
fi
STAGES_TEXT="${1:-${STAGES:-all}}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/budget-rl/run_budget_rl_pipeline.sh [all|rollout|prepare|sft|rl|rollout,prepare,...]

This runs a local-model budget RL loop inside agent-budget-control:
  1. rollout: generate task trajectories with a HF/vLLM model via RAGEN
  2. prepare: convert trajectories into SFT and RL parquet data
  3. sft: supervised warm-up on budget-estimation probes
  4. rl: GRPO on the budget-estimation reward

Common overrides:
  TASK=sokoban
  TASK=searchr1
  ROLLOUT_MODEL=Qwen/Qwen3-8B
  LEARNER_MODEL=Qwen/Qwen2.5-7B-Instruct
  NUM_TRAJECTORIES=128
  EXP_BASE=/path/to/run
  NGPUS=8
  TP_SIZE=4
  SFT_TOTAL_EPOCHS=5
  RL_TOTAL_EPOCHS=5
  DRY_RUN=1

For SearchR1, start a retrieval server first or set SEARCH_MOCK_MODE=true.
EOF
}

contains_stage() {
  local wanted="$1"
  [[ "$STAGES_TEXT" = "all" ]] && return 0
  [[ ",${STAGES_TEXT// /,}," == *",$wanted,"* ]]
}

task_name_for_prepare() {
  case "$TASK" in
    sokoban|coord_sokoban) printf 'sokoban' ;;
    searchr1|search|searchqa) printf 'searchr1' ;;
    *) printf '%s' "$TASK" ;;
  esac
}

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
  export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl${PYTHONPATH:+:$PYTHONPATH}"
}

run_rollout() {
  ROLLOUT_MODEL="$ROLLOUT_MODEL" \
  TASK="$TASK" \
  DATA_ROOT="$DATA_ROOT" \
  OUTPUT_JSONL="$ROLLOUT_JSONL" \
  bash "$SCRIPT_DIR/run_model_rollout.sh" "$TASK"
}

run_prepare() {
  if [[ ! -f "$ROLLOUT_JSONL" ]]; then
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: would require rollout JSONL at $ROLLOUT_JSONL"
    else
      echo "Missing rollout JSONL: $ROLLOUT_JSONL" >&2
      exit 2
    fi
  fi

  local -a cmd=(
    env
    "INPUT=$ROLLOUT_JSONL"
    "BASE=$EXP_BASE"
    "SEED=${SEED:-42}"
    "SFT_FRAC=${SFT_FRAC:-0.4}"
    "RL_FRAC=${RL_FRAC:-0.5}"
    "TEST_FRAC=${TEST_FRAC:-0.1}"
    "MAX_TOKENS=${MAX_TOKENS:-8192}"
    "PROBE_EVERY_N=${PROBE_EVERY_N:-1}"
    "TOKENIZER=$TOKENIZER"
    "TASK_NAME=$(task_name_for_prepare)"
    "SFT_VARIANTS=$SFT_ABLATION"
    bash "$SCRIPT_DIR/prepare_all_ablations.sh"
  )
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_sft() {
  local -a cmd=(
    env
    "BASE=$EXP_BASE"
    "MODEL=$LEARNER_MODEL"
    "NGPUS=${SFT_NGPUS:-${NGPUS:-8}}"
    "LR=${SFT_LR:-5e-6}"
    "TOTAL_EPOCHS=$SFT_TOTAL_EPOCHS"
    "TRAIN_BS=${SFT_TRAIN_BS:-16}"
    "MICRO_BS=${SFT_MICRO_BS:-2}"
    "PROJECT_NAME=${SFT_PROJECT_NAME:-budget_probe_sft}"
    "EXPERIMENT_NAME=${SFT_EXPERIMENT_NAME:-${EXP_NAME}_sft_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}}"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$EXP_NAME}"
    "WANDB_NAME=${WANDB_NAME:-${EXP_NAME}_sft_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}}"
    bash "$SCRIPT_DIR/run_sft_ablation.sh" "$SFT_ABLATION"
  )
  if [[ -n "${SFT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${SFT_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

run_rl() {
  local sft_ckpt rl_name rl_dir
  sft_ckpt="${SFT_CKPT:-$EXP_BASE/checkpoints/$SFT_ABLATION/huggingface_e$SFT_TOTAL_EPOCHS}"
  rl_name="${RL_EXPERIMENT_NAME:-${EXP_NAME}_rl_${SFT_ABLATION}_e${SFT_TOTAL_EPOCHS}_kl$(printf '%s' "$RL_KL" | tr -d '.')}"
  rl_dir="${RL_SAVE_DIR:-$EXP_BASE/checkpoints/$rl_name}"

  if [[ ! -f "$sft_ckpt/config.json" ]]; then
    if [[ "${DRY_RUN:-0}" = "1" ]]; then
      echo "DRY_RUN: would require SFT checkpoint at $sft_ckpt/config.json"
    else
      echo "Missing SFT checkpoint: $sft_ckpt/config.json" >&2
      exit 3
    fi
  fi

  local -a cmd=(
    env
    "DATA_DIR=$EXP_BASE"
    "MODEL=$sft_ckpt"
    "NGPUS=${RL_NGPUS:-${NGPUS:-8}}"
    "TP_SIZE=${RL_TP_SIZE:-${TP_SIZE:-4}}"
    "TRAIN_BATCH_SIZE=${RL_BATCH_SIZE:-64}"
    "LR=${RL_LR:-5e-7}"
    "TOTAL_EPOCHS=${RL_TOTAL_EPOCHS:-5}"
    "ROLLOUT_N=${RL_ROLLOUT_N:-16}"
    "PROJECT_NAME=${RL_PROJECT_NAME:-budget_probe_grpo}"
    "EXPERIMENT_NAME=$rl_name"
    "WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-$EXP_NAME}"
    "WANDB_NAME=$rl_name"
    "RESUME_MODE=${RESUME_MODE:-disable}"
    bash "$SCRIPT_DIR/run_budget_probe_grpo.sh"
    "actor_rollout_ref.actor.ppo_mini_batch_size=${RL_BATCH_SIZE:-64}"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${RL_PPO_MICRO_BS:-4}"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${RL_LOGPROB_MICRO_BS:-4}"
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${RL_LOGPROB_MICRO_BS:-4}"
    "actor_rollout_ref.actor.kl_loss_coef=$RL_KL"
    "actor_rollout_ref.rollout.gpu_memory_utilization=${RL_GPU_UTIL:-0.4}"
    "actor_rollout_ref.rollout.max_model_len=${RL_MAX_MODEL_LEN:-8192}"
    "trainer.default_local_dir=$rl_dir"
    "trainer.save_freq=${RL_SAVE_FREQ:-10}"
    'actor_rollout_ref.actor.checkpoint.save_contents=["model","extra"]'
    'actor_rollout_ref.actor.checkpoint.load_contents=["model","extra"]'
  )
  if [[ -n "${RL_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    local extra=(${RL_EXTRA_ARGS})
    cmd+=("${extra[@]}")
  fi
  if [[ "${DRY_RUN:-0}" = "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  "${cmd[@]}"
}

main() {
  case "${1:-}" in
    -h|--help)
      usage
      exit 0
      ;;
  esac

  mkdir -p "$EXP_BASE"
  activate_runtime
  echo "Budget RL pipeline"
  echo "  stages: $STAGES_TEXT"
  echo "  task: $TASK"
  echo "  rollout model: $ROLLOUT_MODEL"
  echo "  learner model: $LEARNER_MODEL"
  echo "  exp base: $EXP_BASE"
  echo "  rollout jsonl: $ROLLOUT_JSONL"

  if contains_stage rollout; then
    echo
    echo "=== rollout ==="
    run_rollout
  fi
  if contains_stage prepare; then
    echo
    echo "=== prepare ==="
    run_prepare
  fi
  if contains_stage sft; then
    echo
    echo "=== sft ==="
    run_sft
  fi
  if contains_stage rl; then
    echo
    echo "=== rl ==="
    run_rl
  fi
}

main "$@"
