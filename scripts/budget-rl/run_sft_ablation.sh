#!/bin/bash
# Parameterized SFT script for budget probe ablations.
#
# Usage:
#   bash run_sft_ablation.sh <ablation_name>
#
# Examples:
#   bash run_sft_ablation.sh sft_point
#   bash run_sft_ablation.sh sft_interval_pct30
#
# Reads training data from ${BASE}/<ablation_name>/{train,test}.parquet
# Writes checkpoint to ${BASE}/checkpoints/<ablation_name>/.
#
# Override env vars: NGPUS, MODEL, LR, TOTAL_EPOCHS, MICRO_BS, BASE.
set -euo pipefail
set -x

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <ablation_name>" >&2
  echo "Run 'ls ${BASE:-./ablation_data}' to see available ablations." >&2
  exit 1
fi

ABLATION="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BASE="${BASE:-${SCRIPT_DIR}/ablation_data}"
DATA_DIR="${BASE}/${ABLATION}"
SAVE_DIR="${BASE}/checkpoints/${ABLATION}"

if [ "${SKIP_ENV_ACTIVATE:-0}" != "1" ]; then
  if [ -n "${VENV_PATH:-}" ] && [ -f "${VENV_PATH}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VENV_PATH}/bin/activate"
  elif [ -n "${CONDA_ENV_NAME:-ragenv2}" ]; then
    if command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [ -n "${CONDA_BASE:-}" ] && [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    elif [ -f "/sw/external/python/anaconda3/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1091
      source "/sw/external/python/anaconda3/etc/profile.d/conda.sh"
      conda activate "${CONDA_ENV_NAME:-ragenv2}"
    fi
  fi
fi
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

if [ ! -f "${DATA_DIR}/train.parquet" ]; then
  echo "ERROR: ${DATA_DIR}/train.parquet not found." >&2
  echo "Run prepare_all_ablations.sh first." >&2
  exit 1
fi

# H200 is SM 9.0
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"

NGPUS=${NGPUS:-8}
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
LR=${LR:-5e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
MICRO_BS=${MICRO_BS:-2}
TRAIN_BS=${TRAIN_BS:-16}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","wandb"]'}
PROJECT_NAME=${PROJECT_NAME:-budget_probe_sft_ablation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-${ABLATION}}
SFT_CHECKPOINT_SAVE_CONTENTS=${SFT_CHECKPOINT_SAVE_CONTENTS:-'["model","extra"]'}
SFT_CHECKPOINT_LOAD_CONTENTS=${SFT_CHECKPOINT_LOAD_CONTENTS:-'["model","extra"]'}

mkdir -p "${SAVE_DIR}"

# Use the SFT data's own test split as VeRL val (loss curve only).
# Final apples-to-apples eval runs on ${BASE}/eval_test/ via eval_checkpoint.py.
VAL_FILES="${DATA_DIR}/test.parquet"
if [ ! -f "${VAL_FILES}" ]; then
  # Small SFT splits may have skipped train/test split — fall back to train.
  VAL_FILES="${DATA_DIR}/train.parquet"
fi

# Compute steps per epoch: train samples / batch_size, rounded down.
N_TRAIN=$(python3 -c "import datasets; print(len(datasets.load_dataset('parquet', data_files='${DATA_DIR}/train.parquet')['train']))" 2>/dev/null)
SAVE_FREQ=$(( (${N_TRAIN:-432} + TRAIN_BS - 1) / TRAIN_BS ))
if [ "$SAVE_FREQ" -lt 1 ]; then
  SAVE_FREQ=1
fi
echo "Training samples: ${N_TRAIN}, save_freq=${SAVE_FREQ} (one ckpt per epoch)"

torchrun --standalone --nnodes=1 --nproc_per_node=${NGPUS} \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.train_batch_size=${TRAIN_BS} \
    data.micro_batch_size_per_gpu=${MICRO_BS} \
    data.max_length=9216 \
    +data.ignore_input_ids_mismatch=True \
    optim.lr=${LR} \
    engine=fsdp \
    model.path=${MODEL} \
    model.use_remove_padding=True \
    trainer.default_local_dir="${SAVE_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    "trainer.logger=${TRAINER_LOGGER}" \
    trainer.resume_mode=disable \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    "checkpoint.save_contents=${SFT_CHECKPOINT_SAVE_CONTENTS}" \
    "checkpoint.load_contents=${SFT_CHECKPOINT_LOAD_CONTENTS}" "$@"

# Convert per-epoch FSDP checkpoints to HuggingFace format.
# We want epoch 2, 3, and final epoch (== TOTAL_EPOCHS).
echo "=== Converting per-epoch checkpoints -> HuggingFace ==="
latest_ckpt() {
  find "${SAVE_DIR}" -maxdepth 1 -type d -name 'global_step_*' \
    | sort -V \
    | tail -n 1
}

for E in 2 3 ${TOTAL_EPOCHS}; do
  STEP=$((SAVE_FREQ * E))
  CKPT="${SAVE_DIR}/global_step_${STEP}"
  HF_OUT="${SAVE_DIR}/huggingface_e${E}"
  if [ ! -d "${CKPT}" ] && [ "${E}" -eq "${TOTAL_EPOCHS}" ]; then
    CKPT="$(latest_ckpt)"
  fi
  if [ -d "${CKPT}" ]; then
    echo "  -> epoch ${E} (${CKPT}) -> ${HF_OUT}"
    python3 -m verl.model_merger merge \
      --backend fsdp \
      --local_dir "${CKPT}" \
      --target_dir "${HF_OUT}"
  else
    echo "  skip epoch ${E}: ${CKPT} not found"
  fi
done
