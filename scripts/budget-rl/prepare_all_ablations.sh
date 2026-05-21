#!/bin/bash
# Generate ablation datasets for SFT-based budget probe estimator.
#
# Layout produced under $BASE:
#   splits/{sft,rl,test}_traj_ids.txt        (40/50/10 trajectory split)
#   rl/{train,test}.parquet                   (RL training data, rl format)
#   eval_test/train.parquet                   (held-out test set, rl format, no split)
#   sft_point/{train,test}.parquet            (point estimation)
#   sft_interval_pct{10,30,50}/{train,test}.parquet
#   sft_interval_fix{100,500,1000}/{train,test}.parquet
#
# All ablations share the same trajectory split (deterministic via SEED).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
INPUT="${INPUT:-${SCRIPT_DIR}/sokoban_qwen2.5-7b_6x6_1box_32x16.jsonl}"
BASE="${BASE:-${SCRIPT_DIR}/ablation_data}"
SEED="${SEED:-42}"
SFT_FRAC="${SFT_FRAC:-0.4}"
RL_FRAC="${RL_FRAC:-0.5}"
TEST_FRAC="${TEST_FRAC:-0.1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
PROBE_EVERY_N="${PROBE_EVERY_N:-1}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen2.5-7B-Instruct}"
TASK_NAME="${TASK_NAME:-sokoban}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-}"
SFT_VARIANTS="${SFT_VARIANTS:-all}"

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

mkdir -p "${BASE}/splits"

echo "=== Step 1: 3-way trajectory split (${SFT_FRAC}/${RL_FRAC}/${TEST_FRAC}) ==="
python3 - <<PY
import json, random, sys
input_path = "${INPUT}"
out_dir = "${BASE}/splits"
sft_frac, rl_frac, test_frac = ${SFT_FRAC}, ${RL_FRAC}, ${TEST_FRAC}
seed = ${SEED}

ids = []
with open(input_path) as f:
    for line in f:
        traj = json.loads(line)
        ids.append(traj["custom_id"])
print(f"loaded {len(ids)} trajectories")

rng = random.Random(seed)
rng.shuffle(ids)

n = len(ids)
n_sft = int(n * sft_frac)
n_rl  = int(n * rl_frac)
sft_ids  = ids[:n_sft]
rl_ids   = ids[n_sft:n_sft + n_rl]
test_ids = ids[n_sft + n_rl:]

for name, lst in [("sft", sft_ids), ("rl", rl_ids), ("test", test_ids)]:
    with open(f"{out_dir}/{name}_traj_ids.txt", "w") as f:
        f.write("\n".join(lst) + "\n")
    print(f"  {name}: {len(lst)} trajectories")
PY

SPLITS="${BASE}/splits"
COMMON_FLAGS=(
  --input "${INPUT}"
  --output-dir "${BASE}"
  --max-tokens "${MAX_TOKENS}"
  --probe-every-n "${PROBE_EVERY_N}"
  --tokenizer "${TOKENIZER}"
  --task-name "${TASK_NAME}"
  --seed "${SEED}"
  --min-remaining-tokens 1
)

if [ -n "${SYSTEM_PROMPT_FILE}" ]; then
  COMMON_FLAGS+=(--system-prompt-file "${SYSTEM_PROMPT_FILE}")
fi

echo
echo "=== Step 2: RL training data (interval, percent ±10%, balanced) ==="
python3 "${SCRIPT_DIR}/prepare_budget_probe.py" "${COMMON_FLAGS[@]}" \
  --trajectory-ids-file "${SPLITS}/rl_traj_ids.txt" \
  --split-name rl --emit rl \
  --target-mode interval --interval-type percent --interval-width 0.1 \
  --balance --balance-ratio 1.0 --oversample-possible 2

echo
echo "=== Step 3: held-out test set (rl format, no train/test split) ==="
python3 "${SCRIPT_DIR}/prepare_budget_probe.py" "${COMMON_FLAGS[@]}" \
  --trajectory-ids-file "${SPLITS}/test_traj_ids.txt" \
  --split-name eval_test --emit rl \
  --target-mode interval --interval-type percent --interval-width 0.1 \
  --no-train-test-split

echo
echo "=== Step 4: SFT ablations (7 variants) ==="

should_gen_sft() {
  local NAME=$1
  local VARIANT
  if [ "${SFT_VARIANTS}" = "all" ]; then
    return 0
  fi
  IFS=',' read -r -a VARIANTS <<< "${SFT_VARIANTS}"
  for VARIANT in "${VARIANTS[@]}"; do
    if [ "${VARIANT}" = "${NAME}" ]; then
      return 0
    fi
  done
  return 1
}

gen_sft() {
  local NAME=$1 MODE=$2 ITYPE=$3 IWIDTH=$4
  if ! should_gen_sft "${NAME}"; then
    echo
    echo "--- skip ${NAME} (SFT_VARIANTS=${SFT_VARIANTS}) ---"
    return 0
  fi
  echo
  echo "--- $NAME (mode=$MODE, type=$ITYPE, width=$IWIDTH) ---"
  python3 "${SCRIPT_DIR}/prepare_budget_probe.py" "${COMMON_FLAGS[@]}" \
    --trajectory-ids-file "${SPLITS}/sft_traj_ids.txt" \
    --split-name "${NAME}" --emit sft \
    --target-mode "${MODE}" --interval-type "${ITYPE}" --interval-width "${IWIDTH}" \
    --balance --balance-ratio 1.0 --oversample-possible 2
}

gen_sft sft_point             point    percent 0.1
gen_sft sft_interval_pct10    interval percent 0.1
gen_sft sft_interval_pct30    interval percent 0.3
gen_sft sft_interval_pct50    interval percent 0.5
gen_sft sft_interval_fix100   interval fixed   100
gen_sft sft_interval_fix500   interval fixed   500
gen_sft sft_interval_fix1000  interval fixed   1000

echo
echo "=== DONE: data in ${BASE}/ ==="
ls -la "${BASE}/"
