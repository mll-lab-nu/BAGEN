# Budget RL Pipeline

This directory contains the budget-probe training loop that used to live in
`budget-rl`, wired to RAGEN local-model rollouts inside this repository.

The intended loop is:

1. Generate task trajectories with a HuggingFace model through vLLM/RAGEN.
2. Convert trajectories into budget-estimation SFT and RL datasets.
3. Run SFT warm-up on the budget-estimation probes.
4. Run GRPO with `budget_probe_reward.py`.

## Quick Start

Run a smoke rollout with a local model:

```bash
TASK=sokoban \
ROLLOUT_MODEL=Qwen/Qwen3-8B \
NUM_TRAJECTORIES=16 \
OUTPUT_JSONL=data/budget-rl/smoke/rollouts.jsonl \
bash scripts/budget-rl/run_model_rollout.sh
```

Run the full loop:

```bash
TASK=sokoban \
ROLLOUT_MODEL=Qwen/Qwen3-8B \
LEARNER_MODEL=Qwen/Qwen2.5-7B-Instruct \
NUM_TRAJECTORIES=128 \
NGPUS=8 \
TP_SIZE=4 \
bash scripts/budget-rl/run_budget_rl_pipeline.sh all
```

Run individual stages:

```bash
bash scripts/budget-rl/run_budget_rl_pipeline.sh rollout
bash scripts/budget-rl/run_budget_rl_pipeline.sh prepare
bash scripts/budget-rl/run_budget_rl_pipeline.sh sft
bash scripts/budget-rl/run_budget_rl_pipeline.sh rl
```

For SearchR1, start the retrieval server first or use mock retrieval for smoke
tests:

```bash
TASK=searchr1 SEARCH_MOCK_MODE=true bash scripts/budget-rl/run_budget_rl_pipeline.sh rollout
```

## Important Variables

- `TASK`: `sokoban` or `searchr1`.
- `ROLLOUT_MODEL`: model used to create environment trajectories.
- `LEARNER_MODEL`: model trained by SFT/RL to estimate budget feasibility.
- `EXP_BASE`: output directory for `rollouts.jsonl`, parquet data, and checkpoints.
- `SFT_ABLATION`: SFT target variant, default `sft_interval_pct30`.
- `DRY_RUN=1`: print commands without running heavy stages.

The rollout stage writes OpenAI-style `rollouts.jsonl`; the prepare stage writes:

- `splits/{sft,rl,test}_traj_ids.txt`
- `rl/{train,test}.parquet`
- `eval_test/train.parquet`
- `<SFT_ABLATION>/{train,test}.parquet`
