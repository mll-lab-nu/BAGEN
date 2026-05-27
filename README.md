<h1 align="center">BAGEN</h1>
<h3 align="center"><em>Budget-aware evaluation and training for long-horizon agents.</em></h3>

<p align="center">
  <strong>BAGEN</strong> studies whether foundation-model agents know what they
  will spend before they finish a task.
</p>
<p align="center">
  It provides rollout logging, offline budget-estimation benchmarks, and SFT/RL
  utilities for training budget-aware agents.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Built_on-RAGEN%20%2B%20verl-111827?style=for-the-badge" alt="Built on RAGEN and verl">
  <img src="https://img.shields.io/badge/Tasks-Sokoban%20%7C%20Search--R1%20%7C%20SWE--bench%20%7C%20Warehouse-0F766E?style=for-the-badge" alt="Tasks">
  <img src="https://img.shields.io/badge/API_Eval-OpenAI%20%7C%20Anthropic%20%7C%20OpenRouter%20%7C%20Gemini-7C3AED?style=for-the-badge" alt="API evaluation providers">
</p>

## News

- **2026.05.27.** The project has been renamed to **BAGEN** for public release.
- **2026.05.27.** Public cleanup removed private manuscript drafts, generated figures, local logs, and unrelated legacy scripts.
- **2026.05.27.** Core budget-estimation scripts are kept for Sokoban, Search-R1, SWE-bench-style coding, Warehouse, and Budget-RL training.

## About

BAGEN targets progressive budget estimation in long-horizon agent rollouts. At
each prefix of an interaction, a model is asked to estimate whether the agent can
still finish within the remaining resource budget and, when feasible, how much
budget is needed.

The current public code focuses on four settings:

- **Sokoban:** token-budget estimation over interactive puzzle rollouts.
- **Search-R1:** token-budget estimation for search-agent trajectories.
- **SWE-bench-style coding:** token-budget estimation over coding-agent logs.
- **Warehouse:** multi-resource estimation over time, storage occupancy, and cumulative cost.

BAGEN builds on the RAGEN/verl codebase. The Python package directory is still
named `ragen` to preserve import paths, Hydra configs, wrappers, and training
code compatibility.

## Method

The benchmark is organized as a two-pass pipeline:

1. **Original rollout collection.** Run a task model in an environment and save
   both rollout artifacts and dialogue JSON logs.
2. **Offline budget estimation.** Replay rollout prefixes and ask an evaluator
   model to output a remaining-budget interval or `impossible`.
3. **Budget-RL training.** Convert estimation data into SFT/GRPO datasets and
   train a budget estimator with local-model rollout support.

## Repository Map

| Path | Purpose |
| --- | --- |
| `config/base.yaml`, `config/evaluate_api_llm.yaml` | Budget-control flags under `agent_proxy`. |
| `ragen/wrapper/ctx_manager_wrapper.py` | Prompt injection and dialogue logging for budget-aware rollouts. |
| `ragen/wrapper/es_manager_wrapper.py` | Budget sampling and reward shaping for turn, token, and tool-call budgets. |
| `ragen/env/token_estimation` | Offline token-budget estimation environment. |
| `ragen/env/money_estimation` | Offline Warehouse multi-resource estimation environment. |
| `scripts/evaluation-scripts/origin` | First-pass rollout collection scripts. |
| `scripts/evaluation-scripts/eval` | Second-pass API budget-estimation scripts. |
| `scripts/budget-estimation-benchmark` | Python runners for token and money estimation replay. |
| `scripts/budget-rl` | SFT and GRPO utilities for budget-estimator training. |

## Getting Started

```bash
git clone --recurse-submodules <repo-url>
cd BAGEN
conda create -n ragenv2 python=3.12 -y
conda activate ragenv2
bash scripts/setup_bagen.sh
export PYTHONPATH="$PWD:$PWD/verl"
```

For Search-R1 retrieval experiments, download or build the search index:

```bash
python scripts/download_search_index.py
```

## API Keys

API-based evaluation uses the provider selected by `MODEL_NAME`. Export only the
key required for the model you run:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
export GEMINI_API_KEY=...
export TOGETHER_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Set `DRY_RUN=1` to build prompts and validate inputs without calling an API.

## Collect Original Rollouts

**Sokoban**

```bash
MODEL_NAME=OpenAI-5.2-Instant \
VAL_GROUPS=128 \
OUTPUT_DIR="$PWD/results/estimation/sokoban-origin-gpt5.2-instant-128-main" \
bash scripts/evaluation-scripts/origin/sokoban.sh
```

**Search-R1**

```bash
bash scripts/evaluation-scripts/origin/searchr1_server.sh start

MODEL_NAME=OpenRouter-Gemini-3.1-Pro-Preview \
SEARCH_MOCK_MODE=False \
bash scripts/evaluation-scripts/origin/searchr1.sh
```

Use `status`, `logs`, or `stop` with `searchr1_server.sh` to inspect or manage
the retrieval server.

## Run Budget Estimation

**Sokoban**

```bash
INPUT_JSON="$PWD/results/estimation/sokoban-origin-gpt5.2-instant-128-main/sokoban_api_eval_estimation_eval_estimation_dialogues.json" \
MODEL_NAME=qwen/qwen3-235b-a22b-2507 \
MAX_CONTEXT_WINDOW_TOKENS=2500 \
bash scripts/evaluation-scripts/eval/sokoban.sh
```

**Search-R1**

```bash
INPUT_JSON="$PWD/results/estimation/searchr1-origin-gpt5.2-instant-128-main/search_r1_api_eval_estimation_eval_estimation_dialogues.json" \
MODEL_NAME=qwen/qwen3-235b-a22b-2507 \
MAX_CONTEXT_WINDOW_TOKENS=3500 \
bash scripts/evaluation-scripts/eval/searchr1.sh
```

**SWE-bench-style coding**

```bash
INPUT_SOURCE=/path/to/swebench-origin-rollouts \
MODEL_NAME=Claude-Opus-4.7-low-thinking \
bash scripts/evaluation-scripts/eval/swebench.sh
```

**Warehouse**

```bash
INPUT_SOURCE=/path/to/warehouse_rollouts.json \
MODEL_NAME=qwen/qwen3-235b-a22b-2507 \
BUDGET_PRESET=half-reachable \
bash scripts/evaluation-scripts/eval/warehouse.sh
```

Each eval script writes:

- `OUTPUT_JSON`: predictions, ground truth, API usage, and aggregate metrics
- `TEMP_JSON`: prompt/target pairs for inspection

Smoke-test an eval path without API calls:

```bash
DRY_RUN=1 MAX_SAMPLES=5 INPUT_JSON=/path/to/dialogues.json \
bash scripts/evaluation-scripts/eval/sokoban.sh
```

## Budget-RL Training

The SFT/GRPO pipeline for training a budget estimator lives under
`scripts/budget-rl`.

```bash
DRY_RUN=1 bash scripts/budget-rl/run_budget_rl_pipeline.sh prepare,sft,rl
```

For a real run, remove `DRY_RUN=1` and set the model, data, GPU, and checkpoint
variables:

```bash
TASK=sokoban \
ROLLOUT_MODEL=Qwen/Qwen3-8B \
LEARNER_MODEL=Qwen/Qwen2.5-7B-Instruct \
NUM_TRAJECTORIES=128 \
NGPUS=8 \
bash scripts/budget-rl/run_budget_rl_pipeline.sh all
```

## Public Release Notes

Do not commit local experiment outputs or private manuscript drafts. The release
expects these to stay outside Git:

- `results/`, `logs/`, `wandb/`, `outputs/`, `model_saving/`
- `data/`, `search_data/`, downloaded search indices, and raw Warehouse data
- local PDFs such as `Budget_NeurIPS_2026*.pdf`
- API keys, `.env` files, and machine-specific absolute paths

The Warehouse data used by the paper should be released only in anonymized form.
Do not add raw enterprise records to this repository.

## Citation

Citation information will be added with the paper release.

## License

This project follows the repository license in `LICENSE`. It includes external
submodules with their own licenses and citation requirements.
