#!/usr/bin/env python3
"""
Prepare budget probe data for SFT and RL training from Sokoban rollout trajectories.

For each trajectory, at every N turns, creates a budget estimation probe:
- SFT: messages list with ground-truth assistant response (parquet)
- RL: prompt-only messages + reward_model ground_truth for rule-based reward (parquet)

Usage:
    python prepare_budget_probe.py \
        --input qwen-2.5-7b-sokoban-128.jsonl \
        --output-dir ./budget_probe_data \
        --max-tokens 32768 \
        --probe-every-n 1 \
        --margin 0.1
"""

import argparse
import json
import os
import re

import datasets
from transformers import AutoTokenizer

REWARD_PATTERN = re.compile(r"\s*\(reward:\s*[\-\d.]+\)\s*$")

SYSTEM_PROMPTS = {
    "sokoban": (
    "You're a helpful assistant. You are solving the Sokoban puzzle. "
    "Push all boxes to targets. You are given the grid and zero-indexed "
    "coordinates of the player, boxes, and targets. You can push but not "
    "pull boxes, and cannot push a box through a wall.\n"
    "Your available actions are:\n"
    "Up, Down, Left, Right\n"
    'You may output at most 3 action(s) in a single turn, separated by '
    'the action separator " || ".'
    ),
    "searchr1": (
        "You are a search agent answering questions by searching for information. "
        "Use search[<query>] to retrieve evidence and finish[<answer>] only when "
        "you are ready to submit the final answer. Output exactly one valid action."
    ),
    "swebench": (
        "You are a software engineering agent working on SWE-bench style GitHub "
        "issues. Inspect the repository, make targeted edits, run validation when "
        "useful, and submit a final patch when the issue is resolved."
    ),
    "warehouse": (
        "You are a warehouse-management planning agent. Read the current inventory, "
        "cash, demand, production, transit, and retailer state, then choose valid "
        "procurement, allocation, financing, or pass actions to keep the business "
        "operating successfully."
    ),
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["sokoban"]


def resolve_system_prompt(task_name: str, system_prompt_file: str = None,
                          system_prompt: str = None) -> str:
    if system_prompt is not None:
        return system_prompt
    if system_prompt_file:
        with open(system_prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["sokoban"])


def infer_task_success(metadata: dict) -> bool:
    for key in ("success", "rollout_success", "rollout_resolved"):
        if key in metadata:
            return bool(metadata.get(key))
    for key, value in metadata.items():
        if str(key).endswith("/success"):
            try:
                return float(value) == 1.0
            except (TypeError, ValueError):
                return bool(value)
    return False

def _build_probe_template(target_mode: str, reasoning: bool) -> str:
    """Build the per-turn probe prompt template based on ablation mode."""
    intro = (
        "Based on the provided rollout context, you are provided below information:\n"
        "1. You have completed {completed_turns} turns.\n"
        "2. Each turn, your token consumption is {turn_token_usage_text}.\n"
        "3. You need to finish the task within {max_context_window_tokens} tokens.\n"
        "\n"
        "Now, estimate:\n"
        "1. Whether you can finish the task successfully within "
        "{max_context_window_tokens} total tokens (input + output).\n"
    )
    if target_mode == "interval":
        body = (
            "2. If yes, how many additional tokens (input + output) are still needed "
            "to finish the task, starting from the next turn. Return an estimation "
            "interval: at least est_low tokens and at most est_high tokens.\n"
            "3. If no, answer \"impossible\".\n"
            "4. You should try your best to estimate whether the task can finish "
            "within budget (most important). If you think the task can finish within "
            "budget, your interval should be as tight as possible while still covering "
            "the true remaining token budget.\n"
            "\n"
            "Example:\n"
            "For a three-turn interaction, suppose only Turn 1 has been completed.\n"
            "The full interaction is:\n"
            "Turn 1: input X1 tokens, output Y1 tokens;\n"
            "Turn 2: input X2 tokens, output Y2 tokens;\n"
            "Turn 3: input X3 tokens, output Y3 tokens.\n"
            "You will receive:\n"
            "turn_token_usage_text: Turn 1: input X1 tokens, output Y1 tokens\n"
            "You should estimate:\n"
            "X2 + Y2 + X3 + Y3\n"
            "\n"
        )
        ans_ok = "<answer>[est_low, est_high]</answer>"
    else:  # point
        body = (
            "2. If yes, how many additional tokens (input + output) are still needed "
            "to finish the task, starting from the next turn. Return a single integer "
            "estimate.\n"
            "3. If no, answer \"impossible\".\n"
            "4. You should try your best to estimate whether the task can finish "
            "within budget (most important).\n"
            "\n"
            "Example:\n"
            "For a three-turn interaction, suppose only Turn 1 has been completed.\n"
            "The full interaction is:\n"
            "Turn 1: input X1 tokens, output Y1 tokens;\n"
            "Turn 2: input X2 tokens, output Y2 tokens;\n"
            "Turn 3: input X3 tokens, output Y3 tokens.\n"
            "You will receive:\n"
            "turn_token_usage_text: Turn 1: input X1 tokens, output Y1 tokens\n"
            "You should estimate:\n"
            "X2 + Y2 + X3 + Y3\n"
            "\n"
        )
        ans_ok = "<answer>NUMBER</answer>"

    if reasoning:
        spec = (
            "Output exactly one of the following:\n"
            f"<think>[YOUR THINKING]</think>{ans_ok}\n"
            "or\n"
            "<think>[YOUR THINKING]</think><answer>impossible</answer>"
        )
    else:
        spec = (
            "Output exactly one of the following:\n"
            f"{ans_ok}\n"
            "or\n"
            "<answer>impossible</answer>"
        )
    return intro + body + spec


def _make_target(remaining_tokens: int, target_mode: str, interval_type: str,
                 interval_width: float, reasoning: bool, probe_after: int,
                 tokens_used: int, avg_per_turn: int) -> tuple:
    """Returns (ground_truth, full_answer) for the non-impossible case."""
    if remaining_tokens == 0:
        gt = "0" if target_mode == "point" else "[0, 0]"
        body = gt
        if reasoning:
            return gt, (
                f"<think>All {probe_after} turns completed using {tokens_used} tokens. "
                f"The task is finished.</think><answer>{body}</answer>"
            )
        return gt, f"<answer>{body}</answer>"

    if target_mode == "point":
        gt = str(remaining_tokens)
        body = gt
    else:  # interval
        if interval_type == "percent":
            est_low = max(1, int(remaining_tokens * (1 - interval_width)))
            est_high = int(remaining_tokens * (1 + interval_width))
        else:  # fixed
            est_low = max(1, int(remaining_tokens - interval_width))
            est_high = int(remaining_tokens + interval_width)
        gt = f"[{est_low}, {est_high}]"
        body = gt

    if reasoning:
        if probe_after == 0:
            return gt, (
                f"<think>Given the initial task state and typical trajectories, "
                f"I expect the full solution will need about {remaining_tokens} "
                f"tokens.</think><answer>{body}</answer>"
            )
        return gt, (
            f"<think>I've completed {probe_after} turns using {tokens_used} tokens, "
            f"averaging about {avg_per_turn} tokens per turn. The task should be "
            f"completable within the budget.</think><answer>{body}</answer>"
        )
    return gt, f"<answer>{body}</answer>"


def _make_impossible_target(reasoning: bool, probe_after: int, tokens_used: int,
                            avg_per_turn: int, max_tokens: int) -> tuple:
    """Returns (ground_truth, full_answer) for the impossible case."""
    gt = "impossible"
    if not reasoning:
        return gt, "<answer>impossible</answer>"
    if probe_after == 0:
        return gt, (
            "<think>Looking at the initial state, the task seems hard to solve and "
            "prior rollouts have not converged.</think><answer>impossible</answer>"
        )
    return gt, (
        f"<think>I've completed {probe_after} turns using {tokens_used} tokens "
        f"({avg_per_turn} per turn). At this rate the {max_tokens} budget will be "
        f"exceeded.</think><answer>impossible</answer>"
    )


def count_tokens(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def parse_turns(messages):
    turns = []
    i = 0
    while i + 1 < len(messages):
        if messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
            turns.append({
                "user_msg": messages[i],
                "asst_msg": messages[i + 1],
            })
            i += 2
        else:
            i += 1

    trailing = None
    if i < len(messages) and messages[i]["role"] == "user":
        trailing = messages[i]
    return turns, trailing


def process_trajectory(traj, tokenizer, max_tokens, probe_every_n, margin,
                       target_mode="interval", interval_type="percent",
                       interval_width=None, reasoning=True,
                       system_prompt=SYSTEM_PROMPT):
    """Generate budget probe samples for a trajectory.

    Backward compatible: if interval_width is None and target_mode/interval_type
    are at defaults, falls back to legacy behavior using `margin`.

    Args:
        target_mode: "interval" or "point"
        interval_type: "percent" or "fixed" (only used in interval mode)
        interval_width: width parameter (e.g. 0.1 for ±10% percent, 100 for ±100 fixed).
                        If None, uses `margin` for backward compat.
        reasoning: whether to include <think>...</think> in the answer
    """
    if interval_width is None:
        interval_width = margin
    return _process_trajectory_impl(traj, tokenizer, max_tokens, probe_every_n,
                                     target_mode, interval_type, interval_width,
                                     reasoning, system_prompt)


def _process_trajectory_impl(traj, tokenizer, max_tokens, probe_every_n,
                              target_mode, interval_type, interval_width, reasoning,
                              system_prompt):
    messages = traj["messages"]
    metadata = traj.get("metadata", {})
    task_success = infer_task_success(metadata)
    turns, trailing_user = parse_turns(messages)

    total_turns = len(turns)
    if total_turns == 0:
        return []

    system_tokens = count_tokens(tokenizer, system_prompt)

    # Strip reward annotations from user messages
    for t in turns:
        t["user_msg"] = {
            **t["user_msg"],
            "content": REWARD_PATTERN.sub("", t["user_msg"]["content"]),
        }
    if trailing_user is not None:
        trailing_user = {
            **trailing_user,
            "content": REWARD_PATTERN.sub("", trailing_user["content"]),
        }

    # For output tokens, prefer recorded api_output_tokens (captures the real
    # generation including any thinking that was stripped from raw_response).
    # For input, re-tokenize the user_prompt (clean, not truncated).
    api_tokens = traj.get("per_turn_api_tokens")
    turn_tokens = []
    for i, t in enumerate(turns):
        inp = count_tokens(tokenizer, t["user_msg"]["content"])
        if api_tokens and i < len(api_tokens) and api_tokens[i].get("output") is not None:
            out = int(api_tokens[i]["output"])
        else:
            out = count_tokens(tokenizer, t["asst_msg"]["content"])
        turn_tokens.append((inp, out))

    trailing_tokens = 0
    if trailing_user is not None:
        trailing_tokens = count_tokens(tokenizer, trailing_user["content"])

    results = []

    # probe_after=0 inserts a probe BEFORE any turn (estimate total budget from
    # the initial state alone). For every probe_every_n subsequent turns we
    # insert another probe.
    probe_positions = [0] + list(range(probe_every_n, total_turns + 1, probe_every_n))

    for probe_after in probe_positions:
        tokens_used = system_tokens
        # At probe_after=0 we only include the first user message (initial state)
        # in the context, so account for its tokens too.
        if probe_after == 0:
            tokens_used += turn_tokens[0][0]
        for k in range(probe_after):
            tokens_used += turn_tokens[k][0] + turn_tokens[k][1]

        remaining_tokens = 0
        if probe_after == 0:
            # Model must estimate everything except the system + first user msg
            # that it can already see: all assistant responses + all later user
            # messages (i.e. every token in the trajectory after the initial
            # state shown).
            for k in range(total_turns):
                remaining_tokens += turn_tokens[k][1]  # asst output
                if k > 0:
                    remaining_tokens += turn_tokens[k][0]  # subsequent user input
            remaining_tokens += trailing_tokens
        else:
            for k in range(probe_after, total_turns):
                remaining_tokens += turn_tokens[k][0] + turn_tokens[k][1]
            if probe_after == total_turns:
                remaining_tokens += trailing_tokens

        total_needed = tokens_used + remaining_tokens
        budget_ok = total_needed <= max_tokens

        if probe_after == 0:
            turn_token_usage_text = "(no turns completed yet)"
        else:
            usage_parts = []
            for k in range(probe_after):
                inp, out = turn_tokens[k]
                usage_parts.append(
                    f"Turn {k + 1}: input {inp} tokens, "
                    f"output {out} tokens, total {inp + out} tokens"
                )
            turn_token_usage_text = "; ".join(usage_parts)

        probe_content = _build_probe_template(target_mode, reasoning).format(
            completed_turns=probe_after,
            turn_token_usage_text=turn_token_usage_text,
            max_context_window_tokens=max_tokens,
        )

        is_possible = task_success and budget_ok
        avg_per_turn = (tokens_used // probe_after) if probe_after > 0 else tokens_used

        if (not task_success) or (not budget_ok):
            ground_truth, answer = _make_impossible_target(
                reasoning, probe_after, tokens_used, avg_per_turn, max_tokens
            )
        else:
            ground_truth, answer = _make_target(
                remaining_tokens, target_mode, interval_type, interval_width,
                reasoning, probe_after, tokens_used, avg_per_turn,
            )

        # Build message history (system + conversation turns).
        # probe_after=0 keeps only the system prompt and the first user message
        # (the initial state) so the model has something concrete to reason about.
        history = [{"role": "system", "content": system_prompt}]
        if probe_after == 0:
            history.append(turns[0]["user_msg"])
        else:
            for k in range(probe_after):
                history.append(turns[k]["user_msg"])
                history.append(turns[k]["asst_msg"])

        # VeRL's MultiTurnSFTDataset requires strict role alternation after an
        # optional system message. At probe_after=0, history already ends with
        # the initial user state, so merge the probe into that user turn instead
        # of emitting consecutive user messages.
        probe_history = [dict(msg) for msg in history]
        if probe_history and probe_history[-1]["role"] == "user":
            probe_history[-1] = {
                **probe_history[-1],
                "content": probe_history[-1]["content"].rstrip() + "\n\n" + probe_content,
            }
        else:
            probe_history.append({"role": "user", "content": probe_content})

        # SFT format: full messages including ground-truth assistant response
        sft_messages = probe_history + [
            {"role": "assistant", "content": answer},
        ]

        # RL format: prompt only (no assistant response), model generates its own
        rl_prompt = probe_history

        # Per-turn token breakdown for completed turns (empty list at probe_after=0)
        per_turn_tokens = [
            {"turn": k + 1,
             "input_tokens": turn_tokens[k][0],
             "output_tokens": turn_tokens[k][1]}
            for k in range(probe_after)
        ]

        results.append({
            "custom_id": f"{traj['custom_id']}_probe_after_turn_{probe_after}",
            "sft_messages": sft_messages,
            "rl_prompt": rl_prompt,
            "ground_truth": ground_truth,
            "remaining_tokens": remaining_tokens,
            "is_possible": is_possible,
            "per_turn_tokens": per_turn_tokens,
            "metadata": {
                **metadata,
                "probe_after_turn": probe_after,
                "total_turns": total_turns,
                "tokens_used": tokens_used,
                "remaining_tokens": remaining_tokens,
                "is_possible": is_possible,
                "margin": interval_width,
                "target_mode": target_mode,
                "interval_type": interval_type,
                "interval_width": interval_width,
                "reasoning": reasoning,
            },
        })

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate budget probe SFT/RL data from Sokoban trajectories."
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with rollout trajectories")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for parquet files")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Max context window tokens (default: 32768)")
    parser.add_argument("--probe-every-n", type=int, default=1,
                        help="Insert budget probe every N turns (default: 1)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Legacy: margin for est_low/est_high around ground "
                             "truth (default 0.1 i.e. +/-10%%). Use --interval-width "
                             "in new ablation mode.")
    parser.add_argument("--target-mode", choices=["interval", "point"],
                        default="interval",
                        help="Output target type (default: interval)")
    parser.add_argument("--interval-type", choices=["percent", "fixed"],
                        default="percent",
                        help="Interval shape: percent (multiplicative) or fixed "
                             "(absolute) — only used when --target-mode=interval")
    parser.add_argument("--interval-width", type=float, default=None,
                        help="Interval half-width (e.g. 0.1 for ±10%% percent, "
                             "100 for ±100 fixed). Defaults to --margin.")
    parser.add_argument("--reasoning", action="store_true", default=False,
                        help="Include <think>...</think> in target answers and "
                             "prompt template (default: OFF)")
    parser.add_argument("--trajectory-ids-file", type=str, default=None,
                        help="Optional file with one trajectory custom_id per line. "
                             "Only those trajectories are processed.")
    parser.add_argument("--split-name", type=str, default=None,
                        help="If set, write output to <output-dir>/<split-name>/ "
                             "with a single train.parquet (and test.parquet if "
                             "--no-train-test-split is not given). If not set, "
                             "uses legacy <output-dir>/{sft,rl}/ layout.")
    parser.add_argument("--emit", choices=["sft", "rl", "both"], default="both",
                        help="Which parquet format(s) to write (default: both). "
                             "'sft' = messages with answer; 'rl' = prompt + "
                             "ground_truth.")
    parser.add_argument("--no-train-test-split", action="store_true",
                        help="If set, write all data as train.parquet (no val "
                             "split). Useful for held-out eval sets.")
    parser.add_argument("--tokenizer", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace tokenizer to use")
    parser.add_argument("--task-name", type=str, default="sokoban",
                        choices=sorted(SYSTEM_PROMPTS.keys()),
                        help="Task prompt preset and data_source suffix")
    parser.add_argument("--system-prompt-file", type=str, default=None,
                        help="Optional file overriding the task system prompt")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Optional literal system prompt override")
    parser.add_argument("--train-ratio", type=float, default=0.9,
                        help="Train/test split ratio (default: 0.9)")
    parser.add_argument("--balance", action="store_true",
                        help="Subsample impossible samples to balance with possible")
    parser.add_argument("--balance-ratio", type=float, default=1.0,
                        help="Ratio of impossible:possible when --balance is set "
                             "(default: 1.0, i.e. 1:1)")
    parser.add_argument("--oversample-possible", type=int, default=1,
                        help="Duplicate possible samples N times to amplify signal "
                             "(default: 1, i.e. no oversampling)")
    parser.add_argument("--sft-fraction", type=float, default=0.0,
                        help="Fraction of UNIQUE trajectories reserved for SFT "
                             "warm-up; the rest go to RL. 0.0 means SFT and RL "
                             "share all trajectories (default: 0.0)")
    parser.add_argument("--possible-only", action="store_true",
                        help="Drop all impossible samples (train only on possible)")
    parser.add_argument("--min-remaining-tokens", type=int, default=0,
                        help="Drop samples whose actual remaining tokens are below "
                             "this threshold (avoids trivial [0,0] predictions). "
                             "Default 0 keeps everything.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, trust_remote_code=True
    )

    # Optional trajectory ID filter
    keep_ids = None
    if args.trajectory_ids_file:
        with open(args.trajectory_ids_file) as f:
            keep_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(keep_ids)} trajectory IDs from "
              f"{args.trajectory_ids_file}")

    interval_width = args.interval_width if args.interval_width is not None else args.margin
    system_prompt = resolve_system_prompt(
        args.task_name,
        system_prompt_file=args.system_prompt_file,
        system_prompt=args.system_prompt,
    )
    data_source = f"budget_probe_{args.task_name}"

    sft_records = []
    rl_records = []
    impossible_count = 0
    total_trajs = 0
    skipped_by_filter = 0

    with open(args.input) as fin:
        for line in fin:
            traj = json.loads(line)
            if keep_ids is not None and traj.get("custom_id") not in keep_ids:
                skipped_by_filter += 1
                continue
            total_trajs += 1
            probes = process_trajectory(
                traj, tokenizer, args.max_tokens,
                args.probe_every_n, args.margin,
                target_mode=args.target_mode,
                interval_type=args.interval_type,
                interval_width=interval_width,
                reasoning=args.reasoning,
                system_prompt=system_prompt,
            )
            for probe in probes:
                # SFT record: messages list (VeRL MultiTurnSFTDataset format)
                sft_records.append({
                    "messages": probe["sft_messages"],
                    "per_turn_tokens": probe["per_turn_tokens"],
                    "custom_id": probe["custom_id"],
                })

                # RL record: prompt + reward info (VeRL RLHFDataset format)
                rl_records.append({
                    "data_source": data_source,
                    "prompt": probe["rl_prompt"],
                    "ability": "budget_estimation",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": probe["ground_truth"],
                    },
                    "extra_info": {
                        "custom_id": probe["custom_id"],
                        "remaining_tokens": probe["remaining_tokens"],
                        "is_possible": probe["is_possible"],
                        "per_turn_tokens": probe["per_turn_tokens"],
                        "margin": args.margin,
                    },
                })

                if not probe["is_possible"]:
                    impossible_count += 1

    total_probes = len(sft_records)
    print(f"Processed {total_trajs} trajectories")
    print(f"Generated {total_probes} budget probe samples")
    print(f"  Possible: {total_probes - impossible_count}")
    print(f"  Impossible: {impossible_count}")

    if args.min_remaining_tokens > 0:
        before = len(rl_records)
        keep = [i for i, r in enumerate(rl_records)
                if r["extra_info"]["remaining_tokens"] >= args.min_remaining_tokens]
        sft_records = [sft_records[i] for i in keep]
        rl_records = [rl_records[i] for i in keep]
        print(f"Filtered remaining_tokens >= {args.min_remaining_tokens}: "
              f"{before} -> {len(rl_records)}")

    if args.possible_only:
        keep_idxs = [i for i, r in enumerate(rl_records)
                     if r["extra_info"]["is_possible"]]
        # Apply oversampling if requested
        if args.oversample_possible > 1:
            keep_idxs = keep_idxs * args.oversample_possible
        import random
        rng = random.Random(args.seed)
        rng.shuffle(keep_idxs)
        sft_records = [sft_records[i] for i in keep_idxs]
        rl_records = [rl_records[i] for i in keep_idxs]
        print(f"After --possible-only (oversample={args.oversample_possible}): "
              f"{len(rl_records)} total (all possible, "
              f"unique: {len(set(keep_idxs))})")
    elif args.balance or args.oversample_possible > 1:
        import random
        rng = random.Random(args.seed)
        possible_idxs = [i for i, r in enumerate(rl_records)
                         if r["extra_info"]["is_possible"]]
        impossible_idxs = [i for i, r in enumerate(rl_records)
                           if not r["extra_info"]["is_possible"]]

        # Oversample possible by duplication
        oversampled_possible_idxs = possible_idxs * args.oversample_possible

        if args.balance:
            target_impossible = int(len(oversampled_possible_idxs) * args.balance_ratio)
            if target_impossible < len(impossible_idxs):
                impossible_idxs = rng.sample(impossible_idxs, target_impossible)

        keep_idxs = oversampled_possible_idxs + impossible_idxs
        rng.shuffle(keep_idxs)
        sft_records = [sft_records[i] for i in keep_idxs]
        rl_records = [rl_records[i] for i in keep_idxs]
        print(f"After balancing (ratio {args.balance_ratio}, "
              f"oversample_possible={args.oversample_possible}): "
              f"{len(oversampled_possible_idxs)} possible "
              f"(unique: {len(set(oversampled_possible_idxs))}) + "
              f"{len(impossible_idxs)} impossible = {len(rl_records)} total")

    # Optionally split unique trajectories: sft_fraction for SFT, rest for RL
    # so that SFT warm-up and RL training use DISJOINT trajectories.
    if args.sft_fraction > 0:
        import random as _random
        rng2 = _random.Random(args.seed + 1)

        def traj_id(custom_id):
            return custom_id.rsplit("_probe_after_turn_", 1)[0]

        unique_trajs = sorted({traj_id(r["custom_id"]) for r in sft_records})
        rng2.shuffle(unique_trajs)
        n_sft_trajs = max(1, int(len(unique_trajs) * args.sft_fraction))
        sft_traj_set = set(unique_trajs[:n_sft_trajs])

        sft_filtered = [r for r in sft_records if traj_id(r["custom_id"]) in sft_traj_set]
        rl_filtered = [r for r in rl_records
                       if traj_id(r["extra_info"]["custom_id"]) not in sft_traj_set]
        print(f"Trajectory split (sft_fraction={args.sft_fraction}): "
              f"{n_sft_trajs}/{len(unique_trajs)} trajs -> SFT, "
              f"{len(unique_trajs) - n_sft_trajs} -> RL")
        print(f"  SFT samples: {len(sft_filtered)}, RL samples: {len(rl_filtered)}")
        sft_records = sft_filtered
        rl_records = rl_filtered

    # Drop custom_id from SFT records before writing (VeRL SFT doesn't need it)
    sft_records_out = [{k: v for k, v in r.items() if k != "custom_id"}
                       for r in sft_records]

    sft_ds = datasets.Dataset.from_list(sft_records_out)
    rl_ds = datasets.Dataset.from_list(rl_records)

    os.makedirs(args.output_dir, exist_ok=True)

    def _write(ds, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        if args.no_train_test_split or len(ds) < 10:
            ds.to_parquet(os.path.join(out_dir, "train.parquet"))
            return len(ds), 0
        sp = ds.train_test_split(test_size=1.0 - args.train_ratio, seed=42)
        sp["train"].to_parquet(os.path.join(out_dir, "train.parquet"))
        sp["test"].to_parquet(os.path.join(out_dir, "test.parquet"))
        return len(sp["train"]), len(sp["test"])

    if args.split_name:
        # New ablation mode: write to <output_dir>/<split_name>/
        out_dir = os.path.join(args.output_dir, args.split_name)
        if args.emit in ("sft", "both"):
            n_tr, n_te = _write(sft_ds, out_dir)
            print(f"\nSFT -> {out_dir}/  Train: {n_tr}, Test: {n_te}")
        if args.emit in ("rl", "both"):
            # If both, RL goes to a sibling _rl directory to avoid clobber
            rl_out = out_dir if args.emit == "rl" else out_dir + "_rl"
            n_tr, n_te = _write(rl_ds, rl_out)
            print(f"RL  -> {rl_out}/  Train: {n_tr}, Test: {n_te}")
    else:
        # Legacy layout: <output_dir>/{sft,rl}/
        if args.emit in ("sft", "both"):
            sft_dir = os.path.join(args.output_dir, "sft")
            n_tr, n_te = _write(sft_ds, sft_dir)
            print(f"\nSFT data -> {sft_dir}/  Train: {n_tr}, Test: {n_te}")
        if args.emit in ("rl", "both"):
            rl_dir = os.path.join(args.output_dir, "rl")
            n_tr, n_te = _write(rl_ds, rl_dir)
            print(f"RL data  -> {rl_dir}/  Train: {n_tr}, Test: {n_te}")


if __name__ == "__main__":
    main()
