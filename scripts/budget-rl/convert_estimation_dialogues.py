#!/usr/bin/env python3
"""
Convert agent-budget-control "estimation_dialogues.json" rollouts into the
flat JSONL format expected by prepare_budget_probe.py.

Input (one JSON file): list of dicts with fields:
  - env_id, group_id, total_turns
  - turns: list of {turn_idx, user_prompt, raw_response, success,
                    api_input_tokens, api_output_tokens, ...}

Output JSONL: one line per dialogue:
  {
    "custom_id": "traj_<idx>",
    "messages": [user, assistant, user, assistant, ...],
    "metadata": {
      "env_id", "group_id", "num_turns", "success",
      "CoordSokoban/success": 1.0 or 0.0  (used by prepare_budget_probe.py)
    }
  }

Usage:
  python convert_estimation_dialogues.py \
      --input /workspace/agent-budget-control/results/estimation/sokoban-origin-qwen3-8b-6x6-1box-32x16/sokoban_api_eval_estimation_eval_estimation_dialogues.json \
      --output /workspace/verl-x/sokoban_qwen3-8b_6x6_1box_32x16.jsonl
"""

import argparse
import json


def _first_present(mapping, keys, default=None):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return default


def _rollout_success(item, turns):
    if "success" in item:
        return bool(item.get("success"))
    if "rollout_success" in item:
        return bool(item.get("rollout_success"))
    if "rollout_resolved" in item:
        return bool(item.get("rollout_resolved"))
    return any(bool(t.get("success")) for t in turns)


def convert(item, custom_id, max_turns=None):
    turns = item.get("turns", [])
    if max_turns is not None:
        turns = turns[:max_turns]
    messages = []
    per_turn_api_tokens = []
    for t in turns:
        user_content = _first_present(t, ("user_prompt", "prompt", "observation"), "")
        asst_content = _first_present(
            t,
            ("raw_response", "raw_generation", "response", "assistant"),
            "",
        )
        if not user_content or not asst_content:
            continue
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": asst_content})
        # IMPORTANT: raw_response in this dataset is post-processed (thinking
        # stripped). api_output_tokens reflects the REAL generation cost.
        per_turn_api_tokens.append({
            "input": t.get("api_input_tokens"),
            "output": t.get("api_output_tokens"),
        })

    tag = item.get("tag") or item.get("env_tag") or "unknown"
    any_success = _rollout_success(item, turns)

    return {
        "custom_id": custom_id,
        "messages": messages,
        "per_turn_api_tokens": per_turn_api_tokens,
        "metadata": {
            "env_id": item.get("env_id"),
            "group_id": item.get("group_id"),
            "tag": tag,
            "num_turns": len(messages) // 2,
            "total_turns_from_rollout": item.get("total_turns"),
            "api_total_tokens": item.get("api_total_tokens"),
            "api_input_tokens": item.get("api_input_tokens"),
            "api_output_tokens": item.get("api_output_tokens"),
            "success": any_success,
            "rollout_success": any_success,
            "CoordSokoban/success": 1.0 if any_success else 0.0,
            f"{tag}/success": 1.0 if any_success else 0.0,
            "max_turns_truncated_to": max_turns,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True,
                        help="Path to estimation_dialogues.json")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    parser.add_argument("--max-turns", type=int, default=None,
                        help="Truncate each trajectory to this many turns "
                             "(success label recomputed within window)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    written = 0
    success_count = 0
    with open(args.output, "w") as fout:
        for i, item in enumerate(data):
            record = convert(item, f"traj_{i}", max_turns=args.max_turns)
            if not record["messages"]:
                continue
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if record["metadata"]["success"]:
                success_count += 1

    print(f"Converted {written}/{len(data)} dialogues -> {args.output}")
    print(f"  Successful: {success_count}")
    print(f"  Failed:     {written - success_count}")


if __name__ == "__main__":
    main()
